"""ai-rm protocol v0.1 KVTransport for GPU <-> RNGD over DMA-BUF (entry point `airm.kv_transports:p2p-dmabuf`).

Wraps `RngdDmabufBackend` (completion-fenced, RNGD-initiated) with the same shape as `airm.P2PTransport`: the owning
process binds live memory per request with `bind(request_id, DmabufBinding(...))` before `plan()`. Token ranges map
to contiguous bytes of a token-major cache on both sides, so the two layouts must be identical (no conversion).
GPU <-> RNGD needs the NVIDIA open kernel module; with the proprietary module `plan()` raises TransportError.
No host-staging fallback (PROTOCOL §4.5).
"""
from collections import deque
from dataclasses import dataclass
import importlib, os, re, time, uuid

from rm_core import NoHardware, TransferHandle, TransferPlan, TransferResult, TransportError
from rm_core.metrics import Measurement

from .gpu import DmabufCopy, PAGE, RngdDmabufBackend
from .paths import PathUnavailable, resolve

DEVICE_KINDS = ('device-hbm', 'device-dram')
VENDOR_KIND = {'furiosa': 'rngd', 'nvidia': 'cuda'}
DIRECTIONS = {('cuda', 'rngd'): 'write', ('rngd', 'cuda'): 'read'}


@dataclass(frozen=True)
class DmabufBinding:
    """Live memory for one request: token 0 of the cache at RNGD DRAM `npu_offset` and at `region_offset` of the
    GPU region (a CudaRegion, or a CUDA tensor passed to `cuda_region`). The caller keeps both alive."""
    npu_offset: int
    region: object
    region_offset: int = 0


def _kind(tier):
    node, _, device = tier.accelerator_id.partition('/')
    return node, VENDOR_KIND.get(device.split(':', 1)[0])


class DmabufKVTransport:
    """Keyword arguments (all optional): `library` path of build/librngd_dmabuf.so (env HETERO_P2P_DMABUF_LIB),
    `device` RNGD PDMA path (env HETERO_P2P_DEVICE, default /dev/rngd/npu0ch0), `queue_depth`, `backend` an object
    with `copy(direction, copies, owner)` (tests), `metric_sink` a callable receiving each Measurement."""
    name = 'p2p-dmabuf'

    def __init__(self, library=None, device=None, queue_depth=4, backend=None, metric_sink=None):
        self.library = library or os.environ.get('HETERO_P2P_DMABUF_LIB')
        self.device = device or os.environ.get('HETERO_P2P_DEVICE', '/dev/rngd/npu0ch0')
        self.queue_depth, self._backend, self.metric_sink = queue_depth, backend, metric_sink
        self._bindings, self._plans, self._results = {}, {}, {}
        self.measurements = deque(maxlen=1024)

    def supports(self, src, dst):
        if src.kind not in DEVICE_KINDS or dst.kind not in DEVICE_KINDS:
            return False
        (sn, sk), (dn, dk) = _kind(src), _kind(dst)
        return (sk, dk) in DIRECTIONS and sn == dn != ''

    def bind(self, request_id, binding):
        if not isinstance(binding, DmabufBinding):
            raise TransportError(f'{self.name}: bind() needs a DmabufBinding')
        self._bindings[request_id] = binding

    def plan(self, request_id, src, dst, src_layout, dst_layout, tokens):
        binding = self._bindings.pop(request_id, None)   # one-shot, even when the plan is rejected
        if not self.supports(src, dst):
            raise TransportError(f'{self.name}: only same-host GPU <-> RNGD device memory, not {src.name} -> {dst.name}')
        src.validate(); dst.validate(); src_layout.validate(); dst_layout.validate()
        start, end = tokens
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise TransportError(f'{self.name}: bad token range {tokens}')
        if src_layout != dst_layout or src_layout.order != 'token-major':
            raise TransportError(f'{self.name}: needs identical token-major layouts; convert in the model adapter')
        pair = (_kind(src)[1], _kind(dst)[1])
        try:
            resolve(*pair)
        except PathUnavailable as e:
            raise TransportError(f'{self.name}: {e}') from e
        if binding is None:
            raise TransportError(f'{self.name}: no live memory bound for {request_id!r}; call bind() first')
        bpt = src_layout.bytes_per_token()
        offset, nbytes = start * bpt, (end - start) * bpt
        if offset % PAGE or nbytes % PAGE or binding.npu_offset % PAGE or binding.region_offset % PAGE:
            raise TransportError(f'{self.name}: token range {tokens} is not 4 KiB aligned at {bpt} bytes per token')
        region = binding.region
        if not hasattr(region, 'locate'):
            from .gpu import cuda_region
            try:
                region = cuda_region(region)
            except (PathUnavailable, ValueError) as e:
                raise TransportError(f'{self.name}: {e}') from e
        npu_tier, gpu_tier = (src, dst) if pair[0] == 'rngd' else (dst, src)
        configured = re.fullmatch(r'/dev/rngd/npu(0|[1-9][0-9]*)ch0', self.device)
        native_device = getattr(self._backend, 'device', self.device)
        if (configured is None or native_device != self.device
                or npu_tier.accelerator_id.rsplit('/', 1)[-1] != f'furiosa:{configured[1]}'
                or type(region.device) is not int or region.device < 0
                or gpu_tier.accelerator_id.rsplit('/', 1)[-1] != f'nvidia:{region.device}'):
            raise TransportError(f'{self.name}: configured device identity does not match the bound route')
        copy = DmabufCopy(binding.npu_offset + offset, region, binding.region_offset + offset, nbytes)
        if copy.region_offset + nbytes > region.size:
            raise TransportError(f'{self.name}: tokens {tokens} exceed the bound GPU region')
        plan_id = uuid.uuid4().hex
        self._plans[plan_id] = (DIRECTIONS[pair], copy, binding)
        return TransferPlan(request_id, src, dst, src_layout, dst_layout, (start, end), nbytes,
                            {'plan_id': plan_id, 'direction': DIRECTIONS[pair], 'device': self.device})

    def _get_backend(self):
        if self._backend is None:
            if not self.library:
                raise TransportError(f'{self.name}: set HETERO_P2P_DMABUF_LIB to the built librngd_dmabuf.so')
            try:
                self._backend = RngdDmabufBackend(self.library, self.device, self.queue_depth)
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                raise TransportError(f'{self.name}: backend unavailable: {e}') from e
        return self._backend

    def submit(self, plan):
        plan_id = plan.detail.get('plan_id', '')
        if plan_id not in self._plans:
            raise TransportError(f'{self.name}: unknown or already-submitted plan for {plan.request_id!r}')
        backend = self._get_backend()
        direction, copy, binding = self._plans.pop(plan_id)
        started = time.perf_counter()
        try:
            receipt = backend.copy(direction, (copy,), owner=binding)    # returns after every CQE (fence)
            result = TransferResult(plan.request_id, True, receipt['bytes_requested'], time.perf_counter() - started)
        except Exception as e:
            # The destination is uncertain: the caller must poison it, not retry on another path.
            result = TransferResult(plan.request_id, False, 0, time.perf_counter() - started, f'{type(e).__name__}: {e}')
        if result.ok:
            self._report(result)
        token = uuid.uuid4().hex
        self._results[token] = result
        return TransferHandle(plan.request_id, self.name, token)

    def wait(self, handle, timeout_s):
        result = self._results.pop(handle.token, None)
        if result is None:
            raise TransportError(f'{self.name}: unknown or already-waited handle {handle.token}')
        return result

    def _report(self, result):
        tags = {'kv_transport': self.name, 'component': 'hetero-npu-p2p'}
        rows = [Measurement('kv_transfer_ms', result.seconds * 1e3, tags)]
        if result.seconds > 0:
            rows.append(Measurement('kv_transfer_gbps', result.nbytes / result.seconds / 1e9, tags))
        for row in rows:
            self.measurements.append(row.validate())
            if self.metric_sink:
                try:
                    self.metric_sink(row)
                except Exception:
                    pass   # a metrics outage must not turn a fenced transfer into a failure

    def close(self):
        self._bindings.clear(); self._plans.clear(); self._results.clear()

    def conformance_case(self):
        """Real tiers, or NoHardware. HETERO_P2P_DMABUF_FIXTURE=`module:callable` receives this transport and returns
        `(src_tier, dst_tier, layout, DmabufBinding)` with live memory it keeps alive."""
        if not os.path.exists(self.device):
            raise NoHardware(f'{self.name}: {self.device} not present')
        if not self.library or not os.path.exists(self.library):
            raise NoHardware(f'{self.name}: native library not found; set HETERO_P2P_DMABUF_LIB')
        try:
            resolve('cuda', 'rngd')
        except PathUnavailable as e:
            raise NoHardware(f'{self.name}: {e}') from e
        fixture = os.environ.get('HETERO_P2P_DMABUF_FIXTURE')
        if not fixture:
            raise NoHardware(f'{self.name}: no live GPU/RNGD memory fixture; set HETERO_P2P_DMABUF_FIXTURE')
        module, _, attr = fixture.partition(':')
        src, dst, layout, binding = getattr(importlib.import_module(module), attr)(self)
        self.bind('conf-1', binding)
        return src, dst, layout
