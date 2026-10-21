"""GPU <-> RNGD transport: RNGD PDMA reads/writes CUDA memory exported as DMA-BUF (#7).

Measured in #6 (RNGD+A100 host, NVIDIA open kernel module): RNGD -> GPU 17-19 GB/s, GPU -> RNGD 10-13 GB/s, byte-exact.
The RNGD driver attaches and maps the whole DMA-BUF on every command, so the cost per command grows with the exported
size: a CudaRegion exports its memory as slices of at most 256 MiB, once, and plans address those slices. The GPU
cannot initiate this copy (cudaHostRegisterIoMemory on the RNGD BAR is refused), so the RNGD is always the DMA master.
CUDA DMA-BUF export needs the NVIDIA open kernel module; with the proprietary module it fails here, never silently
through the host.
"""
from dataclasses import dataclass
import ctypes, hashlib, json, os, signal, subprocess, tempfile, threading
from pathlib import Path

SLICE = 16 * 1024 ** 2           # DMA-BUF export per slice (16 MiB: best for 2 MiB-1 GiB, see docs/gpu-rngd.md)
SPAN = 2 * 1024 ** 2             # descriptor size
PAGE = 4096
_ATTR_DMA_BUF = 124              # CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
_RANGE_DMA_BUF_FD = 1            # CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD


def _cuda():
    return ctypes.CDLL('libcuda.so.1')


def dmabuf_supported(device=0):
    """True when this GPU can export its memory as DMA-BUF (NVIDIA open kernel module)."""
    cu, dev, v = _cuda(), ctypes.c_int(), ctypes.c_int()
    if cu.cuInit(0) or cu.cuDeviceGet(ctypes.byref(dev), device) or \
            cu.cuDeviceGetAttribute(ctypes.byref(v), _ATTR_DMA_BUF, dev):
        return False
    return bool(v.value)


class CudaRegion:
    """CUDA memory [ptr, ptr+size) registered for RNGD DMA: exported lazily as DMA-BUF slices (16 MiB by default) and kept
    until close(). `owner` (e.g. the torch tensor) keeps the memory alive; ptr and size must be 4 KiB aligned."""

    def __init__(self, ptr, size, owner, device=0, slice_bytes=SLICE):
        if type(ptr) is not int or type(size) is not int or ptr % PAGE or size % PAGE or size <= 0 or owner is None:
            raise ValueError('live owner and 4 KiB aligned CUDA range required')
        if slice_bytes % PAGE or not 0 < slice_bytes <= 256 * 1024 ** 2:
            raise ValueError('slice size must be 4 KiB aligned and <= 256 MiB')
        self.ptr, self.size, self.owner, self.device, self.slice_bytes = ptr, size, owner, device, slice_bytes
        self._fds = {}
        self._lock = threading.Lock()

    @classmethod
    def from_tensor(cls, tensor, **kw):
        """a contiguous CUDA tensor whose storage starts and ends on 4 KiB boundaries"""
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError('contiguous CUDA tensor required')
        return cls(tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor, device=tensor.device.index or 0,
                   **kw)

    def fd(self, index):
        """DMA-BUF FD of slice `index` (exported on first use)"""
        with self._lock:
            if index not in self._fds:
                start = index * self.slice_bytes
                if not 0 <= start < self.size:
                    raise ValueError('slice outside the region')
                n, h = min(self.slice_bytes, self.size - start), ctypes.c_int(-1)
                rc = _cuda().cuMemGetHandleForAddressRange(ctypes.byref(h), ctypes.c_uint64(self.ptr + start),
                                                           ctypes.c_size_t(n), _RANGE_DMA_BUF_FD, ctypes.c_ulonglong(0))
                if rc:
                    raise RuntimeError(f'CUDA DMA-BUF export failed (CUresult {rc}); needs the NVIDIA open kernel '
                                       'module (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 1)')
                self._fds[index] = h.value
            return self._fds[index]

    def locate(self, offset, length):
        """[(slice index, offset in slice, bytes)] covering region bytes [offset, offset+length)"""
        if offset < 0 or length <= 0 or offset + length > self.size:
            raise ValueError('range outside the region')
        out = []
        while length:
            i, o = divmod(offset, self.slice_bytes)
            n = min(length, self.slice_bytes - o)
            out.append((i, o, n))
            offset, length = offset + n, length - n
        return out

    def close(self):
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


@dataclass(frozen=True)
class DmabufSpan:
    fd: int
    npu_offset: int        # RNGD DRAM offset (as SourceAllocation.address)
    buffer_offset: int     # offset inside the DMA-BUF
    length: int


@dataclass(frozen=True)
class DmabufPlan:
    """direction 'read': RNGD DRAM -> DMA-BUF (e.g. NPU KV -> GPU); 'write': DMA-BUF -> RNGD DRAM."""
    direction: str
    spans: tuple
    owner: object                       # keeps the RNGD allocation and the region alive
    descriptors_per_command: int = 64

    def validate(self):
        if self.direction not in ('read', 'write') or self.owner is None:
            raise ValueError('direction read|write and a live owner required')
        b = self.descriptors_per_command
        if type(b) is not int or not 1 <= b <= 64 or not self.spans or len(self.spans) > 8192:
            raise ValueError('invalid command grouping')
        limit = 48 * 1024 ** 3 - 512 * 1024 ** 2
        written = {}                  # destination -> [(start, end)]: RNGD DRAM for writes, the DMA-BUF for reads
        for s in self.spans:
            if any(type(x) is not int for x in (s.fd, s.npu_offset, s.buffer_offset, s.length)):
                raise ValueError('span values must be integers')
            if not 0 < s.length <= SPAN or any(x < 0 or x % PAGE for x in (s.npu_offset, s.buffer_offset, s.length)):
                raise ValueError('4 KiB aligned spans of at most 2 MiB required')
            if s.npu_offset > limit - s.length:
                raise ValueError('span outside the validated RNGD DRAM region')
            st = os.fstat(s.fd)
            if s.buffer_offset + s.length > st.st_size:
                raise ValueError('span outside the exported DMA-BUF')
            key, lo = ('rngd', s.npu_offset) if self.direction == 'write' else ((st.st_dev, st.st_ino), s.buffer_offset)
            written.setdefault(key, []).append((lo, lo + s.length))
        for ranges in written.values():            # FD aliases of one DMA-BUF share (dev, ino)
            ranges.sort()
            if any(b < a_end for (_, a_end), (b, _) in zip(ranges, ranges[1:])):
                raise ValueError('overlapping destination ranges')
        return self

    @property
    def commands(self):
        """as the helper groups them: a new command where the FD changes or descriptors_per_command is reached"""
        n, run = 0, 0
        for i, s in enumerate(self.spans):
            if not i or s.fd != self.spans[i - 1].fd or run == self.descriptors_per_command:
                n, run = n + 1, 0
            run += 1
        return n

    @property
    def bytes(self):
        return sum(s.length for s in self.spans)


def plan_copy(direction, region, region_offset, npu_offset, length, owner=None, descriptors_per_command=64):
    """DmabufPlan moving `length` bytes between region[region_offset:] and RNGD DRAM [npu_offset:], in 2 MiB spans
    over the region's slices."""
    spans = []
    for index, off, n in region.locate(region_offset, length):
        fd = region.fd(index)
        for k in range(0, n, SPAN):
            m = min(SPAN, n - k)
            spans.append(DmabufSpan(fd, npu_offset, off + k, m))
            npu_offset += m
    return DmabufPlan(direction, tuple(spans), (owner, region), descriptors_per_command)


class RngdDmabufBackend:
    """Runs DmabufPlans through the checked native helper (build/rngd-dmabuf, native/build.py)."""

    def __init__(self, helper, device='/dev/rngd/npu0ch0', queue_depth=2):
        self.helper = Path(helper).resolve()
        manifest = json.loads(self.helper.with_suffix('.json').read_text())
        if hashlib.sha256(self.helper.read_bytes()).hexdigest() != manifest['sha256']:
            raise RuntimeError('native helper does not match build manifest')
        if manifest.get('rngd_driver') != '2026.4.0':
            raise RuntimeError('unsupported helper build ABI')
        if not Path(device).exists():
            raise RuntimeError(f'{device} not present')
        self.device, self.queue_depth = device, queue_depth

    def transfer(self, plan: DmabufPlan, reps=1):
        plan.validate()
        with tempfile.TemporaryDirectory(prefix='hetero-p2p-dmabuf-') as directory:
            path = Path(directory) / 'plan.txt'
            path.write_text(''.join(f'{s.fd} {s.npu_offset} {s.buffer_offset} {s.length}\n' for s in plan.spans))
            fds = tuple(sorted({s.fd for s in plan.spans}))
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            try:
                proc = subprocess.run([str(self.helper), self.device, plan.direction, str(path),
                                       str(plan.descriptors_per_command), str(self.queue_depth), str(reps)],
                                      pass_fds=fds, capture_output=True, text=True)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        if proc.returncode:
            raise RuntimeError(f'DMA-BUF PDMA failed ({proc.returncode}): {proc.stdout} {proc.stderr}')
        receipt = json.loads(proc.stdout)
        if (receipt.get('cqe_result') != 0 or receipt.get('completed') != plan.commands * reps
                or receipt.get('bytes_requested') != plan.bytes):
            raise RuntimeError(f'invalid completion receipt: {receipt}')
        return receipt
