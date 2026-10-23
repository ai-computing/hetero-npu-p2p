"""GPU <-> RNGD KV hand-off for Optimus Prime PD (`opt_prime.accel.pd.contracts.Transport`, structural; no import).

    transport = DmabufPDTransport(RngdDmabufBackend(lib, '/dev/rngd/npu0ch0'), copy_builder,
                                  source='cuda:0', destination='rngd:/dev/rngd/npu0ch0', spec=spec)
    PDRuntime(gpu_prefill, rngd_decode, transport)

`copy_builder(source_lease, destination_lease)` returns the DmabufCopy items that move the KV bytes: RNGD DRAM
offsets on the RNGD side, CudaRegion ranges (`cuda_region(tensor)`) on the GPU side. Both sides must already hold
the negotiated physical representation (`KVSpec.representation` / `layout`): nothing is converted here.
`transfer` returns only after every DMA completion (fence) and after `notify(destination)`; any failure raises and
nothing falls back to a host copy. Endpoints are 'cuda:<index>' and 'rngd:<device path or index>'.
"""
import threading

from .gpu import DmaCompletionUncertain, DmabufCopy
from .paths import resolve

DIRECTIONS = {('cuda', 'rngd'): 'write', ('rngd', 'cuda'): 'read'}


def endpoint_kind(endpoint):
    kind = endpoint.split(':', 1)[0] if isinstance(endpoint, str) else ''
    if kind not in ('cuda', 'rngd'):
        raise ValueError(f'endpoint {endpoint!r} is neither cuda:<n> nor rngd:<device>')
    return kind


class DmabufPDTransport:
    name = 'p2p-dmabuf'

    def __init__(self, backend, copy_builder, notify=None, *, source, destination, spec):
        pair = (endpoint_kind(source), endpoint_kind(destination))
        if pair not in DIRECTIONS:
            raise ValueError(f'GPU <-> RNGD only, not {source} -> {destination}')
        resolve(*pair)                    # ProprietaryModuleError / PathUnavailable: explicit, before any request
        self.backend, self.copy_builder, self.notify = backend, copy_builder, notify
        self.source, self.destination, self.spec = source, destination, spec
        self.direction = DIRECTIONS[pair]
        self._active, self._lock = {}, threading.Lock()

    def supports(self, source, destination, spec):
        return (source, destination, spec) == (self.source, self.destination, self.spec)

    def transfer(self, request_id, source, destination):
        if source.spec != destination.spec or not self.supports(source.endpoint, destination.endpoint, source.spec):
            raise ValueError('unsupported P2P route or physical format')
        with self._lock:
            if request_id in self._active:
                raise RuntimeError(f'request {request_id!r} already transferred and not finished')
            self._active[request_id] = None
        try:
            copies = tuple(self.copy_builder(source, destination))
            if not copies or not all(isinstance(c, DmabufCopy) for c in copies):
                raise ValueError('copy_builder must return DmabufCopy items')
            receipt = self.backend.copy(self.direction, copies, owner=(source.owner, destination.owner))
            if self.notify:
                self.notify(destination)  # the destination becomes visible only after the fence
        except DmaCompletionUncertain:
            with self._lock:
                self._active[request_id] = DmaCompletionUncertain
            raise
        except BaseException:
            with self._lock:
                self._active.pop(request_id, None)
            raise
        receipt = dict(receipt, transport=self.name)
        with self._lock:
            self._active[request_id] = receipt
        return receipt

    def finish(self, request_id):
        with self._lock:
            if request_id not in self._active:
                raise RuntimeError(f'stale or duplicate finish for {request_id!r}')
            if self._active[request_id] is DmaCompletionUncertain:
                raise DmaCompletionUncertain('DMA completion is uncertain; cannot finish request')
            del self._active[request_id]

    def recover(self):
        """An unknown completion keeps the request active until the native backend observes its CQEs."""
        with self._lock:
            if any(value is None for value in self._active.values()):
                return False
            if any(value is DmaCompletionUncertain for value in self._active.values()) and not self.backend.recover():
                return False
            self._active.clear()
        return True
