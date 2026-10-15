from dataclasses import dataclass
import os
import threading
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class SourceAllocation:
    """Live raw RNGD DRAM offset, not a host virtual address. Owner pins lifetime."""
    address: int
    size: int
    owner: Any
    format: str
    device: str = '/dev/rngd/npu1ch0'


@dataclass(frozen=True)
class DestinationAllocation:
    """Borrowed live DMA-BUF FD; owner must remain alive through decode."""
    fd: int
    owner: Any
    format: str


@dataclass(frozen=True)
class CopySpan:
    destination: int
    source_offset: int
    destination_offset: int
    length: int


@dataclass(frozen=True)
class TransferPlan:
    source: SourceAllocation
    destinations: tuple[DestinationAllocation, ...]
    spans: tuple[CopySpan, ...]
    descriptors_per_command: int = 8

    def validate(self):
        s = self.source
        ints = [s.address, s.size, self.descriptors_per_command]
        if any(type(x) is not int for x in ints):
            raise ValueError('addresses, sizes and grouping must be integers')
        limit = 48*1024**3 - 512*1024**2
        if s.owner is None or not s.format or s.address <= 0 or s.address % 64:
            raise ValueError('live owner, format and 64-byte-aligned source required')
        if not 0 < s.size <= 64*1024**2 or s.size % 4096 or s.address > limit-s.size:
            raise ValueError('source exceeds validated RNGD DRAM region')
        batch = self.descriptors_per_command
        if not 1 <= batch <= 8 or not 0 < len(self.spans) <= 512 or len(self.spans) % batch:
            raise ValueError('invalid command grouping')
        stats = []
        for d in self.destinations:
            if type(d.fd) is not int or d.fd < 0 or d.owner is None or d.format != s.format:
                raise ValueError('live destination owner and identical physical format required')
            stats.append(os.fstat(d.fd))
        ranges = {}
        for i, span in enumerate(self.spans):
            if any(type(x) is not int for x in (span.destination, span.source_offset, span.destination_offset, span.length)):
                raise ValueError('span values must be integers')
            if not 0 <= span.destination < len(stats):
                raise ValueError('unknown destination')
            st = stats[span.destination]
            if not 0 < span.length <= 2*1024**2 or any(x < 0 or x % 4096 for x in (span.source_offset, span.destination_offset, span.length)):
                raise ValueError('4 KiB aligned positive span required')
            if span.source_offset+span.length > s.size or span.destination_offset+span.length > st.st_size:
                raise ValueError('span outside source or exported destination')
            key = (st.st_dev, st.st_ino)
            lo, hi = span.destination_offset, span.destination_offset+span.length
            if any(lo < end and start < hi for start, end in ranges.get(key, [])):
                raise ValueError('overlapping destination, including duplicate FD aliases')
            ranges.setdefault(key, []).append((lo, hi))
            first = self.spans[i-i % batch]
            if self.destinations[first.destination].fd != self.destinations[span.destination].fd:
                raise ValueError('one DMA-BUF FD per command required')
        return self

    @property
    def commands(self):
        return len(self.spans)//self.descriptors_per_command

    @property
    def bytes(self):
        return sum(x.length for x in self.spans)


class Backend(Protocol):
    def transfer(self, plan: TransferPlan) -> dict: ...


class CopySession:
    """One destination cache lease. Failure poisons it; no premature reuse."""
    def __init__(self, backend: Backend):
        self.backend = backend
        self.state = 'idle'
        self.request_id = None
        self._plan = None
        self._lock = threading.RLock()

    def admit(self, request_id: str, plan: TransferPlan, notify: Callable[[], None]):
        with self._lock:
            if self.state != 'idle' or not request_id:
                raise RuntimeError('destination cache is not idle or request ID is empty')
            plan.validate()
            self.request_id, self._plan, self.state = request_id, plan, 'admitting'
            try:
                receipt = self.backend.transfer(plan)
                if (receipt.get('cqe_result') != 0 or receipt.get('completed') != plan.commands
                        or receipt.get('requested') != plan.commands or receipt.get('bytes_requested') != plan.bytes):
                    raise RuntimeError('incomplete or inconsistent DMA completion receipt')
                notify()  # Mark device cache updated only after every CQE completed.
                self.state = 'ready'
                return receipt
            except BaseException:
                self.state = 'poisoned'  # Retain owners; never recycle on an uncertain transfer.
                raise

    def start_decode(self, request_id: str):
        with self._lock:
            if self.state != 'ready' or request_id != self.request_id:
                raise RuntimeError('decode requires matching completed admission')
            self.state = 'decoding'

    def finish_decode(self, request_id: str, passed: bool):
        with self._lock:
            if self.state != 'decoding' or request_id != self.request_id:
                raise RuntimeError('stale or duplicate decode completion')
            if passed is not True:
                self.state = 'poisoned'
                raise RuntimeError('decode failed; cache cannot be reused')
            self.state, self.request_id, self._plan = 'idle', None, None

    def discard_ready(self, request_id: str):
        """TTL/cancel only after fenced admission and before decode starts."""
        with self._lock:
            if self.state != 'ready' or request_id != self.request_id:
                raise RuntimeError('cannot discard active or uncertain work')
            self.state, self.request_id, self._plan = 'idle', None, None
