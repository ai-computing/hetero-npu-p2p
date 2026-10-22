"""GPU <-> RNGD transport: RNGD PDMA reads/writes CUDA memory exported as DMA-BUF (#6, #7).

The RNGD is always the DMA master (the GPU-initiated path is refused, see `paths`). CUDA DMA-BUF export needs the
NVIDIA open kernel module; with the proprietary module `cuda_region()` raises ProprietaryModuleError, and nothing
falls back to host staging.

The RNGD driver attaches and maps the *whole* DMA-BUF on every command, so a command costs time in proportion to the
exported size, not to the bytes it moves (about 7 ms for a 1 GiB export, #6). A CudaRegion therefore exports its
memory once, as slices (16 MiB by default, at most 256 MiB), and keeps the DMA-BUF FDs for reuse. A transfer no larger than one slice
uses an export of exactly its own range instead, also kept (bounded cache), so a small, repeated transfer (one KV
slot) maps only its own bytes.

    region = cuda_region(tensor)                            # exported once, cached per tensor
    backend = RngdDmabufBackend('build/librngd_dmabuf.so', '/dev/rngd/npu0ch0', queue_depth=4)
    backend.write(region, npu_address)                      # GPU -> RNGD DRAM, returns after every CQE
    backend.read(npu_address, region)                       # RNGD DRAM -> GPU
"""
from collections import OrderedDict
from dataclasses import dataclass
import ctypes, hashlib, json, os, threading, time
from pathlib import Path

from .paths import PathUnavailable, require_nvidia_open_module

MiB = 1024 ** 2
SLICE = 16 * MiB                 # slice exported for transfers larger than one slice (measured best, docs/gpu-rngd.md)
MAX_SLICE = 256 * MiB
SPAN = 2 * MiB                   # descriptor size
PAGE = 4096
DESCRIPTORS_PER_COMMAND = 64
QUEUE_DEPTH = 4
EXACT_CACHE = 256                # exact-range exports kept per region
DRAM_LIMIT = 48 * 1024 ** 3 - 512 * MiB   # RNGD DRAM usable by PDMA (driver: NPU_DRAM_SIZE - NPU_FW_SIZE)
_ATTR_DMA_BUF = 124              # CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
_RANGE_DMA_BUF_FD = 1            # CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
NATIVE_ABI = 1


def _cuda():
    cu = ctypes.CDLL('libcuda.so.1')
    cu.cuInit(0)
    return cu


def dmabuf_supported(device=0):
    """True when this GPU can export its memory as DMA-BUF (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED)."""
    try:
        cu = _cuda()
    except OSError:
        return False
    dev, v = ctypes.c_int(), ctypes.c_int()
    if cu.cuDeviceGet(ctypes.byref(dev), device) or cu.cuDeviceGetAttribute(ctypes.byref(v), _ATTR_DMA_BUF, dev):
        return False
    return bool(v.value)


def cuda_export(ptr, size, device=0):
    """DMA-BUF FD for CUDA memory [ptr, ptr+size) (cuMemGetHandleForAddressRange)."""
    h = ctypes.c_int(-1)
    rc = _cuda().cuMemGetHandleForAddressRange(ctypes.byref(h), ctypes.c_uint64(ptr), ctypes.c_size_t(size),
                                               _RANGE_DMA_BUF_FD, ctypes.c_ulonglong(0))
    if rc:
        raise PathUnavailable(f'CUDA DMA-BUF export of {size} bytes failed (CUresult {rc}); GPU <-> RNGD needs the '
                              'NVIDIA open kernel module (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 1)')
    return h.value


class CudaRegion:
    """CUDA memory [ptr, ptr+size) registered for RNGD DMA (a MemoryRegion of kind 'cuda').

    `owner` (e.g. the torch tensor) keeps the memory alive while the region exists; ptr and size must be 4 KiB
    aligned. Exports are made on first use and kept until close(): slices of `slice_bytes` for large transfers, and
    up to `exact_cache` exports of exactly the requested range for transfers no larger than one slice.
    `sync()` makes prior GPU writes visible before the RNGD reads the memory (GPU -> RNGD).
    `exporter(ptr, size, device) -> fd` is the CUDA export by default.
    """
    kind = 'cuda'

    def __init__(self, ptr, size, owner, device=0, slice_bytes=SLICE, exact_cache=EXACT_CACHE, sync=None,
                 exporter=None):
        if type(ptr) is not int or type(size) is not int or ptr % PAGE or size % PAGE or size <= 0 or owner is None:
            raise ValueError('live owner and 4 KiB aligned CUDA range required')
        if type(slice_bytes) is not int or slice_bytes % PAGE or not 0 < slice_bytes <= MAX_SLICE:
            raise ValueError('slice size must be 4 KiB aligned and at most 256 MiB')
        if type(exact_cache) is not int or exact_cache < 0:
            raise ValueError('exact_cache must be a non-negative integer')
        self.ptr, self.size, self.owner, self.device = ptr, size, owner, device
        self.slice_bytes, self.exact_cache = slice_bytes, exact_cache
        self._sync, self._exporter = sync, exporter or cuda_export
        self._slices, self._exact, self._pinned = {}, OrderedDict(), set()
        self.exports_made = 0                 # total export calls; stays flat while transfers reuse the cache
        self.lock = threading.RLock()         # held by a backend from planning until the fence
        self.closed = False

    def _export(self, offset, length):
        fd = self._exporter(self.ptr + offset, length, self.device)
        self.exports_made += 1
        return fd

    def slice_fd(self, index):
        """DMA-BUF FD of slice `index` (exported on first use)."""
        with self.lock:
            self._check_open()
            if index not in self._slices:
                start = index * self.slice_bytes
                if not 0 <= start < self.size:
                    raise ValueError('slice outside the region')
                self._slices[index] = self._export(start, min(self.slice_bytes, self.size - start))
            return self._slices[index]

    def range_fd(self, offset, length):
        """DMA-BUF FD exporting exactly [offset, offset+length) (cached, least recently used evicted)."""
        with self.lock:
            self._check_open()
            key = (offset, length)
            self._pinned.add(key)             # until release_pins(): a plan may still name this FD
            if key in self._exact:
                self._exact.move_to_end(key)
                return self._exact[key]
            fd = self._export(offset, length)
            self._exact[key] = fd
            for old in [k for k in self._exact if k not in self._pinned][:max(0, len(self._exact) - self.exact_cache)]:
                os.close(self._exact.pop(old))
            return fd

    def release_pins(self):
        """Allow eviction of the exact-range exports named by finished plans."""
        with self.lock:
            self._pinned.clear()

    def locate(self, offset, length):
        """[(fd, offset in that DMA-BUF, bytes)] covering region bytes [offset, offset+length).

        Hold `lock` from planning until the transfer is fenced, then call release_pins(): exact-range FDs named by
        a plan are not evicted (closed, and their numbers reused) before that.
        """
        if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0 or \
                offset + length > self.size or offset % PAGE or length % PAGE:
            raise ValueError('4 KiB aligned range inside the region required')
        with self.lock:
            if length <= self.slice_bytes and self.exact_cache:
                return [(self.range_fd(offset, length), 0, length)]
            out = []
            while length:
                i, o = divmod(offset, self.slice_bytes)
                n = min(length, self.slice_bytes - o)
                out.append((self.slice_fd(i), o, n))
                offset, length = offset + n, length - n
            return out

    @property
    def exports(self):
        """Live DMA-BUF exports held by this region."""
        return len(self._slices) + len(self._exact)

    def sync(self):
        if self._sync:
            self._sync()

    def _check_open(self):
        if self.closed:
            raise ValueError('region is closed')

    def close(self):
        """Close every export. The owner (and its memory) stays with the caller."""
        with self.lock:
            for fd in list(self._slices.values()) + list(self._exact.values()):
                os.close(fd)
            self._slices.clear()
            self._exact.clear()
            self.closed = True


_REGIONS = {}
_REGIONS_LOCK = threading.Lock()


def _tensor_range(tensor):
    if not getattr(tensor, 'is_cuda', False) or not tensor.is_contiguous():
        raise ValueError('contiguous CUDA tensor required')
    return tensor.device.index or 0, tensor.data_ptr(), tensor.numel() * tensor.element_size()


def cuda_region(tensor, slice_bytes=SLICE, exact_cache=EXACT_CACHE):
    """The CudaRegion of a contiguous CUDA tensor, created and exported once and then reused.

    The tensor's storage must start and end on 4 KiB boundaries. The region holds the tensor, so its memory cannot
    be freed and reused while registered; `release_region(tensor)` closes the exports and drops it.
    Raises ProprietaryModuleError with the proprietary NVIDIA module and PathUnavailable without DMA-BUF support.
    """
    device, ptr, size = _tensor_range(tensor)
    key = (device, ptr, size)
    with _REGIONS_LOCK:
        region = _REGIONS.get(key)
        if region is not None and not region.closed:
            if region.slice_bytes != slice_bytes or region.exact_cache != exact_cache:
                raise ValueError('tensor already registered with other slicing; release_region() it first')
            return region
        require_nvidia_open_module()
        if not dmabuf_supported(device):
            raise PathUnavailable(f'GPU {device} reports CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0; GPU <-> RNGD '
                                  'needs the NVIDIA open kernel module')
        import torch                     # only reached for real CUDA tensors

        def sync(device=tensor.device):
            torch.cuda.synchronize(device)
        region = CudaRegion(ptr, size, tensor, device, slice_bytes, exact_cache, sync=sync)
        _REGIONS[key] = region
        return region


def release_region(tensor):
    """Close the exports of a tensor registered with cuda_region() and forget it."""
    key = _tensor_range(tensor)
    with _REGIONS_LOCK:
        region = _REGIONS.pop(key, None)
    if region is not None:
        region.close()


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
    descriptors_per_command: int = DESCRIPTORS_PER_COMMAND

    def validate(self):
        if self.direction not in ('read', 'write') or self.owner is None:
            raise ValueError('direction read|write and a live owner required')
        b = self.descriptors_per_command
        if type(b) is not int or not 1 <= b <= 256 or not self.spans or len(self.spans) > 65536:
            raise ValueError('invalid command grouping')
        written = {}                  # destination -> [(start, end)]: RNGD DRAM for writes, the DMA-BUF for reads
        for s in self.spans:
            if any(type(x) is not int for x in (s.fd, s.npu_offset, s.buffer_offset, s.length)):
                raise ValueError('span values must be integers')
            if not 0 < s.length <= SPAN or any(x < 0 or x % PAGE for x in (s.npu_offset, s.buffer_offset, s.length)):
                raise ValueError('4 KiB aligned spans of at most 2 MiB required')
            if s.npu_offset > DRAM_LIMIT - s.length:
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

    def groups(self):
        """[(fd, first span, span count)]: a new command where the FD changes or descriptors_per_command is reached."""
        out = []
        for i, s in enumerate(self.spans):
            if not out or s.fd != out[-1][0] or out[-1][2] == self.descriptors_per_command:
                out.append([s.fd, i, 0])
            out[-1][2] += 1
        return [tuple(g) for g in out]

    @property
    def commands(self):
        return len(self.groups())

    @property
    def bytes(self):
        return sum(s.length for s in self.spans)


@dataclass(frozen=True)
class DmabufCopy:
    """One contiguous copy between RNGD DRAM [npu_offset, +length) and region [region_offset, +length)."""
    npu_offset: int
    region: object
    region_offset: int
    length: int


def _spans(region, region_offset, npu_offset, length):
    spans = []
    for fd, off, n in region.locate(region_offset, length):
        for k in range(0, n, SPAN):
            m = min(SPAN, n - k)
            spans.append(DmabufSpan(fd, npu_offset, off + k, m))
            npu_offset += m
    return spans


def plan_copies(direction, copies, owner=None, descriptors_per_command=DESCRIPTORS_PER_COMMAND):
    """One DmabufPlan for several DmabufCopy items, in 2 MiB spans over each region's exports.

    Hold every region's `lock` from planning until the transfer is fenced (RngdDmabufBackend does).
    """
    spans = []
    for c in copies:
        if type(c.npu_offset) is not int or c.npu_offset < 0:
            raise ValueError('non-negative RNGD DRAM offset required')
        spans += _spans(c.region, c.region_offset, c.npu_offset, c.length)
    return DmabufPlan(direction, tuple(spans), (owner, tuple(c.region for c in copies)), descriptors_per_command)


def plan_copy(direction, region, region_offset, npu_offset, length, owner=None,
              descriptors_per_command=DESCRIPTORS_PER_COMMAND):
    """DmabufPlan moving `length` bytes between region[region_offset:] and RNGD DRAM [npu_offset:]."""
    return plan_copies(direction, (DmabufCopy(npu_offset, region, region_offset, length),), owner,
                       descriptors_per_command)


class _Native:
    """ctypes binding of build/librngd_dmabuf.so (native/rngd_dmabuf.c)."""

    def __init__(self, path):
        lib = ctypes.CDLL(str(path))
        lib.rd_abi.restype = ctypes.c_int
        lib.rd_open.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        lib.rd_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32),
                               ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                               ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                               ctypes.POINTER(ctypes.c_uint64)]
        lib.rd_close.argtypes = [ctypes.c_void_p]
        lib.rd_close.restype = None
        if lib.rd_abi() != NATIVE_ABI:
            raise RuntimeError('native library ABI mismatch')
        self.lib = lib

    def open(self, device, depth):
        ctx = ctypes.c_void_p()
        rc = self.lib.rd_open(device.encode(), depth, ctypes.byref(ctx))
        return rc, ctx

    def run(self, ctx, write, fds, counts, npu, off, length):
        n, d = len(fds), len(npu)
        arr = lambda t, v: (t * len(v))(*v)
        done = ctypes.c_uint64(0)
        rc = self.lib.rd_run(ctx, int(write), n, arr(ctypes.c_int32, fds), arr(ctypes.c_uint32, counts), d,
                             arr(ctypes.c_uint64, npu), arr(ctypes.c_uint64, off), arr(ctypes.c_uint64, length),
                             ctypes.byref(done))
        return rc, done.value

    def close(self, ctx):
        self.lib.rd_close(ctx)


class RngdDmabufBackend:
    """Runs DmabufPlans on one RNGD with an in-process native library (one device FD and io_uring, reused).

    `queue_depth` commands are in flight at once (the driver spreads them over its PDMA channels);
    `descriptors_per_command` 2 MiB descriptors make one command. `transfer`, `read` and `write` return a receipt
    only after every command's CQE was reaped (completion fence) and raise on any failure; a failed transfer leaves
    nothing in flight. `native` replaces the library (tests).
    """

    def __init__(self, library, device='/dev/rngd/npu0ch0', queue_depth=QUEUE_DEPTH,
                 descriptors_per_command=DESCRIPTORS_PER_COMMAND, native=None):
        if type(queue_depth) is not int or not 1 <= queue_depth <= 64:
            raise ValueError('queue depth 1..64 required')
        if type(descriptors_per_command) is not int or not 1 <= descriptors_per_command <= 256:
            raise ValueError('descriptors_per_command 1..256 required')
        if native is None:
            library = Path(library).resolve()
            manifest = json.loads(library.with_suffix('.json').read_text())
            if hashlib.sha256(library.read_bytes()).hexdigest() != manifest['sha256']:
                raise RuntimeError('native library does not match build manifest')
            if manifest.get('rngd_driver') != '2026.4.0':
                raise RuntimeError('unsupported native library build ABI')
            native = _Native(library)
        self.device, self.queue_depth, self.descriptors_per_command = device, queue_depth, descriptors_per_command
        self._native, self._lock = native, threading.Lock()
        rc, self._ctx = native.open(device, queue_depth)
        if rc:
            self._ctx = None
            if rc == -95:          # EOPNOTSUPP: no NPU_PDMA_CAPS_DMABUF
                raise PathUnavailable(f'{device}: RNGD driver without DMA-BUF PDMA (NPU_PDMA_CAPS_DMABUF)')
            raise OSError(-rc, f'cannot open RNGD PDMA {device}: {os.strerror(-rc)}')

    def transfer(self, plan: DmabufPlan):
        plan.validate()
        groups = plan.groups()
        with self._lock:
            if self._ctx is None:
                raise RuntimeError('backend is closed')
            started = time.perf_counter()
            rc, completed = self._native.run(self._ctx, plan.direction == 'write', [g[0] for g in groups],
                                             [g[2] for g in groups], [s.npu_offset for s in plan.spans],
                                             [s.buffer_offset for s in plan.spans], [s.length for s in plan.spans])
            seconds = time.perf_counter() - started
        receipt = dict(direction=plan.direction, cqe_result=rc, completed=completed, requested=len(groups),
                       bytes_requested=plan.bytes, seconds=seconds, queue_depth=self.queue_depth,
                       descriptors_per_command=plan.descriptors_per_command)
        if rc:
            raise RuntimeError(f'DMA-BUF PDMA failed ({os.strerror(-rc) if rc < 0 else rc}): {receipt}')
        if completed != len(groups):
            raise RuntimeError(f'invalid completion receipt: {receipt}')
        return receipt

    def copy(self, direction, copies, owner=None):
        """Plan and run DmabufCopy items, holding their regions' locks until the fence."""
        regions = sorted({id(c.region): c.region for c in copies}.values(), key=id)
        for r in regions:
            r.lock.acquire()
        try:
            if direction == 'write':
                for r in regions:
                    r.sync()      # GPU writes must be complete before the RNGD reads the memory
            plan = plan_copies(direction, copies, owner, self.descriptors_per_command)
            receipt = self.transfer(plan)
            receipt['exports'] = sum(r.exports for r in regions)
            return receipt
        finally:
            for r in reversed(regions):
                r.release_pins()
                r.lock.release()

    @staticmethod
    def _region(region):
        return region if hasattr(region, 'locate') else cuda_region(region)

    def read(self, npu_offset, region, region_offset=0, length=None, owner=None):
        """RNGD DRAM [npu_offset, +length) -> region[region_offset:]; region may be a CUDA tensor."""
        region = self._region(region)
        length = region.size - region_offset if length is None else length
        return self.copy('read', (DmabufCopy(npu_offset, region, region_offset, length),), owner)

    def write(self, region, npu_offset, region_offset=0, length=None, owner=None):
        """region[region_offset:] -> RNGD DRAM [npu_offset, +length); region may be a CUDA tensor."""
        region = self._region(region)
        length = region.size - region_offset if length is None else length
        return self.copy('write', (DmabufCopy(npu_offset, region, region_offset, length),), owner)

    def close(self):
        with self._lock:
            if self._ctx is not None:
                self._native.close(self._ctx)
                self._ctx = None
