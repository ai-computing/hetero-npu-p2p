"""CPU stand-ins for GPU <-> RNGD tests: memfd "exports" that alias one shared GPU memory, and a fake native library
that really copies bytes between them and a fake RNGD DRAM. No device is touched."""
import os
from hetero_p2p.gpu import CudaRegion

MiB = 1024 ** 2


class FakeGpu:
    """GPU memory as a bytearray. exporter() returns a memfd sized like the export (so fstat checks work) and
    remembers which GPU bytes it stands for, so every export aliases the same memory as a real DMA-BUF would."""

    def __init__(self, size, base=1 << 30):
        self.mem, self.base, self.fds = bytearray(size), base, {}

    def exporter(self, ptr, size, device):
        fd = os.memfd_create('fake-dmabuf')
        os.ftruncate(fd, size)
        self.fds[fd] = ptr - self.base
        return fd

    def region(self, **kw):
        return CudaRegion(self.base, len(self.mem), owner=self, exporter=self.exporter, **kw)


class FakeNative:
    def __init__(self, gpu, dram=96 * MiB, open_rc=0):
        self.gpu, self.dram, self.open_rc = gpu, bytearray(dram), open_rc
        self.runs, self.fail_after, self.closed, self.depth = [], None, False, None

    def open(self, device, depth):
        self.depth = depth
        return self.open_rc, object()

    def run(self, ctx, write, fds, counts, npu, off, length):
        self.runs.append(dict(write=write, fds=list(fds), counts=list(counts), npu=list(npu), off=list(off),
                              len=list(length)))
        i = done = 0
        for fd, n in zip(fds, counts):
            if self.fail_after is not None and done == self.fail_after:
                return -5, done                     # -EIO on this command; earlier ones completed
            os.fstat(fd)                            # a closed FD would fail like the driver's dma_buf_get
            for j in range(i, i + n):
                g = self.gpu.fds[fd] + off[j]
                if write:
                    self.dram[npu[j]:npu[j] + length[j]] = self.gpu.mem[g:g + length[j]]
                else:
                    self.gpu.mem[g:g + length[j]] = self.dram[npu[j]:npu[j] + length[j]]
            i += n
            done += 1
        return 0, done

    def close(self, ctx):
        self.closed = True
