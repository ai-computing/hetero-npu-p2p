"""GPU <-> RNGD through hetero_p2p.gpu (#7): byte-exact transfers and bandwidth, 2 MiB - 1 GiB, per GPU/RNGD pair.

Bandwidth is from the wall time of the API call (`backend.write` / `backend.read`: planning, the GPU sync before a
write, the native run and the completion fence), median of --reps; native_* is the native run alone (receipt). Checks per size and pair:
  GPU -> RNGD: 64 random 4 KiB pages (and the first and last page) read back by the CPU through the RNGD BAR4 window;
  RNGD -> GPU: the destination is poisoned first, then compared byte for byte on the GPU; the last page is a marker
               the CPU wrote into RNGD DRAM through BAR4, so the bytes cannot come from anywhere but the RNGD.
Experiments (--experiments):
  pairs    every --pairs chip:gpu pair with the default backend (cached region, default slice, --queue-depth)
  exports  pair 1: cached region vs a fresh export per transfer vs one 1 GiB export (stage 1's large-export case)
  depth    pair 1: queue depth sweep;   slices  pair 1: slice size sweep (exact-range exports off)

The RNGD side is raw DRAM at --npu-offset, not an SDK allocation: the chip must be idle (no process holding
/dev/rngd/npuN*). Needs the NVIDIA open kernel module and build/librngd_dmabuf.so (native/build.py).

    python examples/gpu_rngd_dmabuf.py --library build/librngd_dmabuf.so --pairs 0:0,0:1,1:0,1:1 --out result.json
"""
import argparse, fcntl, json, mmap, os, platform, random, statistics, struct, sys, time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--library', required=True)
ap.add_argument('--pairs', default='0:0', help='chip:gpu,... (RNGD index : CUDA index), run one at a time')
ap.add_argument('--sizes', default='2,4,16,64,256,1024', help='MiB')
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--queue-depth', type=int, default=4)
ap.add_argument('--descriptors', type=int, default=64)
ap.add_argument('--npu-offset', type=int, default=4 << 30, help='RNGD DRAM offset used (bytes, 2 MiB aligned)')
ap.add_argument('--experiments', default='pairs,exports,depth,slices')
ap.add_argument('--depths', default='1,2,4,8')
ap.add_argument('--slice-sweep', default='16,64,256', help='MiB')
ap.add_argument('--out')
a = ap.parse_args()

import torch
from hetero_p2p import gpu
from hetero_p2p.gpu import CudaRegion, DmabufPlan, DmabufSpan, RngdDmabufBackend, cuda_region, release_region
from hetero_p2p.paths import nvidia_kernel_module, resolve

MiB = 1 << 20
SIZES = [int(x) * MiB for x in a.sizes.split(',')]
TOP = max(SIZES)
BAR_DRAM = 256 * MiB                       # BAR4 offset of RNGD DRAM offset 0 (NPU_DRAM_BASE on a PF)
GET_AVAILABLE_DRAM = 0xC0104E02            # _IOWR('N', 2, struct npu_bar_info {u64 addr; u64 size})
rng = random.Random(6)


def log(**kw):
    print(json.dumps(kw), flush=True)


def med(xs):
    return statistics.median(xs)


class Pair:
    def __init__(self, chip, index):
        self.chip, self.index = chip, index
        self.dev = torch.device('cuda', index)
        torch.cuda.set_device(self.dev)
        self.src = torch.empty(TOP, dtype=torch.uint8, device=self.dev)
        self.dst = torch.empty(TOP, dtype=torch.uint8, device=self.dev)
        self.backend = RngdDmabufBackend(a.library, f'/dev/rngd/npu{chip}ch0', a.queue_depth, a.descriptors)
        self.bar = os.open(f'/dev/rngd/npu{chip}bar4', os.O_RDWR | os.O_SYNC)
        info = bytearray(16)
        fcntl.ioctl(self.bar, GET_AVAILABLE_DRAM, info, True)
        usable = struct.unpack('<QQ', info)[1]
        if a.npu_offset % (2 * MiB) or a.npu_offset + TOP > usable:
            sys.exit(f'--npu-offset outside the usable RNGD DRAM ({usable} bytes)')
        self.mm = mmap.mmap(self.bar, TOP, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                            offset=BAR_DRAM + a.npu_offset)

    def close(self):
        self.mm.close()
        os.close(self.bar)
        self.backend.close()
        release_region(self.src)
        release_region(self.dst)


NATIVE = []                                # receipt['seconds'] of the last calls: native run only


def timed(fn, reps):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        NATIVE.append(fn()['seconds'])
        out.append(time.perf_counter() - t)
    return out


def verified_round_trip(p, n, write, read, reps):
    """Fill the GPU source, time write and read, and verify both directions. Returns the row."""
    p.src[:n].random_(0, 256)
    torch.cuda.synchronize(p.dev)
    NATIVE.clear()
    tw = timed(lambda: write(n), reps)
    nw = list(NATIVE)
    host = p.src[:n].cpu().numpy().tobytes()
    pages = sorted({0, n - 4096, *(rng.randrange(0, n // 4096) * 4096 for _ in range(64))})
    bar_ok = all(p.mm[o:o + 4096] == host[o:o + 4096] for o in pages)
    marker = os.urandom(4096)
    p.mm[n - 4096:n] = marker
    times = []
    for _ in range(reps):                 # poison before every read, so each read is checked
        p.dst[:n].fill_(0xA5)
        torch.cuda.synchronize(p.dev)
        t = time.perf_counter()
        nr = read(n)['seconds']
        times.append(time.perf_counter() - t)
        NATIVE.append(nr)
    exact = bool(torch.equal(p.dst[:n - 4096], p.src[:n - 4096]))
    mark = p.dst[n - 4096:n].cpu().numpy().tobytes() == marker
    return dict(MiB=n // MiB, gpu_to_rngd_GBps=round(n / med(tw) / 1e9, 2), rngd_to_gpu_GBps=round(n / med(times) / 1e9, 2),
                native_gpu_to_rngd_GBps=round(n / med(nw) / 1e9, 2),
                native_rngd_to_gpu_GBps=round(n / med(NATIVE[len(nw):]) / 1e9, 2),
                gpu_to_rngd_ms=[round(x * 1e3, 3) for x in tw], rngd_to_gpu_ms=[round(x * 1e3, 3) for x in times],
                bar_pages_checked=len(pages), rngd_bar_exact=bar_ok, gpu_exact=exact, marker_from_rngd=mark,
                passed=bar_ok and exact and mark)


def api(p, slice_bytes=gpu.SLICE, exact=gpu.EXACT_CACHE):
    release_region(p.src)
    release_region(p.dst)
    rs, rd = cuda_region(p.src, slice_bytes, exact), cuda_region(p.dst, slice_bytes, exact)
    return (lambda n: p.backend.write(rs, a.npu_offset, length=n),
            lambda n: p.backend.read(a.npu_offset, rd, length=n), rs, rd)


def per_transfer_export(p):
    """A fresh export of exactly the transferred range for every call, closed afterwards (no cache)."""
    def run(direction, tensor, n):
        r = CudaRegion(tensor.data_ptr(), n, tensor, p.index, sync=lambda: torch.cuda.synchronize(p.dev))
        try:
            return p.backend.write(r, a.npu_offset) if direction == 'write' else p.backend.read(a.npu_offset, r)
        finally:
            r.close()
    return lambda n: run('write', p.src, n), lambda n: run('read', p.dst, n)


def whole_export(p):
    """One export of the full 1 GiB buffer reused for every size: each command maps all of it (stage 1, row 2)."""
    fds = {k: gpu.cuda_export(t.data_ptr(), TOP, p.index) for k, t in (('w', p.src), ('r', p.dst))}

    def run(direction, n):
        fd = fds['w' if direction == 'write' else 'r']
        spans = tuple(DmabufSpan(fd, a.npu_offset + o, o, min(gpu.SPAN, n - o)) for o in range(0, n, gpu.SPAN))
        if direction == 'write':
            torch.cuda.synchronize(p.dev)
        return p.backend.transfer(DmabufPlan(direction, spans, p, a.descriptors))
    return (lambda n: run('write', n), lambda n: run('read', n), fds)


def main():
    resolve('cuda', 'rngd')               # explicit error with the proprietary module
    pairs = [tuple(int(v) for v in x.split(':')) for x in a.pairs.split(',')]
    experiments = a.experiments.split(',')
    env = dict(kernel=platform.release(), torch=torch.__version__,
               nvidia_module=nvidia_kernel_module(), nvidia=open('/proc/driver/nvidia/version').readline().split('  ')[1],
               gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               queue_depth=a.queue_depth, descriptors_per_command=a.descriptors, slice_MiB=gpu.SLICE // MiB,
               exact_cache=gpu.EXACT_CACHE, reps=a.reps, npu_offset=a.npu_offset)
    log(env=env)
    result = dict(env=env, pairs=[], exports=[], depth=[], slices=[])
    for k, (chip, index) in enumerate(pairs):
        p = Pair(chip, index)
        try:
            if 'pairs' in experiments:
                write, read, rs, rd = api(p)
                for n in SIZES:
                    row = dict(chip=chip, gpu=index, **verified_round_trip(p, n, write, read, a.reps))
                    row['exports_made'] = rs.exports_made + rd.exports_made
                    result['pairs'].append(row)
                    log(experiment='pairs', **row)
            if k:
                continue                  # the sweeps run on the first pair only
            if 'exports' in experiments:
                write, read, rs, rd = api(p)
                variants = [('cached', write, read), ('export-per-transfer', *per_transfer_export(p))]
                ww, wr, fds = whole_export(p)
                variants.append(('one-1GiB-export', ww, wr))
                try:
                    for name, w, r in variants:
                        for n in SIZES:
                            row = dict(variant=name, **verified_round_trip(p, n, w, r, a.reps))
                            result['exports'].append(row)
                            log(experiment='exports', **row)
                finally:
                    for fd in fds.values():
                        os.close(fd)
            if 'depth' in experiments:
                for qd in (int(x) for x in a.depths.split(',')):
                    p.backend.close()
                    p.backend = RngdDmabufBackend(a.library, f'/dev/rngd/npu{chip}ch0', qd, a.descriptors)
                    write, read, rs, rd = api(p)
                    for n in SIZES:
                        row = dict(queue_depth=qd, **verified_round_trip(p, n, write, read, a.reps))
                        result['depth'].append(row)
                        log(experiment='depth', **row)
                p.backend.close()
                p.backend = RngdDmabufBackend(a.library, f'/dev/rngd/npu{chip}ch0', a.queue_depth, a.descriptors)
            if 'slices' in experiments:
                for s in (int(x) * MiB for x in a.slice_sweep.split(',')):
                    write, read, rs, rd = api(p, s, 0)
                    for n in SIZES:
                        row = dict(slice_MiB=s // MiB, exact_exports=False, **verified_round_trip(p, n, write, read, a.reps))
                        result['slices'].append(row)
                        log(experiment='slices', **row)
        finally:
            p.close()
    rows = [r for k in ('pairs', 'exports', 'depth', 'slices') for r in result[k]]
    result['passed'] = bool(rows) and all(r['passed'] for r in rows)
    log(passed=result['passed'], rows=len(rows))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(result, f, indent=1)


if __name__ == '__main__':
    main()
