"""GPU <-> RNGD through hetero_p2p.gpu (#7): byte-exact round trips and bandwidth, 2 MiB - 1 GiB.

One CudaRegion is registered once (exported once as 16 MiB DMA-BUF slices, cached) and reused by every transfer. The
RNGD side is a slice in the middle of the chip's usable DRAM window, checked independently through the BAR4 window
by the CPU. The chip must be idle (no PE held). Needs the NVIDIA open kernel module and build/rngd-dmabuf.

    python examples/gpu_rngd_dmabuf.py --helper build/rngd-dmabuf --chip 0 --gpu 0
"""
import argparse, fcntl, json, mmap, os, struct, sys

ap = argparse.ArgumentParser()
ap.add_argument('--helper', required=True)
ap.add_argument('--chip', type=int, default=0)
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--descriptors', type=int, default=64)
ap.add_argument('--slice-mib', type=int, default=16, help='DMA-BUF export slice size')
a = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
import torch
from hetero_p2p.gpu import CudaRegion, RngdDmabufBackend, dmabuf_supported, plan_copy

WINDOW_DRAM, AVAILABLE_DRAM = 256 << 20, 0xC0104E02      # BAR4 offset of DRAM address 0; _IOWR('N', 2, {u64, u64})
SIZES = (2, 4, 16, 64, 256, 1024)
SPAN = SIZES[-1] << 20

torch.cuda.init()
if not dmabuf_supported(0):
    sys.exit('CUDA DMA-BUF export unsupported: the NVIDIA open kernel module is required')
raw = torch.empty(SPAN + (2 << 20), dtype=torch.uint8, device='cuda')
pad = (-raw.data_ptr()) % (2 << 20)
region = CudaRegion.from_tensor(raw[pad:pad + SPAN], slice_bytes=a.slice_mib << 20)
backend = RngdDmabufBackend(a.helper, f'/dev/rngd/npu{a.chip}ch0')

bar = os.open(f'/dev/rngd/npu{a.chip}bar4', os.O_RDWR | os.O_SYNC)
info = bytearray(16)
fcntl.ioctl(bar, AVAILABLE_DRAM, info, True)
usable = struct.unpack('<QQ', info)[1]
npu = (usable // 2) & ~((2 << 20) - 1)
mm = mmap.mmap(bar, SPAN, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=WINDOW_DRAM + npu)
gbuf = region.owner
rows = []
try:
    for mib in SIZES:
        n = mib << 20
        src = torch.randint(0, 256, (n,), dtype=torch.uint8, device='cuda')
        gbuf[:n].copy_(src); torch.cuda.synchronize()
        w = backend.transfer(plan_copy('write', region, 0, npu, n, descriptors_per_command=a.descriptors), a.reps)
        host = src.cpu().numpy().tobytes()
        offs = [0, n - 4096] + [int.from_bytes(os.urandom(4), 'little') % (n - 4096) & ~4095 for _ in range(62)]
        bar_ok = all(mm[o:o + 4096] == host[o:o + 4096] for o in offs)
        mark = os.urandom(4096)
        mm[n - 4096:n] = mark                      # proves the bytes read back come from RNGD DRAM
        gbuf[:n].zero_(); torch.cuda.synchronize()
        r = backend.transfer(plan_copy('read', region, 0, npu, n, descriptors_per_command=a.descriptors), a.reps)
        back = gbuf[:n].clone()
        rows.append(dict(MiB=mib, gpu_to_rngd_GBps=round(n / w['best_seconds'] / 1e9, 2),
                         rngd_to_gpu_GBps=round(n / r['best_seconds'] / 1e9, 2),
                         first_write_s=round(w['first_seconds'], 5), exports=len(region._fds),
                         rngd_bar_spotcheck=bar_ok, gpu_exact=bool(torch.equal(back[:n - 4096], src[:n - 4096])),
                         marker=back[n - 4096:].cpu().numpy().tobytes() == mark))
        print(json.dumps(rows[-1]), flush=True)
finally:
    mm.close(); os.close(bar); region.close()
ok = all(x['rngd_bar_spotcheck'] and x['gpu_exact'] and x['marker'] for x in rows)
print('RESULT ' + json.dumps(dict(gpu=torch.cuda.get_device_name(0), chip=a.chip, passed=ok, rows=rows)), flush=True)
