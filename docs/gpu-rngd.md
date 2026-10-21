# GPU ↔ RNGD over DMA-BUF (`hetero_p2p.gpu`)

The RNGD PDMA engine reads and writes CUDA memory exported as DMA-BUF. The RNGD is always the DMA master: the
GPU-initiated path is refused, because `cudaHostRegisterIoMemory` on the RNGD BAR returns cudaError 800.

**Requirements:**
- The NVIDIA **open** kernel module, so that `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 1`. The proprietary module
  reports 0, and exporting then fails with an explicit error. Nothing falls back to the host.
- RNGD driver 2026.4.0.
- The native helper `build/rngd-dmabuf`: `python native/build.py --liburing-root <liburing>`.

```python
from hetero_p2p.gpu import CudaRegion, RngdDmabufBackend, plan_copy
region = CudaRegion.from_tensor(gpu_buffer)            # 4 KiB aligned CUDA tensor; exported lazily, cached
backend = RngdDmabufBackend('build/rngd-dmabuf', '/dev/rngd/npu0ch0')
backend.transfer(plan_copy('write', region, 0, npu_offset, nbytes))   # GPU -> RNGD DRAM
backend.transfer(plan_copy('read', region, 0, npu_offset, nbytes))    # RNGD DRAM -> GPU
region.close()
```

**API:**
- `DmabufPlan` holds 4 KiB-aligned spans of up to 2 MiB, at most 8192 spans per plan, so up to 16 GiB.
- The helper groups the spans into commands of up to 64 descriptors. It starts a new command wherever the
  DMA-BUF changes.
- Every transfer returns a receipt only after all commands have completed. A failure leaves nothing in flight:
  the helper drains every command before it exits.

## Why the region is exported in 16 MiB slices

The RNGD driver attaches and maps the **whole** DMA-BUF on every command, so the cost of a command grows with the
size of the exported buffer, not with the bytes it moves. Exporting 1 GiB as one buffer costs about 7 ms per command
(#6). The table shows GB/s for GPU → RNGD / RNGD → GPU, best of 5. Hardware: RNGD npu0 and A100 (GPU 1), byte-exact
in every case:

| slice | 2 MiB | 16 MiB | 64 MiB | 256 MiB | 1 GiB |
|---|---:|---:|---:|---:|---:|
| 256 MiB | 0.74 / 0.78 | 4.35 / 4.95 | 7.49 / 12.23 | 12.65 / 18.06 | 13.32 / 21.36 |
| 64 MiB | 2.35 / 2.46 | 8.48 / 11.46 | 9.56 / 17.47 | 14.01 / 20.29 | 13.94 / 20.73 |
| **16 MiB (default)** | **6.26 / 6.79** | **10.68 / 17.86** | **12.13 / 20.16** | **14.07 / 20.69** | **14.17 / 20.77** |

For comparison:
- Stage 1 (#6) exported each transfer as exactly its own size: 10.6 / 17.0 at 2 MiB and 10.3 / 14.8 at 1 GiB.
- RNGD → host runs at 11–14 GB/s.

The 16 MiB slices keep the large-transfer bandwidth without re-exporting per transfer. Small transfers still pay
the mapping cost of one slice.

## Validation

`examples/gpu_rngd_dmabuf.py` covers 2 MiB to 1 GiB in both directions. Its checks:
- the GPU side is compared byte for byte;
- the RNGD side is spot-checked by the CPU through the BAR4 window (64 random pages);
- a 4 KiB marker written into RNGD DRAM by the CPU must arrive on the GPU.

The recorded run is in `gpu-rngd-validation.json`. The CPU tests in `tests/test_gpu_plan.py` cover region slicing,
plan validation, FD-alias overlap, and the helper rejecting invalid plans before it opens the device.
