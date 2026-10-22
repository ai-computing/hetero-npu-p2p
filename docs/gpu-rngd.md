# GPU ↔ RNGD over DMA-BUF (`hetero_p2p.gpu`)

The RNGD PDMA engine reads and writes CUDA memory exported as DMA-BUF (#6, #7). The RNGD is always the DMA master:
the GPU-initiated path is refused, because `cudaHostRegisterIoMemory` on the RNGD BAR returns cudaError 800.

**Requirements**
- The NVIDIA **open** kernel module (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 1`). With the proprietary module,
  `cuda_region()`, the path table and both adapters raise `ProprietaryModuleError` before anything is exported or
  submitted. Nothing falls back to host staging.
- RNGD driver 2026.4.0 with `NPU_PDMA_CAPS_DMABUF`.
- The native library `build/librngd_dmabuf.so`:
  `python native/build.py --target rngd-dmabuf --driver-root /usr/src/furiosa-driver-rngd-2026.4.0`
  (raw io_uring syscalls, so no liburing is needed).

```python
from hetero_p2p.gpu import RngdDmabufBackend, cuda_region
backend = RngdDmabufBackend('build/librngd_dmabuf.so', '/dev/rngd/npu0ch0', queue_depth=4)
region = cuda_region(kv)                                   # contiguous, 4 KiB aligned CUDA tensor; exported once
backend.write(region, npu_offset)                          # GPU -> RNGD DRAM; returns after every CQE
backend.read(npu_offset, region, region_offset=0, length=n)  # RNGD DRAM -> GPU
```

## API

| Piece | Contract |
|---|---|
| `cuda_region(tensor)` | `CudaRegion` (memory kind `cuda`) registered once per tensor and reused. It holds the tensor, so the memory cannot be freed while registered. `release_region(tensor)` closes the exports. |
| `CudaRegion` exports | Transfers larger than one slice use **16 MiB slices** (configurable, at most 256 MiB). A transfer no larger than one slice uses an export of exactly its own range. Both kinds are made on first use and kept; up to 256 exact-range exports per region, least recently used evicted, never while a plan names them. |
| `RngdDmabufBackend` | One device FD and one io_uring per RNGD, kept open across transfers (in-process, no helper process). `queue_depth` commands in flight (default 4), `descriptors_per_command` 2 MiB descriptors per command (default 64), one DMA-BUF per command. |
| `read` / `write` / `copy` | Return a receipt (`cqe_result`, `completed`, `requested`, `bytes_requested`, `seconds`, …) only after every CQE was reaped, as `RngdPdmaBackend`. Any failure raises. The native run validates the whole request before the first submission, submits nothing after a failed command, and still reaps every submitted one, so nothing is left in flight. `write` synchronizes the GPU first. |
| `paths.resolve(src, dst)` | Per-pair path table (below). |
| `DmabufPDTransport` | `opt_prime.accel.pd` `Transport` (`supports` / `transfer` / `finish` / `recover`) for GPU prefill → RNGD decode and back. |
| `DmabufKVTransport` | ai-rm `KVTransport`, entry point `airm.kv_transports:p2p-dmabuf`, with `bind(request_id, DmabufBinding(...))` like `p2p`. |

### Path table (`hetero_p2p.paths`)

| src → dst | Status | Initiator, mechanism | Host requirement |
|---|---|---|---|
| RNGD → ATOM | supported | RNGD PDMA `DMABUF_READ` into an ATOM export (`RngdPdmaBackend`) | ATOM export |
| RNGD → GPU | supported | RNGD PDMA `DMABUF_READ` into a CUDA export (`RngdDmabufBackend`) | NVIDIA open module |
| GPU → RNGD | supported | RNGD PDMA `DMABUF_WRITE` from a CUDA export (`RngdDmabufBackend`) | NVIDIA open module |
| GPU ↔ ATOM | blocked | no host with both devices | — |
| ATOM → RNGD, RNGD ↔ RNGD | unvalidated | not measured | — |
| anything ↔ host | blocked | host memory is a staging leg; stage explicitly | — |

## Measurements (2026-10-22)

**Host:** npu-node-2: 2× A100 80GB PCIe and 2× RNGD, each device on its own root port, IOMMU on.
**Drivers and software:** NVIDIA 580.178.04 open kernel module, RNGD 2026.4.0, kernel 6.8, torch 2.8.0+cu128.

**Method:** `examples/gpu_rngd_dmabuf.py` on Kubernetes, one pair at a time. The chip must be idle: the RNGD side
is raw DRAM at 4 GiB. GB/s = 10⁹ B/s, from the median of 5 API calls, timed around `backend.write` / `backend.read`
(planning, GPU sync, native run, fence). The `native_*` fields in the JSON time the native run alone.

**Checks, every row:**
- GPU → RNGD: 66 random 4 KiB pages, including the first and last, are read back by the CPU through BAR4.
- RNGD → GPU: the destination is poisoned before each of the 5 reads and compared byte for byte. The last page is
  a CPU-written marker in RNGD DRAM.

All 126 rows were byte-exact. Raw data:
[`pairs-20261022.json`](gpu-rngd/pairs-20261022.json) (default configuration) and
[`sweeps-20261022.json`](gpu-rngd/sweeps-20261022.json) (sweeps).

### All pairs, default configuration (16 MiB slices + exact-range exports, queue depth 4, 64 descriptors)

GB/s, GPU → RNGD / RNGD → GPU:

| pair | 2 MiB | 4 MiB | 16 MiB | 64 MiB | 256 MiB | 1 GiB |
|---|---:|---:|---:|---:|---:|---:|
| RNGD 0 – GPU 0 | 7.5 / 12.7 | 7.2 / 14.1 | 10.1 / 16.1 | 11.6 / 17.2 | 12.0 / 17.9 | 14.7 / 20.7 |
| RNGD 0 – GPU 1 | 7.0 / 12.9 | 9.4 / 11.6 | 10.3 / 16.6 | 9.6 / 17.7 | 11.9 / 17.7 | 13.7 / 18.3 |
| RNGD 1 – GPU 0 | 8.1 / 12.7 | 9.8 / 14.4 | 10.3 / 16.4 | 11.3 / 16.7 | 12.1 / 17.6 | 12.5 / 19.0 |
| RNGD 1 – GPU 1 | 8.2 / 12.0 | 9.4 / 11.7 | 9.9 / 16.0 | 12.3 / 16.8 | 12.7 / 18.6 | 12.2 / 18.6 |
| *stage 1 (#6), export = transfer* | *10.6 / 17.0* | | | *9.4 / 18.9* | *13.0 / 19.1* | *10.3 / 14.8* |

From 16 MiB up, the results match or exceed stage 1:
- RNGD → GPU: 16.0–20.7 GB/s.
- GPU → RNGD: 9.6–14.7 GB/s.
- At 1 GiB, both directions are faster than stage 1: 12.2–14.7 / 18.3–20.7 vs 10.3 / 14.8 GB/s.

At 2–4 MiB, the results are **below** stage 1:
- API time: 7.0–8.2 / 12.0–12.9 GB/s at 2 MiB.
- Native run only: 7.7–9.1 / 14.1–14.9 GB/s.

A 2 MiB transfer takes about 0.15 ms. A fixed cost per transfer of tens of µs (one command, the driver's
attach/map, the ring round trip; for writes, also the GPU sync) is therefore a large share at that size.
Same-NUMA and cross-NUMA pairs agree within run-to-run noise, which is about ±10% between runs.

### Cached exports vs alternatives (pair RNGD 0 – GPU 0)

| export | 2 MiB | 4 MiB | 16 MiB | 64 MiB | 256 MiB | 1 GiB |
|---|---:|---:|---:|---:|---:|---:|
| **cached region (default)** | **8.0 / 12.0** | **9.5 / 14.0** | 8.9 / 17.2 | **12.8 / 18.3** | **14.7 / 19.2** | **13.8 / 20.4** |
| fresh export per transfer | 4.9 / 8.1 | 6.5 / 9.8 | 9.5 / 13.8 | 10.9 / 16.1 | 11.8 / 17.1 | 13.5 / 18.6 |
| one 1 GiB export reused | 0.2 / 0.2 | 0.4 / 0.4 | 1.5 / 1.6 | 4.1 / 5.0 | 7.9 / 10.9 | 10.8 / 11.7 |

**Exporting once and reusing:**
- Compared with exporting per transfer, it gains 1.4–1.6× at 2–4 MiB and 2–25% from 16 MiB up. One cell is
  6% lower (GPU → RNGD at 16 MiB), within run-to-run noise.
- Mapping one large export on every command costs 40–60× at 2 MiB. This is the effect #6 found: the driver maps
  the whole exported buffer on every command.
- A region is exported once per slice and reused across sizes and repetitions. Per pair, the run made 134 exports
  for 2 × 1 GiB of buffers and 60 transfers.

### Sweeps (pair RNGD 0 – GPU 0, earlier run with 256 MiB slices as the default)

| queue depth | 2 MiB | 64 MiB | 256 MiB | 1 GiB |
|---|---:|---:|---:|---:|
| 1 | 5.3 / 9.7 | 9.2 / 17.4 | 8.2 / 14.9 | 8.3 / 14.6 |
| 2 | 9.0 / 12.4 | 9.2 / 17.7 | 12.3 / 18.2 | 13.2 / 20.6 |
| 4 | 6.6 / 13.8 | 9.0 / 17.0 | 11.7 / 17.7 | 14.2 / 20.9 |
| 8 | 8.5 / 12.7 | 9.0 / 17.2 | 12.3 / 18.0 | 13.4 / 21.0 |

| slices (no exact-range exports) | 2 MiB | 64 MiB | 256 MiB | 1 GiB |
|---|---:|---:|---:|---:|
| 16 MiB | 5.5 / 8.1 | 13.0 / 20.6 | 14.9 / 21.5 | 14.8 / 21.9 |
| 64 MiB | 2.4 / 3.1 | 9.3 / 17.2 | 14.0 / 20.4 | 14.6 / 21.6 |
| 256 MiB | 0.8 / 0.9 | 7.7 / 12.2 | 12.1 / 18.3 | 14.2 / 20.9 |

**What the sweeps show:**
- **Queue depth.** Depth 2 or more is needed once a transfer spans several commands: at depth 1 a 1 GiB transfer
  drops to 8.3 / 14.6. Beyond depth 2 nothing changes.
- **Slice size.** Smaller slices win at every size, because each command maps less. 16 MiB slices for large
  transfers, plus exact-range exports for transfers up to one slice, became the default.

## Limits

- Transfers move bytes; both sides must already hold the same physical representation (no dtype or layout
  conversion).
- Ranges must be 4 KiB aligned; a CUDA tensor must start and end on 4 KiB boundaries.
- At most 4 GiB per command (driver limit), 47.5 GiB of RNGD DRAM.
- The measurements write raw RNGD DRAM, not an SDK allocation. In an engine, the RNGD offset comes from the
  RNGD-side allocation that owns the KV cache.
- The proprietary-module rejection is tested on CPU with the module's version string. The host runs the open module
  and was not switched back for this.
