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
  (raw io_uring syscalls, so no liburing is needed). This safety update requires native ABI 3; rebuild the
  library and its manifest together. ABI 2 libraries are rejected before opening a device.

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
| `RngdDmabufBackend` | One device FD and one io_uring per RNGD, kept open across transfers (in-process, no helper process). `queue_depth` commands in flight (default 4), `descriptors_per_command` 2 MiB descriptors per command (default 64), one DMA-BUF per command. `poll=True` (default): completion-polled ring. `spread` (default = `poll`): a transfer too small to fill the queue is cut into about `queue_depth` commands of at least 256 KiB. Prepared plans of repeated copies are cached (`cache=True`). |
| `read` / `write` / `copy` | Return a receipt (`cqe_result`, `completed`, `requested`, `bytes_requested`, `seconds`, …) only after every CQE was reaped. Any failure raises; unknown DMA completion quarantines the allocations (below). The native run validates the whole request before the first submission and submits nothing after a failed command. `write` synchronizes the GPU first (`ready=None`), or not (`ready=True`, the caller already did), or waits on a `torch.cuda.Event` / stream (`ready=event`). `copy` moves several ranges under one fence (the batch API). Each receipt has a `profile`. |
| `paths.resolve(src, dst)` | Per-pair path table (below). |
| `DmabufPDTransport` | `opt_prime.accel.pd` `Transport` (`supports` / `transfer` / `finish` / `recover`) for GPU prefill → RNGD decode and back. |
| `DmabufKVTransport` | ai-rm `KVTransport`, entry point `airm.kv_transports:p2p-dmabuf`, with `bind(request_id, DmabufBinding(...))` like `p2p`. |

If a fatal ring error prevents draining submitted commands, `rd_run` returns `RD_UNCERTAIN` (-4096), and Python
raises `DmaCompletionUncertain`. The native descriptors, ring, current request owners, regions and exports stay
alive, even if the caller drops the backend. Quarantined regions cannot be used, closed or have their exports
evicted. `DmabufPDTransport.finish()` refuses that request and `recover()` returns `False` while completion is
unknown. A failed command with an observed CQE is drained and does not need this quarantine.

`backend.recover()` (also called by PD recovery) polls the existing ring without submitting more work. It returns
`True` and releases the quarantine only after every outstanding CQE has actually been observed, including error
CQEs. The damaged ring still refuses transfers; close it and create a new backend. If those CQEs never arrive,
the quarantine remains for the process lifetime. Closing the backend or receiving a syscall error is not a fence.
Direct `transfer(DmabufPlan(...))` users must keep their raw FDs and allocations alive until recovery proves
quiescence; plans from `plan_copies` additionally track their CUDA regions.

The ai-rm adapter requires the RNGD tier's full `furiosa:<index>` identity to match its configured
`/dev/rngd/npu<index>ch0`, and the GPU tier's `nvidia:<index>` identity to match the bound region's CUDA device.
No logical-to-physical device remapping is inferred. Read batches reject overlapping original CUDA address ranges
per device before exporting, including overlaps between different `CudaRegion` views and exact-range exports.

CPU regressions include a native ring fault-injection harness compiled with GCC and test-only UAPI structures.
These checks exercise error handling and ownership; they do not validate the installed driver ABI or real DMA.

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

### All pairs, #8 configuration (16 MiB slices + exact-range exports, queue depth 4, 64 descriptors, interrupt ring)

The current defaults are faster; see [Small transfers](#small-transfers-profile-and-levers-7-24-mib).

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

## Small transfers: profile and levers (#7, 2–4 MiB)

#8 left 2–4 MiB below stage 1: 7.0–8.2 / 12.0–12.9 GB/s at 2 MiB, against 10.6 / 17.0.

**Method:** the `small` experiment of `examples/gpu_rngd_dmabuf.py`, on all 4 pairs, median of 11 calls.
- Each receipt carries a `profile`; the table shows medians in µs, pair RNGD 0 – GPU 0, 2 MiB.
- The profile stages are:
  - `sync`: GPU sync before a write.
  - `plan`: planning, or the plan-cache lookup.
  - `validate`: native validation.
  - `submit`: the first `io_uring_enter`, in which the driver issues the commands.
  - `fence`: the wait until every CQE is reaped.
- The host-side remainder (ctypes, receipt) is about 5 µs.
- Every row was byte-exact, with the same BAR4 and marker checks as above.

| configuration | direction | sync | plan | submit | fence | total | GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| #8 behaviour (one command, interrupt ring) | GPU → RNGD | 6.7 | 16.6 | 28.4 | 203.4 | 266.9 | 7.7 |
| | RNGD → GPU | — | 22.9 | 31.6 | 121.7 | 192.4 | 10.5 |
| + `ready=True`, cached plan | GPU → RNGD | 0.3 | 2.5 | 26.8 | 200.7 | 244.9 | 8.3 |
| | RNGD → GPU | — | 1.9 | 27.8 | 119.3 | 160.0 | 12.6 |
| **new default** (polled ring, spread over 4 commands, cached plan) | GPU → RNGD | 5.5 | 1.8 | 60.8 | 71.8 | 151.4 | 13.4 |
| | RNGD → GPU | — | 1.6 | 63.0 | 66.7 | 142.2 | 14.1 |
| new default, queue depth 8 | RNGD → GPU | — | 1.7 | 120.3 | 75.5 | 210.5 | 9.7 |
| **batch API**, 8 × 2 MiB under one fence | RNGD → GPU | — | 3.0 | 63.8 | 728.2 | 810.1 | 20.4 |

**Where the time goes.**
- **Host side.** The device sync (~7 µs) and planning (~17–23 µs) shrink to ~2 µs together with `ready=True` and the
  plan cache.
- **One command on one PDMA channel.** That is the floor of the #8 behaviour: ~120 µs for RNGD → GPU and ~200 µs
  for GPU → RNGD.
- **Spreading a small transfer over `queue_depth` smaller commands.**
  - The commands run on separate channels, which cuts the DMA (`fence`) to ~70 µs.
  - In return, the driver's per-command issue cost grows, because it attaches and maps the DMA-BUF again for every
    command. Four commands take ~61 µs in `submit`; eight take ~120 µs, which is why queue depth 8 is slower.
- **Spreading needs the polled ring.**
  - On an interrupt-completed ring, a command whose channel is busy blocks the submitter on the channel lock.
  - The result was bimodal: 12.8 / 13.4 GB/s on one pair, but 7.4–7.5 / 10.0–10.2 GB/s on the other three
    ([`small-irq-spread-20261022.json`](gpu-rngd/small-irq-spread-20261022.json)).
  - The polled ring (`IORING_SETUP_IOPOLL`, already supported by the driver's `uring_cmd_iopoll`) instead waits for
    a busy channel inside the poll loop.
  - `poll=True` is therefore the default, and `spread` follows `poll`.
- **The batch API** (`backend.copy(direction, [DmabufCopy, ...])`) overlaps the issue of later commands with the DMA
  of earlier ones. The fixed cost is paid once per call.

### Result, all 4 pairs (GB/s, GPU → RNGD / RNGD → GPU)

| call | 2 MiB | 4 MiB | 16 MiB | 64 MiB | 256 MiB | 1 GiB |
|---|---:|---:|---:|---:|---:|---:|
| stage 1 (#6) | 10.6 / 17.0 | | | 9.4 / 18.9 | 13.0 / 19.1 | 10.3 / 14.8 |
| #8 behaviour | 7.4–7.8 / 10.5–11.5 | 8.4–8.8 / 13.1–13.6 | 9.2–9.7 / 14.5–15.6 | | | |
| **default single call** | 12.7–13.6 / 14.0–14.2 | 14.3–15.4 / 15.3–15.8 | 16.3–16.8 / 17.2–17.4 | 14.3–14.7 / 20.1–20.3 | 15.0 / 22.0 | 15.1 / 22.5 |
| default, `ready=True` | 13.7–14.2 / 13.8–14.4 | 15.5–15.8 / 15.5–15.7 | 16.7–16.8 / 17.2–17.4 | | | |
| **batch API, 8 ranges of this size** | 20.5–20.6 / 20.4–20.5 | 20.9–21.0 / 20.9–21.1 | 14.3 / 21.4–21.5 | | | |

Raw data: [`small-20261022.json`](gpu-rngd/small-20261022.json).

**Against stage 1:**
- **Batch API:** 2–4 MiB transfers reach and exceed stage 1 in both directions.
- **Single call, GPU → RNGD:** exceeds stage 1 (12.7–13.6 vs 10.6 GB/s at 2 MiB).
- **Single call, RNGD → GPU:** stays at 14.0–14.2 GB/s at 2 MiB, 17% below stage 1's 17.0. The native run alone
  reaches 15.3–15.5.
- **Why the single call stops there.** A lone 2 MiB call cannot hide the driver's per-command attach and map behind
  DMA: about 60 µs of a 140 µs call.
- **What would remove it:** an attachment that persists across commands. That is a driver change, out of scope
  here.
- **Large transfers also gained** from the polled ring: 15.1 / 22.5 GB/s at 1 GiB.

## Limits

- Transfers move bytes; both sides must already hold the same physical representation (no dtype or layout
  conversion).
- Ranges must be 4 KiB aligned; a CUDA tensor must start and end on 4 KiB boundaries.
- At most 4 GiB per command (driver limit), 47.5 GiB of RNGD DRAM.
- The measurements write raw RNGD DRAM, not an SDK allocation. In an engine, the RNGD offset comes from the
  RNGD-side allocation that owns the KV cache.
- The polled ring keeps the calling thread busy in the kernel for the duration of a transfer.
- The proprietary-module rejection is tested on CPU with the module's version string. The host runs the open module
  and was not switched back for this.
