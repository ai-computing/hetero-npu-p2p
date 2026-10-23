# DMA-BUF safety validation — 2026-10-23

The three safety fixes cover uncertain DMA completion, configured accelerator identities, and overlapping CUDA
destinations. Validation used CPU tests and a native fault-injection harness; no accelerator was accessed.

## Environment and reproduction

- Windows checkout: `C:\Users\USER\orca\scratch\ecosystem-review-20261023\hetero-npu-p2p`
- WSL distribution: `Ubuntu` (Ubuntu 22.04), Python 3.10.12, GCC 11.4.0.
- WSL checkout: `/mnt/c/Users/USER/orca/scratch/ecosystem-review-20261023/hetero-npu-p2p`
- Local rm-core source: `/mnt/c/Users/USER/orca/scratch/ecosystem-review-20261023/rm-core/src`
- Package metadata: actual local wheels installed into `/tmp/hetero-p2p-safety-packages`; rm-core reports 0.1.2.
  Source directories precede that installation on `PYTHONPATH`, so tests exercise the current checkout.

The metadata installation was needed for entry-point discovery. From PowerShell, create it with:

```powershell
wsl -d Ubuntu -- python3 -m pip install --target /tmp/hetero-p2p-safety-packages --no-deps --no-build-isolation /mnt/c/Users/USER/orca/scratch/ecosystem-review-20261023/rm-core /mnt/c/Users/USER/orca/scratch/ecosystem-review-20261023/hetero-npu-p2p
```

Exact full-suite command, also from PowerShell:

```powershell
wsl -d Ubuntu -- env PYTHONPATH=/mnt/c/Users/USER/orca/scratch/ecosystem-review-20261023/hetero-npu-p2p/src:/mnt/c/Users/USER/orca/scratch/ecosystem-review-20261023/rm-core/src:/tmp/hetero-p2p-safety-packages python3 -m unittest discover -s /mnt/c/Users/USER/orca/scratch/ecosystem-review-20261023/hetero-npu-p2p/tests -v
```

To run only the 13 new safety tests, add `-p test_dmabuf_safety.py` to that command.

## Results

Final full-suite result: **155 tests, 149 passed, 6 skipped, no failures or errors**.

The skipped tests are the three `test_gpu_device.DeviceTests` cases and the three
`test_nixl_hetero_npu.HeteroNpuNixlTest` cases. They require accelerator hardware and native libraries.
Existing warnings remain in `test_gpu_plan.BackendTests.test_library_manifest_is_checked` (unclosed fixture files)
and `test_airm.P2PTransportTests.test_no_hardware_matches_cli_main_module` (runpy module already imported).
`git diff --check` passed.

An earlier full-suite run using only source directories failed
`test_airm.P2PTransportTests.test_llmd_selects_transport_by_name` because entry-point metadata was absent.
Installing the local packages into the temporary target above resolved that environment failure.

## Regression evidence

Before implementation, tests reproduced the defects: the native fatal-submit/fatal-drain path returned ordinary
`-EIO`; same-vendor routes with wrong device indices were accepted in both directions; overlapping destinations
through distinct exports and region views were accepted. These tests passed after the fixes.

Additional failing-then-passing regressions covered cached plans retaining the current request owner, another
backend reusing a quarantined region, interruption without a native receipt, and `release_region()` losing its
registry entry when quarantine refused closure. Positive checks cover matching nonzero device indices,
adjacent destinations, and identical addresses on different CUDA devices.

The GCC harness compiles the actual `native/rngd_dmabuf.c` submission loop with controlled ring memory and syscall
failures. It verifies descriptor retention and refusal to close while completion is unknown, recovery remaining
false without CQEs, recovery succeeding after an error CQE, and the broken ring continuing to reject transfers.
Its IOPOLL case first failed when the recovery probe omitted `GETEVENTS`, then passed after the probe requested
events with `min_complete=0`. Drained failures and failures before submission still return ordinary `-EIO`.

## Limits and deployment requirement

The harness uses clearly labeled test-only vendor UAPI structures. It does **not** verify vendor structure layout,
the installed RNGD driver ABI, real CUDA export behavior, real DMA, or throughput. The Python memory tests use
Linux memfd-backed stand-ins. No Windows memfd compatibility shim was used for this run.

This update requires **native helper ABI 3**. ABI 2 libraries are rejected; rebuild the library and its matching
manifest against the real RNGD 2026.4.0 headers before deployment:

```bash
python native/build.py --target rngd-dmabuf --driver-root /usr/src/furiosa-driver-rngd-2026.4.0
```

Unknown completion retains the native descriptors, ring, request owners, regions and exports until actual CQEs
prove quiescence. If those CQEs never arrive, quarantine remains for the process lifetime. Neither closing a
backend nor receiving a syscall error establishes a fence. No production hardware deployment was performed.

## Follow-up: real GPU/RNGD hardware

The ABI 3 library was built from `d5eabfa3a5913791d0d9538584d17c97d7d557fb` against the installed
RNGD **2026.4.0** headers, using `native/build.py --target rngd-dmabuf`. Library SHA256:
`f0a6c4bccfc7218d72e799925bd32da501cb3577088f068029afbb91c2c25507`;
native source SHA256: `12718f4c3fa631bec1156ee3441fcbb90b4fd85bb589f17ffb035bfa78f4cb3c`.
The generated manifest hash matched the library and `rd_abi()` returned **3**.

On the GPU/RNGD host, Linux 6.8.0-106-generic, NVIDIA open driver 580.178.04, PyTorch 2.10.0+cu128,
physical A100 80 GB GPU **1** and RNGD **1** were used. Device-handle checks and `furiosa-smi ps`
confirmed the selected RNGD was idle before each phase. Another process had used RNGD 0, which was excluded.
All work ran from a separate clone and scratch directory; installed environments and serving processes were untouched.

All **three** `tests/test_gpu_device.py` cases passed with **zero skips**: API round trip, invalid descriptors
rejected without submission, and eight-command transfers across interrupt queue depths 1/2/3/8 plus polled
depths 1/4. Every successful receipt reported all commands complete and the data compared equal.
[Raw test result](gpu-rngd/abi3-device-tests-20261023.json).

The existing `examples/gpu_rngd_dmabuf.py` then ran only pair `1:1`, five calls per direction/size:

| MiB | GPU → RNGD GB/s | RNGD → GPU GB/s |
|---:|---:|---:|
| 2 | 12.87 | 13.82 |
| 4 | 16.16 | 15.89 |
| 16 | 17.45 | 18.24 |
| 64 | 14.63 | 20.74 |
| 256 | 14.75 | 22.19 |

Each row passed sampled CPU BAR checks, full GPU destination comparison and an independent marker written
through the RNGD BAR. [Raw measurements](gpu-rngd/abi3-20261023.json) retain all timings and environment fields.
These five-repetition medians establish functionality on this pair; they do not establish a performance
improvement or cover every pair. The test process exited and closed its backends and exports.

An additional actual-memory check used `DmabufKVTransport` with rm-core's token-major BF16 layout
(4 KiB per token) and the same physical pair. Wrong GPU and RNGD tier indices were rejected before exports;
a 2 MiB adapter round trip was byte-exact and emitted four transfer metric samples. Overlapping original
CUDA destinations were rejected before an additional export or DMA.
[Raw adapter result](gpu-rngd/abi3-adapter-20261023.json). These buffers were synthetic test regions on
an idle RNGD, not an inference engine's allocated KV cache.

No fatal DMA failure was injected into real hardware. Uncertain-completion retention and recovery remain
covered by the native fault harness; successful normal transfers do not prove behavior after an actual driver fault.
