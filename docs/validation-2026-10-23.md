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
