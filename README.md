# hetero-npu-p2p

**Completion-fenced KV-cache transport for heterogeneous NPUs on the same Linux host.**

`hetero-npu-p2p` separates peer-to-peer transfer planning, buffer ownership, and decode admission from model execution. Its current hardware backend transfers KV data from a Furiosa RNGD into a Rebellions ATOM-owned DMA-BUF. Experimental serving examples connect this data path to llm-d.

The Python contracts are model-independent. Hardware adapters are version-specific, and the current implementation is a research prototype rather than a general-purpose inference runtime.

## Features

- **Explicit transfer plans:** describe source allocations, destination buffers, and scatter-copy spans.
- **Completion-fenced admission:** start decode only after all DMA completion entries have been checked.
- **Buffer lifetime management:** retain allocation owners through transfer and decode; preserve failed leases for diagnosis.
- **Physical KV layout planning:** construct prefix-copy plans for token-major caches.
- **Serving primitives:** request validation, capacity control, cache leases, and iteration-level scheduling.
- **Experimental llm-d integration:** route prefill/decode control requests while KV payloads move through the local P2P path.
- **Validation tools:** CPU contract tests, hardware compatibility checks, benchmark analysis, and board-energy integration.

## How it works

```text
Prefill model adapter                           Decode model adapter
        │                                                ▲
        ▼                                                │
 RNGD-owned KV ── transfer plan ── RNGD PDMA ──► ATOM-owned KV
                                      │          DMA-BUF
                                      ▼
                              completion check
                                      │
                                      └──► decode admission

llm-d / application: request routing and control
model adapters: compute, KV representation, and device synchronization
hetero-npu-p2p: transfer contracts, completion checks, and cache ownership
```

The destination allocation is exported by ATOM and imported by the RNGD transfer path. This backend does not stage the KV payload through host memory or silently fall back to host copies. Model adapters may still perform representation conversion or use host memory for validation and other work.

## Requirements

| Component | Current requirement or validation scope |
|---|---|
| Python | 3.10 or later; core package uses the standard library |
| Platform | Linux x86-64 for the hardware backend |
| Topology | Source and destination on the same host; P2P support must be verified on the target system |
| Source | Furiosa RNGD, driver 2026.4.0 |
| Destination | Rebellions ATOM, driver 3.0.0, checked thunk-library ABI |
| Destination memory | Live, exportable eager allocation; static compiled resident caches are not supported by this adapter |
| Native helper | GCC, matching RNGD driver development headers, and a built liburing library |
| Model examples | Additional vendor SDKs, model artifacts, and model-specific dependencies |

Version checks reject unknown environments. Matching versions alone do not establish that a device is available or that P2P works on a particular PCIe topology. See [support and ownership contracts](docs/SUPPORT.md).

## Installation

From a checkout of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

This installs the Python package and the `hetero-p2p` command. It does not install vendor drivers, SDKs, model weights, or the native DMA helper. No package-index installation is assumed.

For hardware transfers, build the helper against the locally installed driver headers and a built liburing tree:

```bash
python native/build.py \
  --driver-root /path/to/furiosa-driver-rngd-2026.4.0 \
  --liburing-root /path/to/liburing-2.6
```

The build produces `build/rngd-pdma` and its build manifest. Keep them together: the backend checks the helper against the manifest before use.

## Quick start: no NPU required

Run the contract tests on Linux:

```bash
python -m unittest discover -s tests -v
```

These tests cover transfer bounds, buffer aliasing, ownership transitions, incomplete completions, scheduling, and serving contracts. They use test doubles and host-backed fixtures; passing them does not demonstrate hardware DMA.

The following example exercises a transfer session with a mock backend. It does **not** copy data:

```python
import os

from hetero_p2p import (
    CopySession, CopySpan, DestinationAllocation,
    SourceAllocation, TransferPlan,
)

class MockBackend:
    def transfer(self, plan):
        return {
            "cqe_result": 0,
            "completed": plan.commands,
            "requested": plan.commands,
            "bytes_requested": plan.bytes,
        }

fd = os.memfd_create("contract-example")
os.ftruncate(fd, 4096)
try:
    source = SourceAllocation(64, 4096, object(), "example-format")
    destination = DestinationAllocation(fd, object(), "example-format")
    plan = TransferPlan(
        source, (destination,), (CopySpan(0, 0, 0, 4096),),
        descriptors_per_command=1,
    )
    session = CopySession(MockBackend())
    session.admit("request-1", plan, notify=lambda: None)
    session.start_decode("request-1")
    # A real integration must wait for device decode completion here.
    session.finish_decode("request-1", passed=True)
    assert session.state == "idle"
finally:
    os.close(fd)
```

The synthetic address, host file descriptor, and placeholder owners above are valid only for this mock example. Hardware transfers require addresses and exported descriptors obtained from live SDK allocations.

## Hardware integration

Check the installed environment, using the intended physical device path:

```bash
hetero-p2p doctor --device /dev/rngd/npu3ch0
```

`doctor` checks versions, ABI, and device-file compatibility. It does not reserve the device or execute a P2P transfer.

To integrate a model:

1. Create and synchronize the source KV allocation through its SDK.
2. Export a supported destination allocation and retain its owning SDK objects.
3. Describe the actual physical format with a `TransferPlan` or `TokenMajorLayout`.
4. Create a `CopySession` using `RngdPdmaBackend(helper, device=...)` on the owning process's main thread.
5. Admit the request, run decode after verified transfer completion, and release the lease only after verified decode completion.

The backend's device path must match the source allocation. A source address is a raw RNGD DRAM offset, not a CPU pointer. Do not manually close exported descriptors or release SDK allocations while a session owns them. Failed or uncertain transfers poison the session rather than recycling the destination.

See [library API and serving contracts](docs/SERVING_LIBRARY.md), [support details](docs/SUPPORT.md), and [hardware validation](docs/VALIDATION.md).

## llm-d and model examples

The llm-d integration implements a local prefill/decode stage contract. HTTP carries requests and control information; the owner processes manage KV transfer and cache leases. This is not a NIXL/Mooncake connector or a cross-node RDMA transport.

The `examples/koni/` directory contains model-specific research examples for KONI-Llama3.1-8B. They require additional model code, compiled artifacts, and separate vendor environments. They are not a turnkey service installed by `pip`.

Start with:

- [llm-d integration](docs/LLMD.md)
- [HTTP batching setup](docs/HTTP_BATCHING.md)
- [Example configuration](examples/koni/serve.config.example.json)
- [Iteration scheduling](docs/ITERATION_SERVING_KO.md)
- [Independent decode replicas](docs/ITERATION_REPLICAS_KO.md)

Compiled B4/B8 examples and opt-in B12/B14 paths have been exercised. Batch size and context length depend on the compiled artifact and available memory; they are not interchangeable runtime switches.

## Benchmarks

Performance depends on model, batch occupancy, input/output lengths, request arrivals, and hardware allocation. The repository preserves workload-specific results, including regressions and failed experiments.

| Evaluation | Evidence |
|---|---|
| Same-batch host-admitted vs P2P output validation | [Validation record](docs/VALIDATION.md) |
| Batch-width scaling | [B8 results](docs/PD_B8_RESULTS_KO.md), [B12 results](docs/PD_B12_RESULTS_KO.md) |
| B14/context448, saturation and mixed workloads | [Results and limitations](docs/B14_CONTEXT448_KO.md) |
| Scheduling policies and research basis | [Research log](docs/OPTIMIZATION_RESEARCH_KO.md) |
| Memory constraints and unsuccessful configurations | [B16 feasibility](docs/B16_FEASIBILITY_KO.md) |

Transport output agreement is not proof of model-quality equivalence. PD throughput is not necessarily better than standalone inference, and a throughput gain need not improve latency or energy efficiency. Raw local runs, model weights, and compiled artifacts may be excluded from Git; checked-in summaries identify their provenance and limitations.

## Limitations

- Hardware support is limited to the validated adapter/ABI combination and transfer direction.
- Other models require compatible allocation, layout, conversion, and synchronization adapters.
- Multi-node transport, full Kubernetes/EPP/Envoy deployment, general sampling, and production token streaming are outside the validated examples.
- Forced process termination or device reset during DMA is not a supported recovery mechanism.
- Board-power integration excludes host CPU/DRAM and does not establish wall-plug efficiency or MFU.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

For changes to hardware adapters, also verify actual allocation/export behavior, physical layout, completion handling, output correctness, and clean shutdown on the target hardware. Preserve failed runs and record SDK versions, source revisions, and measurement conditions.

| Directory | Contents |
|---|---|
| `src/hetero_p2p/` | Python transport, layout, scheduling, and serving contracts |
| `native/` | RNGD PDMA helper and build script |
| `examples/koni/` | Model-specific integration and benchmark tools |
| `tests/` | CPU contract tests and native validation utilities |
| `docs/` | API guidance, experiment reports, and support boundaries |

## License

A project license has not yet been declared in a `LICENSE` file. Public repository access does not by itself grant an open-source license. Vendor SDKs, driver components, and model artifacts remain subject to their respective terms.
