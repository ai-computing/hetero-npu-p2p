<div align="center">

<h1>hetero-npu-p2p</h1>

<h3>Completion-fenced KV-cache transport between heterogeneous NPUs on one host</h3>

<p>
  <strong>English</strong> · <a href="README.ko.md">한국어</a>
</p>

<p>
  <a href="https://github.com/ai-computing/ai-rm"><img src="https://img.shields.io/badge/ai--rm-L2%20memory%20%26%20transport-22d3ee" alt="ai-rm layer: L2 memory and transport"></a>
  <a href="https://github.com/ai-computing/rm-core/blob/main/PROTOCOL.md"><img src="https://img.shields.io/badge/protocol-v0.1-7c5cff" alt="Protocol v0.1"></a>
  <a href="ai-rm.yaml"><img src="https://img.shields.io/badge/maturity-experimental-f59e0b" alt="Maturity: experimental"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&amp;logoColor=white" alt="Python 3.10+"></a>
  <a href="#-license"><img src="https://img.shields.io/badge/license-not%20declared-64748b" alt="License: not declared"></a>
</p>

<p>
  <a href="#-highlights"><b>Highlights</b></a> ·
  <a href="#-quick-start"><b>Quick start</b></a> ·
  <a href="#-ai-rm-integration"><b>ai-rm integration</b></a> ·
  <a href="#-results"><b>Results</b></a> ·
  <a href="#-documentation"><b>Docs</b></a>
</p>

</div>

---

**hetero-npu-p2p** moves KV cache from a Furiosa RNGD into a Rebellions ATOM-owned DMA-BUF on the same Linux host,
and admits decode only after every DMA completion entry has been checked. It separates transfer planning, buffer
ownership and decode admission from model execution, and connects the data path to llm-d's prefill/decode stages.
In [ai-rm](https://github.com/ai-computing/ai-rm) it is the **L2 memory & transport** component that provides the
`KVTransport` contract as `airm.kv_transports:p2p`.

It is also a **migration source for the Optimus Prime engine**: KV-transport technology is planned to move into an
Optimus Prime module ([consolidation plan](https://github.com/ai-computing/ai-rm/blob/master/docs/engine-integration.md)),
keeping the same-host scope, ABI constraints, completion fences and explicit failures. Until that consolidation is
accepted, this repository remains the executable path.

> [!WARNING]
> **Experimental research prototype.** Hardware support is limited to the validated RNGD→ATOM adapter/ABI
> combination. P2P output has been verified as exact against a same-batch host-admitted reference at context 512
> (and an opt-in context 448 path); that is evidence of transport correctness, not of model-quality equivalence.
> It is not a general-purpose inference runtime.

## ✨ Highlights

<table>
<tr>
<td width="33%" valign="top">
<b>Completion-fenced admission</b><br>
Decode starts only after all DMA completion entries (CQEs) are checked. Failed or uncertain transfers poison the session instead of recycling the destination.
</td>
<td width="33%" valign="top">
<b>No silent host staging</b><br>
The ATOM destination allocation is exported as a DMA-BUF and written by RNGD PDMA. The backend never falls back to host copies of the KV payload.
</td>
<td width="33%" valign="top">
<b>Explicit transfer plans</b><br>
Source allocations, destination buffers and scatter-copy spans are described up front; <code>TokenMajorLayout</code> builds prefix-copy plans for token-major caches.
</td>
</tr>
<tr>
<td width="33%" valign="top">
<b>Buffer lifetime management</b><br>
Allocation owners are retained through transfer and decode; failed leases are preserved for diagnosis.
</td>
<td width="33%" valign="top">
<b>llm-d stage integration</b><br>
Implements the llm-d <code>pd-sidecar</code> (shared-storage) prefill/decode stage contract: HTTP carries control, KV moves over the local P2P path.
</td>
<td width="33%" valign="top">
<b>Serving primitives and validation</b><br>
Request validation, capacity control, cache leases, iteration-level scheduling, CPU contract tests, hardware checks and board-energy integration.
</td>
</tr>
</table>

## 🏗 Architecture

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

The destination allocation is exported by ATOM and imported by the RNGD transfer path. This backend does not stage
the KV payload through host memory or silently fall back to host copies. Model adapters may still perform
representation conversion or use host memory for validation and other work. The Python contracts are
model-independent; hardware adapters are version-specific.

<details>
<summary><b>Requirements</b></summary>
<br>

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

Version checks reject unknown environments. Matching versions alone do not establish that a device is available or
that P2P works on a particular PCIe topology. See [support and ownership contracts](docs/SUPPORT.md).
</details>

## 🚀 Quick start

No NPU is required for the contract tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .              # Python package + `hetero-p2p` command
python -m unittest discover -s tests -v
```

The tests cover transfer bounds, buffer aliasing, ownership transitions, incomplete completions, scheduling and
serving contracts. They use test doubles and host-backed fixtures; passing them does not demonstrate hardware DMA.
Installation does not include vendor drivers, SDKs, model weights or the native DMA helper, and no package-index
installation is assumed.

<details>
<summary><b>Mock transfer session (no data is copied)</b></summary>
<br>

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

The synthetic address, host file descriptor and placeholder owners above are valid only for this mock example.
Hardware transfers require addresses and exported descriptors obtained from live SDK allocations.
</details>

<details>
<summary><b>Hardware integration: native helper, doctor and model integration steps</b></summary>
<br>

Build the helper against the locally installed driver headers and a built liburing tree:

```bash
python native/build.py \
  --driver-root /path/to/furiosa-driver-rngd-2026.4.0 \
  --liburing-root /path/to/liburing-2.6
```

The build produces `build/rngd-pdma` and its build manifest. Keep them together: the backend checks the helper
against the manifest before use.

Check the installed environment, using the intended physical device path:

```bash
hetero-p2p doctor --device /dev/rngd/npu3ch0
```

`doctor` checks versions, ABI and device-file compatibility. It does not reserve the device or execute a P2P transfer.

To integrate a model:

1. Create and synchronize the source KV allocation through its SDK.
2. Export a supported destination allocation and retain its owning SDK objects.
3. Describe the actual physical format with a `TransferPlan` or `TokenMajorLayout`.
4. Create a `CopySession` using `RngdPdmaBackend(helper, device=...)` on the owning process's main thread.
5. Admit the request, run decode after verified transfer completion, and release the lease only after verified decode completion.

The backend's device path must match the source allocation. A source address is a raw RNGD DRAM offset, not a CPU
pointer. Do not manually close exported descriptors or release SDK allocations while a session owns them. Failed or
uncertain transfers poison the session rather than recycling the destination.

See [library API and serving contracts](docs/SERVING_LIBRARY.md), [support details](docs/SUPPORT.md) and
[hardware validation](docs/VALIDATION.md).
</details>

<details>
<summary><b>llm-d and model examples</b></summary>
<br>

The llm-d integration implements a local prefill/decode stage contract. HTTP carries requests and control
information; the owner processes manage KV transfer and cache leases. This is not a NIXL/Mooncake connector or a
cross-node RDMA transport; the transfer itself is also available to NIXL clients through the `HETERO_NPU` backend
plugin ([NIXL backend](docs/nixl-backend.md)).

The `examples/koni/` directory contains model-specific research examples for KONI-Llama3.1-8B. They require
additional model code, compiled artifacts and separate vendor environments. They are not a turnkey service installed
by `pip`.

- [llm-d integration](docs/LLMD.md)
- [HTTP batching setup](docs/HTTP_BATCHING.md)
- [Example configuration](examples/koni/serve.config.example.json)
- [Iteration scheduling](docs/ITERATION_SERVING_KO.md)
- [Independent decode replicas](docs/ITERATION_REPLICAS_KO.md)

Compiled B4/B8 examples and opt-in B12/B14 paths have been exercised. Batch size and context length depend on the
compiled artifact and available memory; they are not interchangeable runtime switches.
</details>

## 🔌 ai-rm integration

| | |
|---|---|
| **Layer / kind** | `l2-memory` library, used inside a serving pod ([`ai-rm.yaml`](ai-rm.yaml)) |
| **Provides** | `KVTransport` (`TransferPlan` / `CopySession` / `RngdPdmaBackend`, llm-d stage broker) |
| **Uses** | No other ai-rm component. The KONI examples load the `rngd_pd` package from `rngd-pd` through `PYTHONPATH` for model execution |
| **Adapter** | `hetero_p2p.airm:P2PTransport`, entry point `airm.kv_transports` name `p2p` |
| **NIXL backend** | `HETERO_NPU` plugin (`nixl_plugin/`, `libplugin_HETERO_NPU.so`): RNGD → ATOM DMA-BUF through `nixl_agent` ([design](docs/nixl-backend.md)) |
| **Metrics** | `kv_transfer_ms`, `kv_transfer_gbps` (tag `kv_transport=p2p`) |
| **KPIs** | Heterogeneous resource count (RNGD + ATOM together); power efficiency (board energy integrated over HTTP intervals by `hetero_p2p.energy`) |
| **Site config** | None. No endpoints or IPs in code; model, SDK and artifact paths come from config files copied from `examples/koni/*.example.json` |

```python
from rm_core import registry
transport = registry.load("airm.kv_transports", "p2p")
```

```bash
python -m rm_core.conformance airm.kv_transports p2p
```

Without a built helper (`HETERO_P2P_HELPER`) and a live-buffer fixture (`HETERO_P2P_CONFORMANCE_FIXTURE`),
conformance reports **SKIP** with the reason.

<details>
<summary><b>Protocol v0.1 adapter reference</b></summary>
<br>

The adapter only wraps the existing `TokenMajorLayout.prefix_plan` and `RngdPdmaBackend.transfer` (which returns
after all CQEs are checked); it does not substitute host staging.

- `supports`: True only for `furiosa` device memory → `rebellions` device memory (`device-hbm` / `device-dram`) on the same node.
- `plan`: accepts only an identical `dlfloat16` token-major layout and a prefix range `[0, end)`. An rm-core plan has
  no real allocation, so the owning process first passes a live RNGD source and ATOM export with
  `bind(request_id, P2PBinding(source, destinations, mappings, context, slot, batch))`. Range, alignment and format
  errors raise `TransportError`.
- `submit` / `wait`: `submit` runs the helper synchronously and stores the result after the completion fence; `wait`
  returns that result once. Call `submit` on the owning process's main thread. Failure is `ok=False` (no retry, no
  fallback). A binding is consumed by one `plan`. Each success emits `kv_transfer_ms` and `kv_transfer_gbps`
  (tag `kv_transport=p2p`) to `transport.measurements` and `metric_sink`.
- Constructor keywords: `helper`, `device`, `backend`, `metric_sink`. Environment variables: `HETERO_P2P_HELPER`
  (path to the built `rngd-pdma`), `HETERO_P2P_DEVICE` (default `/dev/rngd/npu1ch0`), `HETERO_P2P_CONFORMANCE_FIXTURE`
  (`module:callable`, below). The `HETERO_P2P_*` values are local paths, not site values.
- llm-d stage: `StageBroker(..., kv_transport='p2p')` takes the path name (shown in `/health`), and the owning process
  loads the implementation by name with `hetero_p2p.llmd.load_kv_transport(name, **kwargs)`.

`conformance_case()` tests on real tiers only when the environment check (`probe`) and helper match and the callable
named by `HETERO_P2P_CONFORMANCE_FIXTURE` returns `(src_tier, dst_tier, layout, P2PBinding)`; otherwise it SKIPs with
a reason. The fixture creates live allocations with the model SDKs and owns them for the duration of the test (see
the setup in `examples/koni/serve.py`).

Power efficiency covers board power only, so neither wall-plug efficiency nor MFU is derived.
</details>

<details>
<summary><b>Kubernetes + llm-d: what is still missing</b></summary>
<br>

There is no Kubernetes deployment yet. `src/hetero_p2p/llmd.py` implements the prefill/decode stage contract of the
official llm-d `pd-sidecar` (shared-storage) and was validated with local processes in `examples/koni/`
([llm-d integration](docs/LLMD.md)). A Kubernetes deployment needs:

- an authenticated RPC to a single cache owner and a Unix FD (DMA-BUF) passing path when prefill and decode run in different pods
- a container image and manifests (RNGD and ATOM devices, the native helper `build/rngd-pdma`)
- EPP/Envoy routing, multi-node and failure-recovery validation
</details>

## 📊 Results

KONI-Llama3.1-8B, context 512, inputs of 140/167/227/271 tokens, fixed output length, greedy non-streaming, through
the llm-d HTTP path. Medians of 3 runs; timings include the cost of full-logit verification.

| Configuration (same 3 cards: 1 RNGD prefill + ATOM decode) | B4 PD | B8 PD | B8/B4 |
|---|---:|---:|---:|
| 16 requests, output 32 | 85.66 tok/s | 134.83 tok/s | 1.57× |
| 16 requests, output 128 | 100.52 tok/s | 176.50 tok/s | 1.76× |
| 32 requests, mixed output 32/64/96/128 | 93.10 tok/s | 127.47 tok/s | 1.37× |

- All 424 compared requests matched the same-batch host-admitted reference with **exact full logits**; every request
  completed 64/64 DMA CQEs, with a median DMA time of about 3.0 ms per request ([B8 results](docs/PD_B8_RESULTS_KO.md)).
- With 1 RNGD + 1 ATOM at B12, 640 requests were exact and the median DMA time was 2.985 ms
  ([B12 results](docs/PD_B12_RESULTS_KO.md)).
- Against single-card baselines (output 128, board energy only): PD 98.285 tok/s and 0.6070 tok/J on 3 cards, RNGD
  solo 66.284 tok/s and 0.4326 tok/J, ATOM vendor solo 52.255 tok/s and 0.8713 tok/J
  ([comparison](docs/benchmark-b4/RESULTS_KO.md)). PD had higher throughput but lower energy efficiency than ATOM solo.

Performance depends on model, batch occupancy, input/output lengths, request arrivals and hardware allocation.
Transport output agreement is not proof of model-quality equivalence; PD throughput is not necessarily better than
standalone inference, and a throughput gain need not improve latency or energy efficiency. The repository preserves
workload-specific results, including regressions and failed experiments; raw local runs, model weights and compiled
artifacts may be excluded from Git, and checked-in summaries identify their provenance and limitations.

### RNGD → ATOM transfer-time curve (existing path vs NIXL backend)

RNGD PE0 → ATOM on one host, flat byte buffers, median of 5 transfers per cell, timed from post to the completion
fence (every CQE reaped). GB/s = 10⁹ bytes/s. Every cell was checked byte-exact on ATOM: the destination is poisoned
first, then must hold the source pattern, with bytes past the range untouched. 2026-10-19, NIXL 1.5.0 (pip wheel);
raw data in [`docs/nixl/`](docs/nixl/curve-20261019.json), method in [NIXL backend](docs/nixl-backend.md).

| Size (GiB) | Existing path (helper, 64 MiB per call) | NIXL qd 1 | NIXL qd 4 | NIXL qd 8 | NIXL 32 layers, qd 4: first layer / all | RNGD → host leg (staged) |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 (104 MiB) | 16.0 ms · 6.8 GB/s | 3.2 ms · 33.7 | 2.6 ms · 41.4 | 2.7 ms · 41.1 | 0.32 / 2.7 ms | 27.1 ms · 4.0 |
| 0.25 | 13.2 ms · 20.3 | 8.2 ms · 32.5 | 6.5 ms · 41.6 | 6.4 ms · 41.7 | 0.81 / 6.5 ms | 33.0 ms · 8.1 |
| 0.5 | 31.9 ms · 16.8 | 16.2 ms · 33.1 | 12.9 ms · 41.7 | 12.9 ms · 41.7 | 1.57 / 12.9 ms | 54.0 ms · 9.9 |
| 1 | 64.0 ms · 16.8 | 33.7 ms · 31.9 | 25.7 ms · 41.9 | 25.6 ms · 41.9 | 1.64 / 25.7 ms | 95.5 ms · 11.2 |
| 2 | 133.3 ms · 16.1 | 70.3 ms · 30.6 | 51.8 ms · 41.5 | 51.7 ms · 41.5 | 1.71 / 51.3 ms | 157.2 ms · 13.7 |
| 4 | 276.3 ms · 15.5 | 140.2 ms · 30.6 | 103.9 ms · 41.3 | 102.6 ms · 41.9 | 3.32 / 103.2 ms | 334.3 ms · 12.8 |

- The NIXL backend issues the same PDMA DMA-BUF command as the helper. At queue depth 1 it is about 2× the helper
  (one request instead of one helper process per 64 MiB); at queue depth 4 it holds about 41.5 GB/s from 0.1 to 4 GiB,
  and depth 8 adds nothing.
- Layer-wise ([aicomp#47](https://github.com/ai-computing/aicomp/issues/47)): 32 per-layer requests take as long as
  one request in total, and layers reach their fences in order. The first fence comes after roughly
  max(layer size, queue depth × 16 MiB) / bandwidth — 3.3 ms of 103 ms at 4 GiB (qd 1: 4.0 ms; qd 8: 3.4 ms). If
  prefill produces layers no faster than they transfer, about one layer of transfer stays exposed after prefill instead
  of the whole buffer. TTFT is not measured here.
- Staged comparison: the RNGD → host leg alone (PDMA into pinned host pages) runs at 11–14 GB/s from 1 GiB up, before
  any host → ATOM copy (the ATOM SDK copy of 4 GiB from host took about 0.8 s here, host buffer creation included).
- The helper and staged columns vary between runs at ≤ 0.25 GiB (an earlier run: helper 13.7 / 17.7 GB/s, staged
  18.5 / 11.7 GB/s at 0.1 / 0.25 GiB, [`run1`](docs/nixl/curve-20261019-run1.json)); the NIXL columns agree within 1%.
- 4 GiB uses two 2 GiB allocations per side (neither SDK allocates a single tensor of 4 GiB). This checks bytes, not
  tokens; a KONI token-exact run through NIXL needs the engine hook ([aicomp#48](https://github.com/ai-computing/aicomp/issues/48)).

| Evaluation | Evidence |
|---|---|
| Same-batch host-admitted vs P2P output validation | [Validation record](docs/VALIDATION.md) |
| llm-d HTTP batching (B4) | [HTTP results](docs/HTTP_RESULTS_KO.md) |
| Batch-width scaling | [B8 results](docs/PD_B8_RESULTS_KO.md), [B12 results](docs/PD_B12_RESULTS_KO.md) |
| B14/context448, saturation and mixed workloads | [Results and limitations](docs/B14_CONTEXT448_KO.md) |
| Scheduling policies and research basis | [Research log](docs/OPTIMIZATION_RESEARCH_KO.md) |
| Memory constraints and unsuccessful configurations | [B16 feasibility](docs/B16_FEASIBILITY_KO.md) |
| RNGD → ATOM through NIXL, 0.1–4 GiB curve | [NIXL backend](docs/nixl-backend.md), [raw data](docs/nixl/curve-20261019.json) |

## ⚠️ Limitations

- Hardware support is limited to the validated adapter/ABI combination and transfer direction.
- Other models require compatible allocation, layout, conversion and synchronization adapters.
- Multi-node transport, full Kubernetes/EPP/Envoy deployment, general sampling and production token streaming are outside the validated examples.
- Forced process termination or device reset during DMA is not a supported recovery mechanism.
- Board-power integration excludes host CPU/DRAM and does not establish wall-plug efficiency or MFU.

## 🛠 Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

For changes to hardware adapters, also verify actual allocation/export behavior, physical layout, completion
handling, output correctness and clean shutdown on the target hardware. Preserve failed runs and record SDK versions,
source revisions and measurement conditions.

| Directory | Contents |
|---|---|
| `src/hetero_p2p/` | Python transport, layout, scheduling and serving contracts |
| `native/` | RNGD PDMA helper and build script |
| `nixl_plugin/` | `HETERO_NPU` NIXL backend plugin (C++, meson) |
| `examples/nixl/` | NIXL transfer benchmark and ATOM export owner |
| `examples/koni/` | Model-specific integration and benchmark tools |
| `tests/` | CPU contract tests and native validation utilities |
| `docs/` | API guidance, experiment reports and support boundaries |

## 📚 Documentation

| Goal | Guides |
|---|---|
| 🔧 **Use the library** | [Serving library API](docs/SERVING_LIBRARY.md) · [Support and ownership](docs/SUPPORT.md) · [Batch pipeline](docs/BATCH_PIPELINE.md) |
| 🔌 **Connect to llm-d** | [llm-d integration](docs/LLMD.md) · [HTTP batching](docs/HTTP_BATCHING.md) · [Length scheduling](docs/LENGTH_SCHEDULING.md) |
| 🔁 **Use from NIXL** | [HETERO_NPU NIXL backend](docs/nixl-backend.md) |
| 📊 **Check the evidence** | [Validation](docs/VALIDATION.md) · [Benchmarks](docs/benchmark-b4/RESULTS_KO.md) · [Research log](docs/OPTIMIZATION_RESEARCH_KO.md) |
| 🧭 **ai-rm context** | [ai-rm](https://github.com/ai-computing/ai-rm) · [Optimus Prime consolidation plan](https://github.com/ai-computing/ai-rm/blob/master/docs/engine-integration.md) · [rm-core protocol](https://github.com/ai-computing/rm-core/blob/main/PROTOCOL.md) |

Most experiment reports (`*_KO.md`) are written in Korean.

## 📄 License

A project license has not yet been declared in a `LICENSE` file. Public repository access does not by itself grant
an open-source license. Vendor SDKs, driver components and model artifacts remain subject to their respective terms.

<div align="center"><sub>Part of <a href="https://github.com/ai-computing/ai-rm">ai-rm</a> · ai-computing team · ETRI</sub></div>
