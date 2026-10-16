# Replica serving contracts

The transport library now separates three responsibilities in `hetero_p2p.serving`:

- `BatchOptions`: validates batch width, bounded lookahead, and collection time. The KONI example supports compiled B4/B8 artifacts, checks the artifact's runtime batch dimension before export, and passes the chosen width to layout planning and scheduling.
- `CompletionLedger`: validates generation, request identity, output length, success, duplicate responses, and consistency between early and final outputs. Early completion returns response rows but retains the replica record. Only validated final completion permits the caller to finish scheduler and CopySession leases. It does not perform DMA or release memory itself. Calls must be serialized by the owning thread.
- `RngdBinding`: records physical DMA device and optional SDK logical index. These indices need not match. Before exporting ATOM buffers the example acquires a small compute tensor, verifies the actual channel device files opened by that process, and retains the probe through the run. The source allocation and backend are both bound to the verified physical path.

```python
from hetero_p2p.serving import BatchOptions, CompletionLedger, RngdBinding
options = BatchOptions(width=8, lookahead=16, collection_seconds=.02)
# Example from this host, not a portable assumption about ordinal numbering:
binding = RngdBinding.from_path('/dev/rngd/npu3ch0', logical_index=2)
# After acquiring a tensor on binding.torch_device, check owned channel FDs:
binding.verify_open_channels(['/dev/rngd/npu3ch0'])
ledger = CompletionLedger()
ledger.start(replica=2, batch_id='2-1', outputs={'request-a': 32})
# fresh_rows, is_final = ledger.accept(replica=2, message=worker_message)
# Return fresh rows; keep memory leased until is_final and all checks succeed.
```

The example `serve_length.py` accepts `batch_size`, `lookahead`, `collection_ms`, `rngd_device_path`, and `rngd_logical_index`. B4 remains the default. This is configuration-time selection, not hot switching two full model artifacts on a card. `serve_batch_worker.py` now delegates to the shared B4/B8 worker rather than duplicating the decoder/export/parity implementation. Old benchmark revisions and raw results remain reproducible via Git and saved manifests.

## Lessons from startup failures

The current physical RNGD nodes are0,1,3, while fusion8 SDK indices are0,1,2. An attempted physical node2 was rejected before any worker/DMA. A subsequent attempt on physical0 hit EBUSY from a container workload omitted by the SMI process list; its misleading wrapper error was OutOfMemory. Neither attempt submitted DMA. A small allocation on logical2 proved that its channel FDs belong to physical3. Both failures are retained.

SMI is useful for diagnostics but is not sufficient proof that a card is idle. The runtime acquisition and live-channel mapping check now happen before ATOM worker startup. No driver reset, reverse import, raw-address guessing, or other users' process termination is involved.

## Scope

The core contracts are independent of the model. The example and physical KV conversion remain KONI- and SDK-specific; compiler/runtime versions stay pinned. This change does not provide a generic llm-d NIXL/Mooncake connector, continuous insertion into a running replica, cancellation recovery, automatic device enumeration/mapping, or an arbitrary-hardware compatibility guarantee. The broker's TTL cleanup remains request-triggered. Power and MFU require separate measurement.

## Burst handling

The HTTP listen backlog is now at least128 (or twice broker capacity). A request that arrives before an unstarted partial batch closes can fill that replica; previously only completely empty replicas were considered, which could leave a seven-request B8 batch followed by a one-request batch. The length queue accepts an explicit free-slot limit, avoids waiting for an entire new batch when filling such a gap, and measures its collection quiet interval from the newest queued request. The final experiment uses100ms partial dispatch wait for both B4 and B8, versus10ms previously. Full batches dispatch immediately; sparse requests may pay the additional wait.

## Running the validated example

Copy `examples/koni/serve.config.example.json`, replace artifact/runtime/model paths, and choose a new output directory. The physical3/logical2 mapping in that file is specific to the measured host. The executable acquisition check must succeed; do not bypass it on another topology.

```sh
PYTHONPATH=src:/path/to/rngd_llmd/src /path/to/furiosa/python examples/koni/serve_length.py --config config.json
/path/to/pd-sidecar --port 18823 --vllm-port 18822 --kv-connector shared-storage --secure-proxy=false
python3 examples/koni/validate_pd_scaling.py --config config.json
python3 examples/koni/analyze_pd_scaling.py /path/to/b4-run /path/to/b8-run --output summary.json
```

The core wheel contains no model weights, compiled NPU artifacts, or SDK binaries. The example still requires the separately installed, pinned vendor runtimes and legacy KONI model adapter. Every timed request includes full-logit verification overhead; these are validation-mode throughput figures, not a claim about a stripped production server.
