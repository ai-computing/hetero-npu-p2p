# Length-aware experimental KONI serving

`examples/koni/serve_length.py` is a separate experimental server so the validated B4 baseline remains reproducible.

The llm-d shared-storage protocol rewrites the prefill output limit to one. The server therefore performs prefill and creates an owned RNGD source allocation first, but defers ATOM slot assignment and P2P admission until the decode request supplies its real output limit. This is still an actual prefill operation; no KV bytes are returned through HTTP. Source allocations remain alive while queued, with the broker's admission capacity limiting the number of requests.

The owner collects up to eight queued decode requests. FIFO chooses the first four; length-aware selection includes the oldest request and the three closest output lengths in that window. This limits overtaking and avoids indefinitely favoring short requests. With no prefill work pending, a partial window waits at most20ms for collection (excluding compute and resource queueing). The requested output limit is useful for these fixed-length tests; it does not predict EOS termination in a general application.

The optional early-response mode returns a completed request after its own logits have passed validation. It **does not free its destination KV allocation or slot**. The entire replica remains leased until all B4 lanes finish. Stale completions and duplicate completion messages fail validation. Continuous insertion into a running batch is not supported.

A local `policy.json` in the run directory selects `{"policy":"fifo"|"length","early_response":true|false}` only when all work has drained. `policy-ack.json` acknowledges the applied setting. This control is for reproducible local ablation tests, not an HTTP administration API.

## Reproduce

Use a new output directory in the same config schema as `serve_batch.py`. These examples remain pinned to RNGD1 and ATOM2/3, KONI, context512 and the previously validated SDK artifacts.

```sh
PYTHONPATH=src:/path/to/rngd_llmd/src /path/to/furiosa/python examples/koni/serve_length.py --config config.json
/path/to/pd-sidecar --port 18823 --vllm-port 18822 --kv-connector shared-storage --secure-proxy=false
python3 examples/koni/validate_length_http.py --config config.json
python3 examples/koni/analyze_length_http.py /path/to/run
```

The comparison rotates FIFO, FIFO+early response, and length-aware+early response across three repeats. Each measured burst contains32 requests with output limits32/64/96/128 repeated eight times. Prompt IDs, model, devices, lookahead and HTTP timing boundary are shared. Fixed128 and three-request partial-batch checks are additional single-run diagnostics. No power-efficiency claim is inferred from this latency/throughput experiment.

The same-B4 host-admitted full logits must match for every request. The analyzer also checks every DMA receipt and rejects any admission overlapping an active decoder replica. Failed experiments are retained.

## Lifetime and deployment limits

An early HTTP response is not a memory-release event: request IDs should remain globally unique, and the destination is reclaimed only after the final batch completion. The experimental server retains the original broker’s request-triggered expiration rather than adding a background TTL worker. Its source queue increases device-memory residency compared with immediate ATOM admission; capacity defaults to64 and is not a hardware-independent memory budget. This example is not yet a production cancellation/retry implementation.
