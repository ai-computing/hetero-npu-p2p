# Experimental KONI HTTP batching

`examples/koni/serve_batch.py` connects the llm-d shared-storage control protocol to the library's fenced DMA admission and two B4 ATOM decoder replicas. This remains a pinned KONI/SDK example, not a general NIXL or Mooncake connector.

## What changed

The stage broker no longer holds its global state lock while waiting for NPU callbacks. A pending decode therefore permits other requests to reach prefill/decode and join its batch. Duplicate IDs, conflicting prompts, and unsuccessful admissions remain rejected or quarantined.

A single owner thread performs all P2P transfers and schedules decoder work. It fills an idle replica, waits for every allocated slot's decode command, and dispatches up to four requests. The second replica can decode while the owner admits a subsequent batch. A partial batch waits for `batch_wait_ms` once its requests are queued. This bound does not include prefill, queueing behind busy replicas, or waiting for a missing decode command after successful prefill.

The compiled B4 kernel touches every lane. Consequently, no slot on a running replica may be admitted, discarded, or reused until the entire batch finishes. Requests with shorter outputs currently receive their HTTP response when the longest request in their batch finishes. This conservative implementation trades ragged-request latency for clear buffer ownership.

## Running

Copy `batch.config.example.json`, use absolute model/artifact/runtime paths, choose a new short output directory, and add `prefill_port`, `decode_port`, and optionally `batch_wait_ms` (default 10). Only ATOM 2/3 and RNGD 1 are validated in this example. Check device ownership before launch.

```sh
PYTHONPATH=src:/path/to/rngd_llmd/src /path/to/furiosa/python examples/koni/serve_batch.py --config config.json
/path/to/pd-sidecar --port 18823 --vllm-port 18822 --kv-connector shared-storage --secure-proxy=false
python3 examples/koni/validate_batch_http.py --config config.json --sidecar-port 18823 --output results.json
```

Send greedy, non-streaming token-ID `/v1/completions` requests to the sidecar with `x-request-id` and `x-prefiller-host-port`. The sidecar performs prefill before decode; KV bytes travel through PDMA, not HTTP. The server supports outputs up to 128 tokens and inputs within the configured prefill bucket. Creating `stop` in the output directory requests shutdown after active decoder batches complete.

Each of the four configured reference prompts is checked against full same-B4 host-admitted logits, including placement in different slots. Other valid prompts receive a finite-logit check and are explicitly marked `reference_checked: false`; this is not an accuracy guarantee. Failures, copy receipts, request IDs, useful batch widths, source hashes, and raw HTTP results are retained. HTTP throughput includes client-to-response time and must not be silently mixed with an engine-only benchmark boundary.
