# KONI batch 4 pipeline

`examples/koni/batch_pipeline.py` connects the library's live ATOM exporter, physical layout planner and synchronous PDMA backend to KONI prefill and decode. It uses two ATOM replicas with four independent KV slots each. Admission on one replica overlaps decode on the other; a replica's KV is never overwritten during its decode.

This is a reproducible batch benchmark. The existing llm-d HTTP example remains B1; this file does not claim HTTP microbatch scheduling is implemented.

## Run

Copy `examples/koni/batch.config.example.json` and set absolute paths to the model snapshot, B4 external decoder, tables, workload, helper and ATOM Python. The parent runs in the Furiosa2026.4 environment. Both environments need this package and the existing `rngd_pd` model adapter on their Python path.

```bash
PYTHONPATH=/path/to/hetero-npu-p2p/src:/path/to/rngd_llmd/src \
  /path/to/furiosa-2026.4/bin/python examples/koni/batch_pipeline.py \
  --config /path/to/batch-config.json
```

The workload JSON has a `requests` list of exactly four objects with `prompt_token_ids` integer lists. Each prompt must contain 2–384 tokens, each token must be in KONI's vocabulary. The model is the validated 32-layer KONI Llama3.1 8B snapshot. Output lengths are32/128; every wave runs16requests. Context capacity is512. Warmups/repeats and output lengths are configurable. The output directory must not already exist.

Only RNGD1→ATOM2/3 on the pinned driver/SDK versions is admitted by this example. Device occupancy and RSD groups are checked before model loading. No reset or reverse PRIME import is used.

## Bounded prefix conversion

`hetero_p2p.adapters.rngd_dlfloat.BoundedPrefixPacker` compiles one16-token encoder. It materializes each strided BF16 slice as a contiguous NPU tensor before execution, encodes each chunk, and concatenates the FP16 bit carriers on the NPU. Only scalar range checks are read on the host during encoding. It never submits DMA itself. Torch/Furiosa imports are optional adapter dependencies; the core remains stdlib-only.

The contiguous conversion is required: the dense compiled input did not correctly interpret a narrow view with the original512-token strides. CPU boundary tests and RNGD bitwise comparisons verify the fix. Large monolithic prefix initialization had previously crashed before any DMA; failed evidence is retained in the research workspace.

## Verification and limits

Before timing, the example stores fresh host KV references for each prompt and generates a128-token reference on both ATOMs. Both reference hashes must match. Every measured P2P decode then checks every logit against that same-batch reference. This host reference phase is excluded from timing; timed KV admission uses the device packer and PDMA.

Output includes wave/stage timings, per-board power samples, export metadata and exact-parity results. Outputs can be large because reference KV/logits are stored. Do not commit runtime artifacts or include them in packages by default.

This verification establishes transport correctness, not general model quality equivalence. PD uses three boards; standalone baselines use one. Performance and energy advantages must be reported separately. Four repeated prompts are not a production request distribution.

## Hardware result (2026-10-15)

The library path completed128requests, including96measured requests, with exact full-logit host/P2P parity and no same-buffer overlap. Median output throughput was87.83 tokens/s for output32 and96.88 tokens/s for output128. Board efficiency was0.537 and0.598 tokens/J. See `batch-hardware-validation.json`. The standalone results in `benchmark-b4` were separate runs on the same workload. The ATOM standalone baseline remains more energy-efficient.
