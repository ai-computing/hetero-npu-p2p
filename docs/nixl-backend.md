# HETERO_NPU: a NIXL backend for RNGD → ATOM KV transfer

**English** · [한국어](nixl-backend.ko.md)

Status: experimental, validated on one host (RNGD 2026.4.0 PE0 → ATOM driver 3.0.0, NIXL 1.5.0 pip
wheel). ai-rm gap W3; engine-side hooks are tracked in
[aicomp#48](https://github.com/ai-computing/aicomp/issues/48) (NIXL transport hook) and
[aicomp#47](https://github.com/ai-computing/aicomp/issues/47) (layer-wise transfer). No engine code
lives here.

## 1. Why

NIXL is the KV-transfer layer that Dynamo, llm-d and vLLM's `NixlConnector` already speak: an agent
registers memory, builds transfer requests from descriptor lists and polls them to completion, and
backends are plugins (`libplugin_<NAME>.so`). Our RNGD → ATOM path was reachable only through the
hetero-npu-p2p Python API (`CopySession` + the `rngd-pdma` helper). `HETERO_NPU` puts the same PDMA
data path behind the standard NIXL plugin interface, so a NIXL client can move KV from RNGD device
memory into an ATOM DMA-BUF without knowing about either SDK.

## 2. What is direct and what is staged

| Route (source → destination) | NIXL memory types | PDMA command | Host memory in the data path |
|---|---|---|---|
| RNGD DRAM → ATOM DMA-BUF | VRAM → VRAM | `DMABUF_READ` | **none** (direct P2P, the hetero-npu-p2p direction) |
| RNGD DRAM → host DRAM | VRAM → DRAM | `SCATTER_READ` | destination is host memory (explicit staging leg) |
| host DRAM → RNGD DRAM | DRAM → VRAM | `SCATTER_WRITE` | source is host memory (explicit staging leg) |
| ATOM DMA-BUF → RNGD | VRAM → VRAM | — | refused (`NIXL_ERR_NOT_SUPPORTED`): not validated |
| RNGD → RNGD, host → host, host ↔ DMA-BUF | — | — | refused |

The backend never chooses a route on its own and never falls back: a request whose descriptors do
not form one supported route fails in `prepXfer`. A host-staged RNGD → ATOM transfer is two
explicit steps chosen by the caller (RNGD → host with this backend, host → ATOM with the ATOM SDK),
which is what rm-core PROTOCOL §4.5 requires.

## 3. Plugin API mapping

`nixl_plugin/` implements `nixlBackendEngine` (NIXL plugin API version of the tag it is built
against; 1.5.0 here) and exports `nixl_plugin_init` / `nixl_plugin_fini`.

| NIXL call | HETERO_NPU behaviour |
|---|---|
| `supportsLocal / Remote / Notif` | `true / false / true` — same-host, one agent; notifications are intra-agent |
| `getSupportedMems` | `DRAM_SEG`, `VRAM_SEG` |
| `registerMem` | parses `metaInfo` (below); opens the RNGD PDMA device once per path; `dup()`s a DMA-BUF fd |
| `deregisterMem` | closes the duplicated DMA-BUF fd; the RNGD device closes with its last region |
| `connect`, `loadLocalMD`, `unloadMD` | no-ops (local metadata is the registration itself) |
| `prepXfer` | validates route, bounds and alignment; splits into ≤ `desc_bytes` descriptors and groups ≤ `descs_per_cmd` descriptors (one DMA-BUF) per command |
| `postXfer` | queues every command on the device's io_uring; at most `queue_depth` commands in flight per RNGD device |
| `checkXfer` | reaps completions without blocking; `NIXL_IN_PROG` until the fence below |
| `releaseReqH` | drops commands not yet submitted, **waits** for submitted ones, then frees |
| `getNotifs` / `genNotif` | delivers the `notifMsg` of a request after its fence; `genNotif` to the local agent |

### Registration (`nixlBlobDesc.metaInfo`)

| Kind | mem type | `addr` | `metaInfo` |
|---|---|---|---|
| RNGD allocation | `VRAM` | raw DRAM offset (`tensor.data_ptr()` of a Furiosa tensor, 64-byte aligned) | `rngd:/dev/rngd/npuNch0` |
| Exported DMA-BUF (ATOM) | `VRAM` | caller-chosen base, e.g. the ATOM DVA; transfer address = base + byte offset | `dmabuf:<fd>[:<offset into the export>]` |
| Host memory | `DRAM` | virtual address (page aligned) | empty |

`devId` is free for the caller, but the two VRAM kinds must not share a `devId` when their address
ranges overlap (NIXL looks regions up by `devId` + range). `src/hetero_p2p/nixl.py` builds these
tuples (`rngd_region`, `dmabuf_region`, `host_region`) and equal-length descriptor pairs across
allocation boundaries (`paired_ranges`).

### Parameters (`create_backend("HETERO_NPU", params)`)

| Key | Default | Meaning |
|---|---|---|
| `queue_depth` | `1` | PDMA commands in flight per RNGD device (1 = the sequential native helper). 1..64 |
| `desc_bytes` | `2097152` | maximum bytes per scatter descriptor (helper's validated 2 MiB) |
| `descs_per_cmd` | `8` | descriptors per URING_CMD (helper's validated grouping) |
| `align` | `4096` | required alignment of addresses and lengths |
| `allow_host` | `1` | `0` rejects `DRAM_SEG` registrations (direct-only deployments) |

## 4. Completion semantics and fences

hetero-npu-p2p admits decode only after every DMA completion entry was checked
(`CopySession.admit`). The plugin keeps that contract at NIXL's level:

- `checkXfer` returns `NIXL_SUCCESS` only after **every** CQE of the request was reaped with result
  0. The PDMA driver completes a DMA-BUF command from its completion work item, after the engine
  finished, so the bytes are in the destination when the status flips.
- A failed command stops the request's remaining queued commands; `checkXfer` returns
  `NIXL_ERR_BACKEND` only once nothing of the request is still in flight. The destination contents
  are then undefined: the caller must treat the lease as poisoned (`CopySession` state
  `poisoned`), not retry another path.
- A notification attached to the request (`notifMsg`) is queued only on a clean fence, never for a
  failed transfer. Because NIXL's default notification list contains only remote-capable backends,
  read it with `agent.get_new_notifs(["HETERO_NPU"])`.
- `releaseReqH` on an active request cannot abort DMA: it cancels commands that were not yet
  submitted and waits for the rest. The descriptor memory and the destination are never released
  under a running DMA.
- After a fence, the destination SDK may still hold a stale host-side view: the ATOM owner must
  call `rebel._C.vmem.mark_device_memory_is_updated(ptr)` before reading or decoding, as
  `adapters/atom_koni_external.ExternalCacheDecoder.notify_cache_write` already does.

Mapping to the hetero-npu-p2p and rm-core contracts:

| hetero-npu-p2p / rm-core | NIXL with HETERO_NPU |
|---|---|
| `TransferPlan` (source, destinations, spans) | registered regions + descriptor lists (`initialize_xfer`) |
| `RngdPdmaBackend.transfer` returns after all CQEs | `transfer` + `check_xfer_state == "DONE"` |
| `CopySession` `ready` / `poisoned` | `DONE` / `ERR` |
| `notify()` after the fence | `notifMsg` delivered after the fence |
| rm-core §4.3 `submit` may return early, §4.4 `wait` is the fence | `transfer` returns `PROC`; `DONE` is the fence |
| rm-core §4.5 no silent fallback | unsupported routes fail in `prepXfer`; staging is a separate, explicit request |

## 5. Engine integration (aicomp#48, #47)

The engine owns allocations; the transport only needs them registered. Proposed use behind the
`KVSpec` / fence contract of Optimus Prime's `PDRuntime`:

1. At startup the prefill (RNGD) side registers its KV source region(s) and the decode side's
   exported ATOM cache DMA-BUF(s) with one agent, `backends=["HETERO_NPU"]`.
2. Per request, `initialize_xfer("WRITE", src_descs, dst_descs, agent.name)` from the physical
   layout (`TokenMajorLayout` spans become descriptors), `transfer`, poll `check_xfer_state`.
3. Decode admission waits for `DONE` (or the notification), then `notify_cache_write` on ATOM.

Layer-wise (#47): one request per layer (or layer group), posted as each layer's KV is final on
RNGD. The requests share the device queue in FIFO order, so layer `i` reaches its fence while later
layers are still moving. The benchmark posts 32 layer requests back to back and records each fence:
the total equals the single-request time, and the first fence arrives after roughly
max(layer bytes, queue_depth × command bytes) / bandwidth, because the first layer shares the PDMA
channels with whatever else is in flight (4 GiB: 3.3 ms of 103 ms at depth 4; 1 GiB: 1.6 ms at
depth 4 but 3.1 ms at depth 8). For layer-wise use, keep `queue_depth × desc_bytes × descs_per_cmd`
at or below one layer's bytes. That is the window a layer-wise prefill can overlap; whether TTFT
improves depends on the engine's prefill schedule (#47).

Cross-pod use needs the DMA-BUF fd passed into the transport process (SCM_RIGHTS, as
`examples/koni` does); the plugin does not move fds itself.

## 6. Build and run

The plugin is built out of tree against the headers of the same NIXL tag as the runtime library
(plugins must match NIXL's C++20 / libstdc++ ABI). With the pip wheel (`pip install nixl==1.5.0`
in the Furiosa environment, CUDA not required):

```bash
git clone --branch v1.5.0 --depth 1 https://github.com/ai-dynamo/nixl.git nixl-src
# Abseil headers matching NIXL 1.5.0 (absl lts_20250814)
curl -sSLO https://github.com/abseil/abseil-cpp/releases/download/20250814.1/abseil-cpp-20250814.1.tar.gz
tar xzf abseil-cpp-20250814.1.tar.gz
SITE=$(python -c 'import nixl_cu12,os;print(os.path.dirname(os.path.dirname(nixl_cu12.__file__)))')
meson setup nixl_plugin/build nixl_plugin \
  -Dnixl_src=$PWD/nixl-src -Dnixl_libdir=$SITE/.nixl_cu12.mesonpy.libs \
  -Dabsl_include=$PWD/abseil-cpp-20250814.1 \
  -Ddriver_root=/usr/src/furiosa-driver-rngd-2026.4.0 -Dliburing_root=/path/to/liburing-2.6
ninja -C nixl_plugin/build            # -> nixl_plugin/build/libplugin_HETERO_NPU.so
export NIXL_PLUGIN_DIR=$PWD/nixl_plugin/build
```

`NIXL_PLUGIN_DIR` replaces NIXL's default plugin directory; symlink the wheel's plugins
(`.nixl_cu12.mesonpy.libs/plugins/*`) into it if the same process also needs UCX or POSIX. With a
NIXL source build, copy `nixl_plugin/` to `src/plugins/hetero_npu` and add a `subdir()` instead.

Tests:

```bash
python -m unittest tests.test_nixl_descs                       # CPU, descriptor helpers
HETERO_NPU_NIXL=1 HETERO_NPU_ATOM_PYTHON=/path/to/atom-venv/bin/python \
  python -m unittest tests.test_nixl_hetero_npu -v             # hardware: register -> xfer -> check -> verify
python examples/nixl/bench_transfer.py --atom-python ... --helper build/rngd-pdma --out runs/curve
```

The ATOM destination is an eager `rbln` tensor exported with `rblnExportMemoryByDva` from the live
allocation captured by `examples/koni/capture.gdb` (pinned thunk ABI, `atom_experimental`). The
hardware test checks host ↔ RNGD round trips against the Furiosa SDK readback, RNGD → ATOM bytes
against the source pattern (and that bytes past the range keep their poison value), the
post-fence notification, and that unvalidated routes are refused.

## 7. Measured

See the README results section for the RNGD → ATOM curve (0.1–4 GiB) through the existing helper
path, the NIXL backend at queue depths 1/4/8, the 32-layer variant and the RNGD → host staging leg.
Raw data: [`docs/nixl/curve-20261019.json`](nixl/curve-20261019.json).

## 8. Limits and open items

- **Same host, one agent.** `supportsRemote=false`: no cross-node transfer, no metadata exchange.
  A two-agent same-host variant (prefill and decode processes each with an agent) needs fd passing
  in `loadRemoteMD`; not built.
- **ATOM export is not a public API.** The DMA-BUF comes from `rblnExportMemoryByDva` on an
  allocation found by tracing the ATOM runtime (gdb capture, thunk SHA-pinned). The plugin only
  needs a DMA-BUF fd; a supported ATOM export API would replace the capture step.
- **Allocation size.** Both SDKs reject a single tensor of 4 GiB or more (Furiosa: `E2BIG` from
  the HAL; rbln: allocation error). The 4 GiB point uses two 2 GiB regions per side; NIXL
  descriptors span them. Each PDMA command is kept ≤ 1 GiB (driver bound: < 4 GiB per DMA-BUF command).
- **Directions.** Only RNGD → DMA-BUF is P2P. The reverse (`DMABUF_WRITE`) and RNGD → RNGD are
  refused until validated; the RNGD-export → ATOM-import direction is known to crash the host and
  is never used.
- **Host staging** uses PDMA scatter I/O on pinned user pages. `/dev/udmabuf` (host DMA-BUF) is not
  accessible to the user on the test node, so host memory is never registered as a DMA-BUF.
- **Visible ATOMs.** On the test node the ATOM runtime exposed one device to the process
  (`RBLN_DEVICES` other than 0 → "0 physical NPUs"); the measurement used that idle card.
- **vLLM `NixlConnector`** registers torch device tensors itself and has no RNGD/ATOM memory type;
  a connector change (rngd-pd) is still needed to use this backend from vLLM.
