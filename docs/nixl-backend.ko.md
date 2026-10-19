# HETERO_NPU: RNGD → ATOM KV 전송용 NIXL 백엔드

[English](nixl-backend.md) · **한국어**

상태: 실험적. 한 호스트에서 검증했다(RNGD 2026.4.0 PE0 → ATOM 드라이버 3.0.0, NIXL 1.5.0 pip
wheel). ai-rm 갭 W3이며, 엔진 쪽 훅은 [aicomp#48](https://github.com/ai-computing/aicomp/issues/48)
(NIXL transport 훅)과 [aicomp#47](https://github.com/ai-computing/aicomp/issues/47)(layer-wise 전송)에서
다룬다. 이 저장소에는 엔진 코드가 없다.

## 1. 목적

NIXL은 Dynamo, llm-d, vLLM `NixlConnector`가 공통으로 쓰는 KV 전송 계층이다. agent가 메모리를
등록하고, descriptor 목록으로 전송 요청을 만들고, 완료까지 상태를 조회한다. 백엔드는 plugin
(`libplugin_<NAME>.so`)이다. 지금까지 RNGD → ATOM 경로는 hetero-npu-p2p Python API(`CopySession` +
`rngd-pdma` helper)로만 쓸 수 있었다. `HETERO_NPU`는 같은 PDMA 데이터 경로를 표준 NIXL plugin
인터페이스 뒤에 둔다. NIXL 클라이언트는 두 SDK를 몰라도 RNGD 장치 메모리의 KV를 ATOM DMA-BUF로
옮길 수 있다.

## 2. 직접 경로와 staging 경로

| 경로 (source → destination) | NIXL 메모리 종류 | PDMA 명령 | 데이터 경로의 호스트 메모리 |
|---|---|---|---|
| RNGD DRAM → ATOM DMA-BUF | VRAM → VRAM | `DMABUF_READ` | **없음** (직접 P2P, hetero-npu-p2p의 검증 방향) |
| RNGD DRAM → 호스트 DRAM | VRAM → DRAM | `SCATTER_READ` | destination이 호스트 메모리 (명시적 staging 구간) |
| 호스트 DRAM → RNGD DRAM | DRAM → VRAM | `SCATTER_WRITE` | source가 호스트 메모리 (명시적 staging 구간) |
| ATOM DMA-BUF → RNGD | VRAM → VRAM | — | 거부 (`NIXL_ERR_NOT_SUPPORTED`): 미검증 |
| RNGD → RNGD, 호스트 → 호스트, 호스트 ↔ DMA-BUF | — | — | 거부 |

백엔드는 경로를 스스로 고르지 않고 fallback도 하지 않는다. descriptor들이 지원 경로 하나를
이루지 않으면 `prepXfer`에서 실패한다. 호스트를 거치는 RNGD → ATOM 전송은 호출자가 고르는 두 개의
명시적 단계(이 백엔드로 RNGD → 호스트, ATOM SDK로 호스트 → ATOM)이며, rm-core PROTOCOL §4.5의
요구와 같다.

## 3. Plugin API 대응

`nixl_plugin/`은 빌드에 쓴 NIXL 태그(여기서는 1.5.0)의 `nixlBackendEngine`을 구현하고
`nixl_plugin_init` / `nixl_plugin_fini`를 export한다.

| NIXL 호출 | HETERO_NPU 동작 |
|---|---|
| `supportsLocal / Remote / Notif` | `true / false / true` — 같은 호스트, agent 하나. 알림은 agent 내부 |
| `getSupportedMems` | `DRAM_SEG`, `VRAM_SEG` |
| `registerMem` | `metaInfo`를 해석(아래). RNGD PDMA 장치는 경로당 한 번 연다. DMA-BUF fd는 `dup()` |
| `deregisterMem` | 복제한 DMA-BUF fd를 닫는다. RNGD 장치는 마지막 region과 함께 닫힌다 |
| `connect`, `loadLocalMD`, `unloadMD` | no-op (로컬 메타데이터는 등록 자체) |
| `prepXfer` | 경로·범위·정렬 검사. `desc_bytes` 이하 descriptor로 나누고 명령당 `descs_per_cmd`개(DMA-BUF 하나)로 묶는다 |
| `postXfer` | 모든 명령을 장치 io_uring에 넣는다. RNGD 장치당 동시 진행 명령은 최대 `queue_depth` |
| `checkXfer` | 막지 않고 완료를 회수한다. 아래 fence 전까지 `NIXL_IN_PROG` |
| `releaseReqH` | 아직 제출하지 않은 명령은 버리고, 제출한 명령은 **기다린 뒤** 해제 |
| `getNotifs` / `genNotif` | 요청의 `notifMsg`를 fence 이후에 전달. `genNotif`는 로컬 agent 대상 |

### 등록 (`nixlBlobDesc.metaInfo`)

| 종류 | mem type | `addr` | `metaInfo` |
|---|---|---|---|
| RNGD allocation | `VRAM` | raw DRAM offset (Furiosa tensor의 `data_ptr()`, 64바이트 정렬) | `rngd:/dev/rngd/npuNch0` |
| export된 DMA-BUF (ATOM) | `VRAM` | 호출자가 정한 base(예: ATOM DVA). 전송 주소 = base + byte offset | `dmabuf:<fd>[:<export 내 offset>]` |
| 호스트 메모리 | `DRAM` | 가상 주소 (페이지 정렬) | 빈 문자열 |

`devId`는 호출자가 정한다. 다만 주소 범위가 겹치는 두 VRAM 종류는 같은 `devId`를 쓰면 안 된다
(NIXL은 `devId` + 범위로 region을 찾는다). `src/hetero_p2p/nixl.py`가 이 tuple(`rngd_region`,
`dmabuf_region`, `host_region`)과 allocation 경계를 넘는 같은 길이의 descriptor 쌍(`paired_ranges`)을
만든다.

### 파라미터 (`create_backend("HETERO_NPU", params)`)

| 키 | 기본값 | 의미 |
|---|---|---|
| `queue_depth` | `1` | RNGD 장치당 동시 진행 PDMA 명령 수 (1 = 순차 native helper). 1..64 |
| `desc_bytes` | `2097152` | scatter descriptor당 최대 바이트 (helper가 검증한 2 MiB) |
| `descs_per_cmd` | `8` | URING_CMD당 descriptor 수 (helper가 검증한 묶음) |
| `align` | `4096` | 주소와 길이의 정렬 요구 |
| `allow_host` | `1` | `0`이면 `DRAM_SEG` 등록 거부 (직접 경로만 쓰는 배포) |

## 4. 완료 의미와 fence

hetero-npu-p2p는 모든 DMA 완료 항목을 확인한 뒤에만 decode를 허용한다(`CopySession.admit`).
plugin은 이 계약을 NIXL 수준에서 지킨다.

- `checkXfer`는 요청의 **모든** CQE가 결과 0으로 회수된 뒤에만 `NIXL_SUCCESS`를 돌려준다. PDMA
  드라이버는 DMA-BUF 명령을 엔진 완료 후 completion work item에서 끝내므로, 상태가 바뀔 때 바이트는
  이미 destination에 있다.
- 명령 하나가 실패하면 요청의 남은 대기 명령을 멈춘다. `checkXfer`는 요청의 어떤 명령도 진행 중이
  아닐 때에만 `NIXL_ERR_BACKEND`를 돌려준다. 이때 destination 내용은 정의되지 않으므로 호출자는
  lease를 poisoned로 다뤄야 하며(`CopySession`의 `poisoned`) 다른 경로로 재시도하지 않는다.
- 요청에 붙인 알림(`notifMsg`)은 정상 fence에서만 큐에 들어가고, 실패한 전송에는 생기지 않는다.
  NIXL 기본 알림 목록은 remote 지원 백엔드만 포함하므로 `agent.get_new_notifs(["HETERO_NPU"])`로 읽는다.
- 진행 중 요청의 `releaseReqH`는 DMA를 중단할 수 없다. 아직 제출되지 않은 명령은 취소하고 나머지는
  기다린다. descriptor 메모리와 destination은 DMA가 도는 동안 해제되지 않는다.
- fence 이후에도 destination SDK가 오래된 호스트 쪽 사본을 가질 수 있다. ATOM owner는 읽거나 decode하기
  전에 `rebel._C.vmem.mark_device_memory_is_updated(ptr)`를 호출해야 한다
  (`adapters/atom_koni_external.ExternalCacheDecoder.notify_cache_write`가 이미 하는 일).

hetero-npu-p2p 및 rm-core 계약과의 대응:

| hetero-npu-p2p / rm-core | HETERO_NPU를 쓰는 NIXL |
|---|---|
| `TransferPlan` (source, destinations, spans) | 등록 region + descriptor 목록 (`initialize_xfer`) |
| 모든 CQE 후 반환하는 `RngdPdmaBackend.transfer` | `transfer` + `check_xfer_state == "DONE"` |
| `CopySession` `ready` / `poisoned` | `DONE` / `ERR` |
| fence 후 `notify()` | fence 후 전달되는 `notifMsg` |
| rm-core §4.3 `submit`은 먼저 반환 가능, §4.4 `wait`가 fence | `transfer`는 `PROC` 반환, `DONE`이 fence |
| rm-core §4.5 묵시적 fallback 금지 | 미지원 경로는 `prepXfer`에서 실패. staging은 별도의 명시적 요청 |

## 5. 엔진 통합 (aicomp#48, #47)

allocation은 엔진이 소유하고, transport는 등록만 하면 된다. Optimus Prime `PDRuntime`의
`KVSpec` / fence 계약 뒤에서의 사용 제안:

1. 시작 시 prefill(RNGD) 쪽 KV source region과 decode 쪽에서 export한 ATOM cache DMA-BUF를 agent
   하나에 `backends=["HETERO_NPU"]`로 등록한다.
2. 요청마다 물리 layout(`TokenMajorLayout` span이 descriptor가 된다)으로
   `initialize_xfer("WRITE", src_descs, dst_descs, agent.name)`를 만들고 `transfer`, 이어서
   `check_xfer_state`를 조회한다.
3. decode admission은 `DONE`(또는 알림)을 기다린 뒤 ATOM에서 `notify_cache_write`를 호출한다.

Layer-wise(#47): layer(또는 layer 묶음)마다 요청 하나를 만들어, RNGD에서 그 layer의 KV가 확정될 때
올린다. 요청들은 장치 큐를 FIFO로 공유하므로 뒤 layer가 움직이는 동안 layer `i`가 fence에 도달한다.
벤치마크는 32개 layer 요청을 연달아 올리고 각 fence 시점을 기록한다. 전체 시간은 단일 요청과 같고,
첫 fence는 대략 max(layer 바이트, queue_depth × 명령 바이트) / 대역폭 뒤에 온다. 첫 layer가 함께 진행 중인
다른 명령과 PDMA 채널을 나눠 쓰기 때문이다(4 GiB: 깊이 4에서 103 ms 중 3.3 ms, 1 GiB: 깊이 4에서 1.6 ms,
깊이 8에서 3.1 ms). layer-wise로 쓸 때는 `queue_depth × desc_bytes × descs_per_cmd`를 layer 하나의 바이트
이하로 둔다. layer-wise prefill이 겹칠 수 있는 구간이 이것이며, TTFT 개선 여부는 엔진의 prefill 일정에
달렸다(#47).

pod를 넘는 사용에는 DMA-BUF fd를 transport 프로세스로 넘겨야 한다(`examples/koni`처럼 SCM_RIGHTS).
plugin은 fd를 직접 옮기지 않는다.

## 6. 빌드와 실행

plugin은 런타임 라이브러리와 같은 NIXL 태그의 헤더로 트리 밖에서 빌드한다(plugin은 NIXL의 C++20 /
libstdc++ ABI와 맞아야 한다). pip wheel(`pip install nixl==1.5.0`, Furiosa 환경, CUDA 불필요) 기준:

```bash
git clone --branch v1.5.0 --depth 1 https://github.com/ai-dynamo/nixl.git nixl-src
# NIXL 1.5.0과 맞는 Abseil 헤더 (absl lts_20250814)
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

`NIXL_PLUGIN_DIR`는 NIXL 기본 plugin 디렉터리를 대체한다. 같은 프로세스에서 UCX나 POSIX도 필요하면
wheel의 plugin(`.nixl_cu12.mesonpy.libs/plugins/*`)을 그 디렉터리에 symlink한다. NIXL을 소스로
빌드한다면 `nixl_plugin/`을 `src/plugins/hetero_npu`로 복사하고 `subdir()`을 추가한다.

시험:

```bash
python -m unittest tests.test_nixl_descs                       # CPU, descriptor helper
HETERO_NPU_NIXL=1 HETERO_NPU_ATOM_PYTHON=/path/to/atom-venv/bin/python \
  python -m unittest tests.test_nixl_hetero_npu -v             # 하드웨어: 등록 -> 전송 -> 확인 -> 바이트 검증
python examples/nixl/bench_transfer.py --atom-python ... --helper build/rngd-pdma --out runs/curve
```

ATOM destination은 eager `rbln` tensor이며, `examples/koni/capture.gdb`로 잡은 live allocation을
`rblnExportMemoryByDva`로 export한다(thunk ABI 고정, `atom_experimental`). 하드웨어 시험은 호스트 ↔ RNGD
왕복을 Furiosa SDK readback과, RNGD → ATOM 바이트를 source 패턴과 비교하고(범위 밖 바이트가 poison
값을 유지하는지도 확인), fence 후 알림과 미검증 경로 거부를 확인한다.

## 7. 측정

RNGD → ATOM 곡선(0.1–4 GiB): 기존 helper 경로, 큐 깊이 1/4/8의 NIXL 백엔드, 32-layer 변형, RNGD →
호스트 staging 구간은 README 결과 절에 있다. 원자료: [`docs/nixl/curve-20261019.json`](nixl/curve-20261019.json).

## 8. 한계와 남은 일

- **같은 호스트, agent 하나.** `supportsRemote=false`: 노드 간 전송과 메타데이터 교환이 없다.
  같은 호스트에서 agent 두 개(prefill·decode 프로세스 각각)를 쓰려면 `loadRemoteMD`에서 fd 전달이
  필요하며 아직 만들지 않았다.
- **ATOM export는 공개 API가 아니다.** DMA-BUF는 ATOM 런타임을 추적해 찾은 allocation(gdb capture,
  thunk SHA 고정)에 `rblnExportMemoryByDva`를 호출해 얻는다. plugin은 DMA-BUF fd만 필요하므로 공식
  ATOM export API가 생기면 capture 단계를 대체할 수 있다.
- **allocation 크기.** 두 SDK 모두 4 GiB 이상 단일 tensor를 거부한다(Furiosa: HAL의 `E2BIG`,
  rbln: 할당 오류). 4 GiB 측정은 양쪽 모두 2 GiB region 두 개를 쓰고 NIXL descriptor가 이를 넘나든다.
  PDMA 명령 하나는 1 GiB 이하로 유지한다(드라이버 한계: DMA-BUF 명령당 4 GiB 미만).
- **방향.** P2P는 RNGD → DMA-BUF뿐이다. 역방향(`DMABUF_WRITE`)과 RNGD → RNGD는 검증 전까지 거부한다.
  RNGD export → ATOM import 방향은 호스트 crash가 확인된 경로이며 쓰지 않는다.
- **호스트 staging**은 pin된 사용자 페이지에 대한 PDMA scatter I/O다. 시험 노드에서 `/dev/udmabuf`
  (호스트 DMA-BUF)는 사용자 접근 권한이 없어 호스트 메모리를 DMA-BUF로 등록하지 않는다.
- **보이는 ATOM.** 시험 노드에서 ATOM 런타임은 프로세스에 장치 하나만 보였다(`RBLN_DEVICES`가 0이 아니면
  "0 physical NPUs"). 측정은 그 유휴 카드로 했다.
- **vLLM `NixlConnector`**는 torch 장치 tensor를 스스로 등록하며 RNGD/ATOM 메모리 종류가 없다.
  vLLM에서 이 백엔드를 쓰려면 connector 변경(rngd-pd)이 더 필요하다.
