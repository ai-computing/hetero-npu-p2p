<div align="center">

<h1>hetero-npu-p2p</h1>

<h3>같은 호스트의 이종 NPU 사이에서 완료를 확인하는 KV cache 전송</h3>

<p>
  <a href="README.md">English</a> · <strong>한국어</strong>
</p>

<p>
  <a href="https://github.com/ai-computing/ai-rm"><img src="https://img.shields.io/badge/ai--rm-L2%20memory%20%26%20transport-22d3ee" alt="ai-rm 계층: L2 메모리·전송"></a>
  <a href="https://github.com/ai-computing/rm-core/blob/main/PROTOCOL.md"><img src="https://img.shields.io/badge/protocol-v0.1-7c5cff" alt="Protocol v0.1"></a>
  <a href="ai-rm.yaml"><img src="https://img.shields.io/badge/maturity-experimental-f59e0b" alt="성숙도: experimental"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&amp;logoColor=white" alt="Python 3.10+"></a>
  <a href="#-라이선스"><img src="https://img.shields.io/badge/license-not%20declared-64748b" alt="라이선스: 미선언"></a>
</p>

<p>
  <a href="#-주요-기능"><b>주요 기능</b></a> ·
  <a href="#-빠른-시작"><b>빠른 시작</b></a> ·
  <a href="#-ai-rm-연동"><b>ai-rm 연동</b></a> ·
  <a href="#-결과"><b>결과</b></a> ·
  <a href="#-문서"><b>문서</b></a>
</p>

</div>

---

**hetero-npu-p2p**는 같은 Linux 호스트에서 Furiosa RNGD의 KV cache를 Rebellions ATOM이 소유한 DMA-BUF로 옮기고,
모든 DMA 완료 항목을 확인한 뒤에만 decode를 허용한다. 전송 계획, 버퍼 소유권, decode admission을 모델 실행과
분리하며, 이 데이터 경로를 llm-d의 prefill/decode stage에 연결한다.
[ai-rm](https://github.com/ai-computing/ai-rm)에서는 **L2 메모리·전송** 구성요소로, `KVTransport` 계약을
`airm.kv_transports:p2p`로 제공한다.

또한 **Optimus Prime 엔진의 이전(migration) 소스**다. KV 전송 기술은 Optimus Prime 모듈로 통합하는 방향이며
([통합 계획](https://github.com/ai-computing/ai-rm/blob/master/docs/engine-integration.md)), 같은 호스트 범위·ABI 제약·완료
fence·명시적 실패를 그대로 유지한다. 엔진 통합이 승인되기 전까지는 이 저장소가 실행 경로다.

> [!WARNING]
> **실험 단계의 연구 프로토타입.** 하드웨어 지원은 검증된 RNGD→ATOM adapter/ABI 조합으로 한정된다. P2P 출력은
> context 512(및 명시적으로 선택하는 context 448 경로)에서 같은 batch의 host admission 기준과 정확히 일치함을
> 확인했다. 이는 전송 정확도의 증거이며 모델 품질 동등성의 증거가 아니다. 범용 추론 런타임이 아니다.

## ✨ 주요 기능

<table>
<tr>
<td width="33%" valign="top">
<b>완료 확인 후 admission</b><br>
모든 DMA 완료 항목(CQE)을 확인한 뒤에만 decode를 시작한다. 실패했거나 불확실한 전송은 목적지를 재사용하지 않고 세션을 poison 처리한다.
</td>
<td width="33%" valign="top">
<b>조용한 host staging 없음</b><br>
ATOM 목적지 할당을 DMA-BUF로 export하고 RNGD PDMA가 쓴다. 백엔드는 KV payload를 host 복사로 우회하지 않는다.
</td>
<td width="33%" valign="top">
<b>명시적 전송 계획</b><br>
source 할당, 목적지 버퍼, scatter-copy 구간을 미리 기술한다. <code>TokenMajorLayout</code>은 token-major cache의 prefix 복사 계획을 만든다.
</td>
</tr>
<tr>
<td width="33%" valign="top">
<b>버퍼 수명 관리</b><br>
전송과 decode 동안 할당 소유 객체를 유지하고, 실패한 lease는 진단을 위해 보존한다.
</td>
<td width="33%" valign="top">
<b>llm-d stage 연동</b><br>
llm-d <code>pd-sidecar</code>(shared-storage)의 prefill/decode stage 계약을 구현한다. HTTP는 제어를, KV는 로컬 P2P 경로를 탄다.
</td>
<td width="33%" valign="top">
<b>서빙 기본 요소와 검증</b><br>
요청 검증, 용량 제어, cache lease, iteration 단위 스케줄링, CPU 계약 테스트, 하드웨어 검사, 보드 에너지 적분.
</td>
</tr>
</table>

## 🏗 구조

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

목적지 할당은 ATOM이 export하고 RNGD 전송 경로가 import한다. 이 백엔드는 KV payload를 host 메모리로 거치거나
host 복사로 조용히 대체하지 않는다. 모델 adapter는 표현 변환을 하거나 검증 등의 목적으로 host 메모리를 쓸 수 있다.
Python 계약은 모델과 무관하며, 하드웨어 adapter는 버전별로 다르다.

<details>
<summary><b>요구 사항</b></summary>
<br>

| 구성 요소 | 현재 요구 사항 또는 검증 범위 |
|---|---|
| Python | 3.10 이상, core 패키지는 표준 라이브러리만 사용 |
| 플랫폼 | 하드웨어 백엔드는 Linux x86-64 |
| 토폴로지 | source와 목적지가 같은 호스트. 대상 시스템에서 P2P 지원을 따로 확인해야 함 |
| Source | Furiosa RNGD, driver 2026.4.0 |
| 목적지 | Rebellions ATOM, driver 3.0.0, 검사된 thunk-library ABI |
| 목적지 메모리 | export 가능한 live eager 할당. 정적 컴파일 상주 cache는 이 adapter가 지원하지 않음 |
| 네이티브 helper | GCC, 맞는 RNGD driver 개발 헤더, 빌드된 liburing |
| 모델 예제 | 추가 vendor SDK, 모델 아티팩트, 모델별 의존성 |

버전 검사는 알 수 없는 환경을 거부한다. 버전이 맞는다고 장치가 사용 가능하거나 특정 PCIe 토폴로지에서 P2P가
동작한다는 뜻은 아니다. [지원 범위와 소유권 계약](docs/SUPPORT.md)을 참고한다.
</details>

## 🚀 빠른 시작

계약 테스트에는 NPU가 필요 없다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .              # Python 패키지 + `hetero-p2p` 명령
python -m unittest discover -s tests -v
```

테스트는 전송 범위, 버퍼 alias, 소유권 전이, 불완전한 완료, 스케줄링, 서빙 계약을 다룬다. test double과 host
기반 fixture를 쓰므로 통과가 하드웨어 DMA를 증명하지는 않는다. 설치에는 vendor driver, SDK, 모델 가중치, 네이티브
DMA helper가 포함되지 않으며 패키지 인덱스 설치를 가정하지 않는다.

<details>
<summary><b>Mock 전송 세션 (데이터를 복사하지 않음)</b></summary>
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

위의 합성 주소, host file descriptor, placeholder 소유 객체는 이 mock 예제에서만 유효하다. 하드웨어 전송에는
live SDK 할당에서 얻은 주소와 export된 descriptor가 필요하다.
</details>

<details>
<summary><b>하드웨어 연동: 네이티브 helper, doctor, 모델 연동 절차</b></summary>
<br>

로컬에 설치된 driver 헤더와 빌드된 liburing으로 helper를 빌드한다.

```bash
python native/build.py \
  --driver-root /path/to/furiosa-driver-rngd-2026.4.0 \
  --liburing-root /path/to/liburing-2.6
```

빌드 결과는 `build/rngd-pdma`와 빌드 manifest다. 백엔드가 사용 전에 helper를 manifest와 대조하므로 둘을 함께 둔다.

사용할 물리 장치 경로로 설치 환경을 확인한다.

```bash
hetero-p2p doctor --device /dev/rngd/npu3ch0
```

`doctor`는 버전, ABI, 장치 파일 호환성을 확인한다. 장치를 점유하거나 P2P 전송을 실행하지 않는다.

모델 연동 절차:

1. SDK로 source KV 할당을 만들고 동기화한다.
2. 지원되는 목적지 할당을 export하고 소유 SDK 객체를 유지한다.
3. 실제 물리 형식을 `TransferPlan` 또는 `TokenMajorLayout`으로 기술한다.
4. 소유 프로세스의 main thread에서 `RngdPdmaBackend(helper, device=...)`로 `CopySession`을 만든다.
5. 요청을 admit하고, 전송 완료를 확인한 뒤 decode를 실행하며, decode 완료를 확인한 뒤에만 lease를 반환한다.

백엔드의 장치 경로는 source 할당과 맞아야 한다. source 주소는 CPU 포인터가 아니라 RNGD DRAM의 raw offset이다.
세션이 소유한 동안 export된 descriptor를 직접 닫거나 SDK 할당을 해제하지 않는다. 실패했거나 불확실한 전송은
목적지를 재사용하지 않고 세션을 poison 처리한다.

[라이브러리 API와 서빙 계약](docs/SERVING_LIBRARY.md), [지원 범위](docs/SUPPORT.md),
[하드웨어 검증](docs/VALIDATION.md)을 참고한다.
</details>

<details>
<summary><b>llm-d와 모델 예제</b></summary>
<br>

llm-d 연동은 로컬 prefill/decode stage 계약을 구현한다. HTTP는 요청과 제어 정보를 나르고, 소유 프로세스가 KV
전송과 cache lease를 관리한다. NIXL/Mooncake connector나 노드 간 RDMA 전송이 아니다. 다만 전송 자체는 `HETERO_NPU`
백엔드 plugin으로 NIXL 클라이언트에서도 쓸 수 있다([NIXL 백엔드](docs/nixl-backend.ko.md)).

`examples/koni/`에는 KONI-Llama3.1-8B용 모델별 연구 예제가 있다. 추가 모델 코드, 컴파일된 아티팩트, 별도 vendor
환경이 필요하며 `pip`로 설치되는 완성형 서비스가 아니다.

- [llm-d 연결](docs/LLMD.md)
- [HTTP batching 설정](docs/HTTP_BATCHING.md)
- [예제 설정](examples/koni/serve.config.example.json)
- [Iteration 스케줄링](docs/ITERATION_SERVING_KO.md)
- [독립 decode replica](docs/ITERATION_REPLICAS_KO.md)

컴파일된 B4/B8 예제와 명시적으로 선택하는 B12/B14 경로를 실행해 보았다. batch 크기와 context 길이는 컴파일된
아티팩트와 가용 메모리에 달려 있으며, 실행 중에 바꿀 수 있는 스위치가 아니다.
</details>

## 🔌 ai-rm 연동

| | |
|---|---|
| **계층 / 종류** | `l2-memory` 라이브러리, 서빙 pod 안에서 사용 ([`ai-rm.yaml`](ai-rm.yaml)) |
| **제공** | `KVTransport` (`TransferPlan` / `CopySession` / `RngdPdmaBackend`, llm-d stage broker) |
| **사용** | 다른 ai-rm 구성요소를 쓰지 않는다. KONI 예제는 모델 실행에 `rngd-pd`의 `rngd_pd` 패키지를 `PYTHONPATH`로 불러온다 |
| **Adapter** | `hetero_p2p.airm:P2PTransport`, entry point `airm.kv_transports` 이름 `p2p` |
| **NIXL 백엔드** | `HETERO_NPU` plugin(`nixl_plugin/`, `libplugin_HETERO_NPU.so`): `nixl_agent`로 RNGD → ATOM DMA-BUF 전송([설계](docs/nixl-backend.ko.md)) |
| **지표** | `kv_transfer_ms`, `kv_transfer_gbps` (tag `kv_transport=p2p`) |
| **KPI** | 이종자원수(RNGD+ATOM 동시 사용), 전력효율(`hetero_p2p.energy`의 HTTP 구간 보드 에너지 적분) |
| **사이트 설정** | 없다. 엔드포인트·IP를 코드에 두지 않으며, 모델·SDK·아티팩트 경로는 `examples/koni/*.example.json`을 복사한 설정 파일로 전달한다 |

```python
from rm_core import registry
transport = registry.load("airm.kv_transports", "p2p")
```

```bash
python -m rm_core.conformance airm.kv_transports p2p
```

빌드한 helper(`HETERO_P2P_HELPER`)와 live 버퍼 fixture(`HETERO_P2P_CONFORMANCE_FIXTURE`)가 없으면 conformance는
이유와 함께 **SKIP**이다.

<details>
<summary><b>Protocol v0.1 adapter 상세</b></summary>
<br>

adapter는 기존 `TokenMajorLayout.prefix_plan`과 `RngdPdmaBackend.transfer`(모든 CQE 확인 후 반환)를 감쌀 뿐이며
host staging으로 대체하지 않는다.

- `supports`: 같은 노드의 `furiosa` 장치 메모리 → `rebellions` 장치 메모리(`device-hbm`/`device-dram`)만 True.
- `plan`: `dlfloat16` token-major 동일 layout, prefix 구간 `[0, end)`만 허용한다. rm-core 계획에는 실제 할당이
  없으므로 소유 프로세스가 먼저 `bind(request_id, P2PBinding(source, destinations, mappings, context, slot, batch))`로
  살아 있는 RNGD source와 ATOM export를 넘긴다. 범위·정렬·형식 오류는 `TransportError`.
- `submit`/`wait`: `submit`이 helper를 동기 실행해 완료 fence 이후 결과를 남기고, `wait`가 그 결과를 한 번만
  돌려준다. `submit`은 소유 프로세스의 main thread에서 호출한다. 실패는 `ok=False`(재시도·우회 없음). binding은 한
  번의 `plan`에서 소비된다. 성공마다 `kv_transfer_ms`, `kv_transfer_gbps`(tag `kv_transport=p2p`)를
  `transport.measurements`와 `metric_sink`로 낸다.
- 생성자 키워드: `helper`, `device`, `backend`, `metric_sink`. 환경 변수: `HETERO_P2P_HELPER`(빌드한 `rngd-pdma` 경로),
  `HETERO_P2P_DEVICE`(기본 `/dev/rngd/npu1ch0`), `HETERO_P2P_CONFORMANCE_FIXTURE`(`module:callable`, 아래).
  `HETERO_P2P_*`는 로컬 경로일 뿐 사이트 값이 아니다.
- llm-d stage: `StageBroker(..., kv_transport='p2p')`가 경로 이름을 받고(`/health`에 표시), 소유 프로세스는
  `hetero_p2p.llmd.load_kv_transport(name, **kwargs)`로 이름에서 구현을 불러온다.

`conformance_case()`는 환경 검사(`probe`)와 helper가 맞고, `HETERO_P2P_CONFORMANCE_FIXTURE`가 가리키는 callable이
`(src_tier, dst_tier, layout, P2PBinding)`을 돌려줄 때만 실제 tier로 시험한다. 그렇지 않으면 이유와 함께 SKIP이다.
fixture는 모델 SDK로 살아 있는 할당을 만들고 시험 동안 소유한다(`examples/koni/serve.py`의 준비 과정 참고).

전력효율은 보드 전력만 포함하므로 wall-plug 효율이나 MFU를 산출하지 않는다.
</details>

<details>
<summary><b>Kubernetes + llm-d: 아직 부족한 것</b></summary>
<br>

아직 Kubernetes 배포는 없다. `src/hetero_p2p/llmd.py`는 llm-d 공식 `pd-sidecar`(shared-storage)의 prefill/decode
stage 계약을 구현하고, `examples/koni/`에서 로컬 프로세스로 검증했다([llm-d 연결](docs/LLMD.md)). k8s 배포에 필요한 것:

- prefill/decode를 다른 Pod로 나눌 때 단일 cache owner로의 인증된 RPC와 Unix FD(DMA-BUF) 전달 경로
- 컨테이너 이미지와 매니페스트(RNGD·ATOM 디바이스, 네이티브 helper `build/rngd-pdma` 포함)
- EPP/Envoy 경로, 다중 노드, 장애 복구 검증
</details>

## 📊 결과

KONI-Llama3.1-8B, context 512, 입력 140/167/227/271토큰, 출력 길이 고정, greedy non-streaming, llm-d HTTP 경로.
3회 중앙값이며 전체 logits 검증 비용을 포함한다.

| 구성 (같은 3장: RNGD 1장 prefill + ATOM decode) | B4 PD | B8 PD | B8/B4 |
|---|---:|---:|---:|
| 16요청, 출력 32 | 85.66 tok/s | 134.83 tok/s | 1.57× |
| 16요청, 출력 128 | 100.52 tok/s | 176.50 tok/s | 1.76× |
| 32요청, 출력 32/64/96/128 혼합 | 93.10 tok/s | 127.47 tok/s | 1.37× |

- 비교한 424요청 모두 같은 batch의 host admission 기준과 **전체 logits가 정확히 일치**했다. 모든 요청에서 DMA
  CQE 64/64 완료를 확인했고, 요청당 DMA 중앙값은 약 3.0ms다([B8 결과](docs/PD_B8_RESULTS_KO.md)).
- RNGD 1장 + ATOM 1장 B12에서는 640요청이 exact였고 DMA 중앙값은 2.985ms였다([B12 결과](docs/PD_B12_RESULTS_KO.md)).
- 단일 카드 기준과의 비교(출력 128, 보드 에너지만): PD 3장 98.285 tok/s·0.6070 tok/J, RNGD 단독 66.284 tok/s·0.4326
  tok/J, ATOM vendor 단독 52.255 tok/s·0.8713 tok/J([비교](docs/benchmark-b4/RESULTS_KO.md)). PD는 처리량이 높았지만
  에너지 효율은 ATOM 단독보다 낮았다.

성능은 모델, batch 점유율, 입출력 길이, 요청 도착 패턴, 하드웨어 배정에 따라 달라진다. 전송 출력 일치는 모델 품질
동등성의 증거가 아니며, PD 처리량이 단독 추론보다 반드시 좋은 것도 아니고, 처리량 향상이 지연이나 에너지 효율
개선을 뜻하지도 않는다. 저장소는 회귀와 실패한 실험을 포함한 워크로드별 결과를 보존한다. 로컬 원시 결과, 모델
가중치, 컴파일된 아티팩트는 Git에서 제외될 수 있으며, 커밋된 요약에 출처와 한계를 적었다.

### RNGD → ATOM 전송 시간 곡선 (기존 경로 대 NIXL 백엔드)

한 호스트의 RNGD PE0 → ATOM, 평면 바이트 버퍼, 칸마다 5회 전송의 중앙값이며, 전송 시작부터 완료 fence(모든 CQE
회수)까지 쟀다. GB/s = 10⁹ bytes/s. 모든 칸을 ATOM에서 바이트 단위로 검증했다: destination을 먼저 poison 값으로 채우고,
전송 후 source 패턴과 같아야 하며 범위 밖 바이트는 그대로여야 한다. 2026-10-19, NIXL 1.5.0(pip wheel).
원자료는 [`docs/nixl/`](docs/nixl/curve-20261019.json), 방법은 [NIXL 백엔드](docs/nixl-backend.ko.md).

| 크기 (GiB) | 기존 경로 (helper, 호출당 64 MiB) | NIXL qd 1 | NIXL qd 4 | NIXL qd 8 | NIXL 32 layer, qd 4: 첫 layer / 전체 | RNGD → 호스트 구간 (staging) |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 (104 MiB) | 16.0 ms · 6.8 GB/s | 3.2 ms · 33.7 | 2.6 ms · 41.4 | 2.7 ms · 41.1 | 0.32 / 2.7 ms | 27.1 ms · 4.0 |
| 0.25 | 13.2 ms · 20.3 | 8.2 ms · 32.5 | 6.5 ms · 41.6 | 6.4 ms · 41.7 | 0.81 / 6.5 ms | 33.0 ms · 8.1 |
| 0.5 | 31.9 ms · 16.8 | 16.2 ms · 33.1 | 12.9 ms · 41.7 | 12.9 ms · 41.7 | 1.57 / 12.9 ms | 54.0 ms · 9.9 |
| 1 | 64.0 ms · 16.8 | 33.7 ms · 31.9 | 25.7 ms · 41.9 | 25.6 ms · 41.9 | 1.64 / 25.7 ms | 95.5 ms · 11.2 |
| 2 | 133.3 ms · 16.1 | 70.3 ms · 30.6 | 51.8 ms · 41.5 | 51.7 ms · 41.5 | 1.71 / 51.3 ms | 157.2 ms · 13.7 |
| 4 | 276.3 ms · 15.5 | 140.2 ms · 30.6 | 103.9 ms · 41.3 | 102.6 ms · 41.9 | 3.32 / 103.2 ms | 334.3 ms · 12.8 |

- NIXL 백엔드는 helper와 같은 PDMA DMA-BUF 명령을 낸다. 큐 깊이 1에서 helper의 약 2배다(64 MiB마다 helper
  프로세스 하나 대신 요청 하나). 큐 깊이 4에서 0.1–4 GiB 전 구간 약 41.5 GB/s를 유지하고, 깊이 8은 더 빨라지지 않는다.
- Layer-wise([aicomp#47](https://github.com/ai-computing/aicomp/issues/47)): layer별 요청 32개의 총시간은 요청 하나와
  같고, layer는 순서대로 fence에 도달한다. 첫 fence는 대략 max(layer 크기, 큐 깊이 × 16 MiB) / 대역폭 뒤에 온다 —
  4 GiB에서 103 ms 중 3.3 ms(qd 1: 4.0 ms, qd 8: 3.4 ms). prefill이 layer를 전송보다 빠르지 않게 만든다면, prefill
  뒤에 드러나는 전송은 버퍼 전체가 아니라 layer 하나 정도다. TTFT는 여기서 재지 않았다.
- staging 비교: RNGD → 호스트 구간만으로도(pin된 호스트 페이지로의 PDMA) 1 GiB 이상에서 11–14 GB/s이며, 호스트 →
  ATOM 복사는 아직 포함하지 않은 값이다(이 노드에서 ATOM SDK의 4 GiB 호스트 → 장치 복사는 호스트 버퍼 생성 포함 약 0.8 s).
- helper와 staging 열은 0.25 GiB 이하에서 실행마다 다르다(앞선 실행: 0.1 / 0.25 GiB에서 helper 13.7 / 17.7 GB/s,
  staging 18.5 / 11.7 GB/s, [`run1`](docs/nixl/curve-20261019-run1.json)). NIXL 열은 1% 이내로 같다.
- 4 GiB는 양쪽 모두 2 GiB allocation 두 개를 쓴다(두 SDK 모두 4 GiB 단일 tensor를 할당하지 못한다). 이 측정은
  토큰이 아니라 바이트를 확인한다. NIXL을 거친 KONI token-exact 실행에는 엔진 훅([aicomp#48](https://github.com/ai-computing/aicomp/issues/48))이 필요하다.

| 평가 | 근거 |
|---|---|
| 같은 batch host admission 대 P2P 출력 검증 | [검증 기록](docs/VALIDATION.md) |
| llm-d HTTP batching (B4) | [HTTP 결과](docs/HTTP_RESULTS_KO.md) |
| batch 폭 확장 | [B8 결과](docs/PD_B8_RESULTS_KO.md), [B12 결과](docs/PD_B12_RESULTS_KO.md) |
| B14/context448, 포화와 혼합 워크로드 | [결과와 한계](docs/B14_CONTEXT448_KO.md) |
| 스케줄링 정책과 연구 근거 | [연구 기록](docs/OPTIMIZATION_RESEARCH_KO.md) |
| 메모리 제약과 실패한 구성 | [B16 가능성 검토](docs/B16_FEASIBILITY_KO.md) |
| NIXL을 거친 RNGD → ATOM, 0.1–4 GiB 곡선 | [NIXL 백엔드](docs/nixl-backend.ko.md), [원자료](docs/nixl/curve-20261019.json) |

## ⚠️ 한계

- 하드웨어 지원은 검증된 adapter/ABI 조합과 전송 방향으로 한정된다.
- 다른 모델에는 호환되는 할당, layout, 변환, 동기화 adapter가 필요하다.
- 다중 노드 전송, 완전한 Kubernetes/EPP/Envoy 배포, 일반 sampling, 운영용 token streaming은 검증된 예제 범위 밖이다.
- DMA 중 프로세스 강제 종료나 장치 reset은 지원하는 복구 방법이 아니다.
- 보드 전력 적분은 host CPU/DRAM을 포함하지 않으며 wall-plug 효율이나 MFU를 뜻하지 않는다.

## 🛠 개발

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

하드웨어 adapter를 바꿀 때는 대상 하드웨어에서 실제 할당/export 동작, 물리 layout, 완료 처리, 출력 정확도, 정상
종료도 확인한다. 실패한 실행을 보존하고 SDK 버전, 소스 revision, 측정 조건을 기록한다.

| 디렉터리 | 내용 |
|---|---|
| `src/hetero_p2p/` | Python 전송, layout, 스케줄링, 서빙 계약 |
| `native/` | RNGD PDMA helper와 빌드 스크립트 |
| `nixl_plugin/` | `HETERO_NPU` NIXL 백엔드 plugin (C++, meson) |
| `examples/nixl/` | NIXL 전송 벤치마크와 ATOM export owner |
| `examples/koni/` | 모델별 연동·벤치마크 도구 |
| `tests/` | CPU 계약 테스트와 네이티브 검증 도구 |
| `docs/` | API 안내, 실험 보고서, 지원 범위 |

## 📚 문서

| 목적 | 안내 |
|---|---|
| 🔧 **라이브러리 사용** | [서빙 라이브러리 API](docs/SERVING_LIBRARY.md) · [지원 범위와 소유권](docs/SUPPORT.md) · [Batch pipeline](docs/BATCH_PIPELINE.md) |
| 🔌 **llm-d 연결** | [llm-d 연결](docs/LLMD.md) · [HTTP batching](docs/HTTP_BATCHING.md) · [길이별 스케줄링](docs/LENGTH_SCHEDULING.md) |
| 🔁 **NIXL에서 사용** | [HETERO_NPU NIXL 백엔드](docs/nixl-backend.ko.md) |
| 📊 **근거 확인** | [검증](docs/VALIDATION.md) · [벤치마크](docs/benchmark-b4/RESULTS_KO.md) · [연구 기록](docs/OPTIMIZATION_RESEARCH_KO.md) |
| 🧭 **ai-rm 맥락** | [ai-rm](https://github.com/ai-computing/ai-rm) · [Optimus Prime 통합 계획](https://github.com/ai-computing/ai-rm/blob/master/docs/ko/engine-integration.md) · [rm-core protocol](https://github.com/ai-computing/rm-core/blob/main/PROTOCOL.md) |

## 📄 라이선스

프로젝트 라이선스는 아직 `LICENSE` 파일로 선언하지 않았다. 공개 저장소로 접근할 수 있다는 사실만으로 오픈소스
라이선스가 부여되지는 않는다. vendor SDK, driver 구성 요소, 모델 아티팩트는 각자의 조건을 따른다.

<div align="center"><sub><a href="https://github.com/ai-computing/ai-rm">ai-rm</a>의 구성요소 · ai-computing 팀 · ETRI</sub></div>
