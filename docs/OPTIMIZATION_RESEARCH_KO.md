# 논문 기반 PD 최적화 연구 기록

2026-10-16 시작. 목표는 이종 RNGD→ATOM PD와 지원 라이브러리에서 개선 여지를 실측으로 확인하고 구현하는 것이다. 처리량만 올리는 대신 응답 지연·정확성·자원 수·전력을 구분해 평가한다. 목표는 진행 중이며 아래 첫 실험으로 완료를 선언하지 않는다.

## 논문과 적용 판단

| 1차 자료 | 핵심 기법 | 현 구현에 대한 판단과 도입 단계 |
|---|---|---|
| [DistServe, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | prefill/decode 분리와 단계별 자원·병렬성 배치, SLO 기반 goodput | 이종 PD 기본 방향과 부합한다. 기존 처리량 시험을 요청 도착률·TTFT/TPOT·goodput 시험으로 확대해야 한다. 논문의 GPU 성능 배율을 NPU에 적용하지 않는다. |
| [ORCA, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | iteration-level scheduling, selective batching | 현재 batch 전체 완료까지 슬롯을 붙잡는다. step 종료 확인→완료 슬롯 교체→KV 동기화→다음 step 프로토콜을 구현할 후보. 실행 중 kernel의 모든 슬롯 접근 때문에 명시적 fence 없이 DMA하면 안 된다. 아직 미구현. |
| [Sarathi-Serve, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal) | chunked prefill, stall-free scheduling, 일정한 작업량 | 이종 카드에서는 prefill/decode가 이미 분리되어 있어 같은 GPU에서의 hybrid batching을 그대로 이식할 수 없다. producer 작업량 제한과 decode control 응답 지연을 줄이는 원리를 검토한다. 현재 prefill은 chunked 구현이 아니다. |
| [TurboTransformers](https://arxiv.org/abs/2010.05680), [공식 구현](https://github.com/Tencent/TurboTransformers) | 가변 입력 길이 배치의 비용 기반 동적 계획법 | 길이별 비용 모델을 적용하는 아이디어를 고정 폭 autoregressive decode에 응용한다. 새 padding 정책은 아래의 자체 알고리즘이며 논문의 DP 재현은 아니다. |

## 첫 구현: padding 비용 최소화

기존 `length` 정책은 가장 오래 기다린 요청의 출력 길이와 가까운 요청을 고른다. 새 `padding`은 같은 bounded lookahead 안에서 가장 오래 기다린 요청을 반드시 포함하고, 선택한 요청 수를 고정한 상태에서 `B * max(L_i) - sum(L_i)`를 최소화한다. B는 컴파일된 배치 폭이다. 가능한 최대 길이마다 그 이하 요청 중 길이가 큰 것을 우선 선택하여 최소 비용 후보를 구한다. 동률이면 더 짧은 최대 길이, 더 오래 기다린 요청 순으로 결정한다.

이 비용은 버려지는 slot-step 수의 대리 지표다. 개별 배치 선택에 대한 최적화이며 전체 요청열의 전역 최적 처리량·실제 커널 시간·실제 EOS 길이에 대한 최적성을 주장하지 않는다. 기존 FIFO와 length 정책도 유지한다. 100개 고정 seed 작은 문제에서 모든 부분집합을 열거한 최적값과 비교하고, 오래된 요청 포함·부분 배치·한 슬롯 선택을 검사했다.

## 고정한 첫 하드웨어 실험

- 실행 ID `pd-padding-r1`, 최초 경로 `/tmp/hetero-pd-padding-r1`.
- RNGD 물리3/논리2, ATOM2 한 장, KONI B8. 기존 모델·artifact·정밀도·전체 logits 검증 유지.
- lookahead16, collection20ms, partial wait100ms 고정. 스케줄링 정책만 변경한다.
- 동일 서버/동일 메모리 상주 상태에서 length→padding / padding→length / length→padding 순서, 각 조건3회.
- 조건: 16요청 출력32, 16요청 출력128, 32요청 출력32/64/96/128 혼합. warm16 포함400요청.
- HTTP 전체 처리량·각 반복·응답시간·lane 사용률·DMA 완료·전체 logits 일치 기록. 실패도 원본 유지. 공유 호스트 순차 비교이며 완전한 통계적 유의성이나 실제 서비스 분포 대표성을 주장하지 않는다.
- 실행: `python3 examples/koni/validate_padding_policy.py --config artifacts/koni-pd-padding-local.json` (서버와 sidecar 먼저 시작).
- 실험 결과로 개선되지 않으면 기본 정책을 바꾸지 않는다. lookahead 확대는 별도 ablation으로 분리한다.

## 다음 검증 단계

1. 첫 정책 비교 결과와 stage별 비용 분석. 부분/혼합 배치의 대리 비용 감소가 실제 처리량 개선인지 확인.
2. lookahead·대기시간 tradeoff, burst 외 요청 도착 분포, 1/2 ATOM 비교.
3. ORCA식 step 경계 프로토콜: fake backend에서 소유권/세대/오류 테스트 후 실제 host-KV→P2P 순으로 검증. 같은 live slot을 실행 중 덮어쓰지 않는다.
4. CPU logits 처리·helper invocation·prefill staging 분리 계측 및 비용 축소. 검증 모드와 서비스 모드를 명시하고 동일 모드끼리 비교.
5. 단독 B8 및 다중 ATOM 기준, 전력/MFU/TPOT p99/TTFT와 SLO goodput 재검증. 개선폭이 반복 변동 이내이면 plateau 근거를 남기고 다음 병목으로 이동.

## Iteration 스케줄러 선행 구현

`src/hetero_p2p/iteration.py`에 ORCA의 iteration scheduling 원리를 적용할 상태 관리 계층을 추가했다. `begin(quantum)`은 quantum 또는 가장 빨리 끝날 요청까지의 step 수 중 작은 값을 선택한다. `complete(ticket, device_completed=True, passed=True)`가 확인된 뒤에만 완료 슬롯을 반환하고, 나머지 요청의 생성 토큰 수·버퍼 owner를 유지한다. 오래된 ticket은 거부하며 실패·불확실한 완료는 슬롯을 보존하고 poison 상태로 전환한다.

이는 **장치 동기화 자체를 수행하는 함수가 아니다**. 향후 worker가 동기 `step_batch` 반환 및 해당 step의 출력 검증을 마친 후 completion을 보내야 한다. bool이나 타이머만으로 실제 device fence를 대신할 수 없다. 아직 기존 HTTP worker와 연결하지 않았고 하드웨어 성능 개선 결과도 없다. 단위시험은 최초 완료 슬롯 교체, 잔여 토큰 보존, DMA 미완료 launch 금지, stale ticket, 불확실한 완료에서 owner 보존, quantum별 진행을 검증한다. 전체52시험 통과.

기존 `pd-one-r1` 사건 로그를 합산하면 출력128의3회 HTTP 구간은67.496초, batch 시작~완료 합은63.853초였다. batch 밖 시간은3.643초(약5.4%)다. 출력32는19.249초 중15.805초가 batch 구간이다. 이는 구간별 wall time이며 GPU/NPU kernel만의 실행시간이나 MFU는 아니다. 긴 출력은 batch 안의 빈 lane·step 비용 최적화가 더 큰 후보임을 시사한다.

## 첫 하드웨어 결과: 처리량 이득과 지연 손해

`pd-padding-r1` 완료, 400요청 모두 전체 logits exact, 요청별 DMA64/64 CQE 완료, 실행 중 replica에 DMA 없음. worker 정상 종료. 원본은 `results/pd-padding-r1`, 분석은 [padding-policy-validation.json](padding-policy-validation.json)에 보존했다.

| 조건 | 기존 length 중앙값 | padding 중앙값 | 변화 |
|---|---:|---:|---:|
| 출력32,16요청 |79.79 tok/s|80.11 tok/s|+0.41%|
| 출력128,16요청 |92.32 tok/s|92.61 tok/s|+0.32%|
| 혼합32요청 |72.83 tok/s|78.52 tok/s|+7.81%|

고정 길이는 정책으로 줄일 padding이 없어 작은 차이를 개선으로 주장하지 않는다. 혼합 반복은 length78.39/72.83/72.51, padding73.02/78.52/78.73 tok/s로 범위가 겹친다. 순서·도착 타이밍 변동이 있으며3회만으로 통계적 우위를 확정하지 않는다.

혼합의 유효 lane 비율은78.95→81.08%였으나, 출력32 요청의 HTTP 응답시간 중앙값은 **3.78→15.61초**로 악화했다. 출력64는22.16→8.92초, 출력96은24.74→14.24초, 출력128은16.88→24.72초였다. 가장 오래된 요청을 포함하는 국소 공정성만으로 길이별 지연의 악화를 막을 수 없다. 전체 응답시간 중앙값19.52→19.45초만 보면 이 문제가 숨는다.

판정: padding은 실험 옵션으로 유지하되 **기본 정책은 length 그대로 유지**한다. 처리량7.8%만을 근거로 기본값을 교체하지 않는다. 다음 핵심 실험은 iteration 경계에서 완료 슬롯을 새 요청으로 채우는 경로로, 지연을 크게 바꾸지 않고 빈 lane 실행을 줄일 수 있는지 검증한다. 요청 도착 분포와 더 다양한 출력 길이에서도 확인해야 한다. 전체 목표는 계속 진행 중이다.


## 후속: iteration 경로 실제 구현 완료

[Step 경계 슬롯 교체 실험](ITERATION_SERVING_KO.md)에서 동일 서버의 sealed/refill을 비교했다. 410요청 exact, 진행 중 trajectory를 보존한 refill70건 통과. 혼합 처리량 중앙값+8.17%, 전체 응답시간 중앙값−18.44%였으나 출력128 지연은 증가했다. 다음 단계는 시간차 도착·adaptive quantum·두 ATOM 확장이며 전체 최적화 목표는 계속 진행 중이다.

## 후속: 시간차 도착에서 adaptive quantum

[304요청 비교](ADAPTIVE_QUANTUM_KO.md)에서 빈 슬롯일 때4step, 꽉 찼을 때128step을 실행하는 정책을 검증했다. fixed128 대비 시간차 도착 처리량+44.0%, 짧은 요청 응답시간−55.5%, 포화 처리량 유지. 긴 요청은+0.65초였다. fixed4도 유사한 이득이며 적응형의 고유 이점은 full batch 제어 왕복 감소다. 실험 iteration 서버의 초기 정책으로 채택했으나 일반적 우위를 주장하지 않는다. 다음은 두 ATOM 확장, 다양한 도착률·출력 분포, 실제 token-interval 지연과 전력 측정이다.


## 추가 논문 검토: 다중 replica 부하

[Llumnix, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/sun-biao)는 실행 중 요청과 상태의 migration을 통해 instance 간 부하 및 지연 요구를 조절한다. 현재 검증한 DMA는 RNGD→ATOM admission이고 ATOM→ATOM live migration은 구현하지 않았다. 우선 초기 배치 충전과 카드별 token work/잔여 길이에 따른 admission 배치부터 검증할 후보로 삼는다. 요청 수만으로 부하를 판정하지 않으며 논문의 성능 배율을 NPU에 전용하지 않는다.

## 후속: 두 ATOM 확장과 성능 저하 보존

[두 replica 검증](ITERATION_REPLICAS_KO.md)을 완료했다. 독립적인 generation/lease로848요청의 전체 logits를 확인했다. 초기 부분 batch8+6+2 문제를 고쳐 고정 길이16요청을8+8로 채웠고, 최종 sealed는136.35/177.73 tok/s였다. 혼합은 sealed147.01, refill117.31 tok/s로 refill이 손해여서 두 장 기본값은 sealed로 두었다. 한 장의 정책 이득을 두 장에 일반화하지 않는다. 초기 실패·성능 저하·최종 반복을 모두 보존했다. 다음은 카드 배치와 더 큰 batch feasibility, token-interval/전력 평가다.

## 2026-10-16: 배치 확대와 KV 메모리 제약

[PagedAttention, SOSP2023](https://arxiv.org/abs/2309.06180)은 KV 단편화·중복을 줄여 더 큰 배치를 수용하는 방향이다. 현재 관측한 B16 eager 실패는 논리 float32 캐시의 사전 할당 비용이며, 이를 단순히 단편화 문제로 단정하지 않는다. 현재 고정512 block/slot graph에서 논문의 동적 page pool을 쓰려면 graph block table, runtime 메모리 계약, P2P 목적지 레이아웃을 함께 바꾸고 재검증해야 한다. op 이름이 paged attention이라고 해서 그 메모리 관리가 구현된 것은 아니다. 우선 B12로 같은 context의 가능한 배치 폭을 실측하고, 이후 논리/물리 캐시 저장량과 export 계약을 분리 계측한다. 논문의 GPU 처리량 배율을 NPU 개선 예상치로 사용하지 않는다.

## 2026-10-16: 다음 graph 비교 시험

B12 eager host wall 계측에서 runtime 호출83.849ms/전체85.987ms였다. Kernel profiler는 아니지만 입력·출력 Python 처리만으로 큰 개선을 얻을 근거는 약하다. 현재 P2P graph는 attention 출력 투영을8개 group 부분 곱과 순차 합산으로 계산한다. ResidentDecode에 full_output_projection 옵션이 있다. 과거 optimization_round2는 resident B24/host KV 경로에서 full projection을 사용했으나 현재 external-KV B12의 동일조건 비교는 확인되지 않았다.

compile_b16.py에 HETERO_FULL_OUTPUT_PROJECTION=1 명시 옵션과 결과 기록을 추가했다. 기본값0은 유지한다. 현재48요청 비교를 마친 뒤 별도 artifact로 컴파일하고, 같은배치 전송 정합성 및 독립 FP32 수치를 검증할 예정이다. 반올림 순서가 달라지므로 기존artifact의 bitwise 일치만으로 품질을 판단하지 않는다. 새 옵션의 컴파일·성능 시험은 아직 수행하지 않았다.

## 2026-10-16: 메모리 제한 이후의 후보

[KIVI, ICML 2024](https://arxiv.org/abs/2402.02750)는 key를 channel 방향, value를 token 방향으로 양자화하는 비대칭 KV 압축을 제안한다. 현재 경로에 적용하려면 압축된 KV를 attention이 직접 소비하거나 작은 단위로 복원해야 한다. 전체 KV를 매 step 복원한 채 별도 원본까지 유지하면 이 환경의 eager 할당 문제를 해결하지 못할 수 있다. 이는 현재 코드의 메모리 수명에 근거한 적용 판단이다.

현재 선택은 먼저 B14/context448의 정밀도를 유지한 경로를 검증하는 것이다. KIVI는 아직 구현하지 않았으며 GPU 논문의 처리량 배수를 ATOM 예상치로 사용하지 않는다. 후속 도입에는 compiler의 저정밀 attention/복원 지원, P2P 물리 레이아웃 계약, 실제 peak memory와 품질 검증이 필요하다. 공식 RBLN0.11.0 검색에서 이 custom graph의 INT4 지원을 확증하지 못했으므로 지원 여부는 미확인이다. 검색 부재를 기능 부재로 간주하지 않는다.

## 혼합84요청의 후속 후보 선택

B12/full/context512 혼합84요청에서는 sealed/refill 중앙값115.103/115.258tok/s로 거의 같고, 출력32 응답은24.532→33.284초로 악화했다. 완료620요청은exact지만 시작소스기록이 누락되어 사후기록임을 명시했다. Worker step 시간85.806–87.347ms에 비해 총step은576–704로 달라졌다.

`simulate_mixed_queue.py`는 관측한 요청 순서만 보존하고 모든 요청을 처음부터 준비된 상태로 둔 진단 도구다. B12의6개 순서에서 lookahead32/length/sealed는640–704step, padding/sealed는모두608step이었다. 이 결과는 prefill/DMA/HTTP 대기와 실제 도착시간을 무시하므로 처리량 예측이나 개선 실측이 아니다. TurboTransformers의 비용 기반 배치 원리에서 착안해 앞서 구현한 padding 정책을 더 큰 배치에서도 검증할 이유이며, 논문 알고리즘 재현은 아니다. 다음 후보는 lookahead를 고정한 length/padding 비교이고, 대기열 확대는 별도 실험으로 분리한다. 세부기록은 workspace `experiments/2026-10-16-b12-b14-mixed84.md`에 있다.

## B14 padding 혼합84 실측: 처리량과 지연의 상충

[비교 JSON](b14-padding84-validation.json)과 [전체 반복·응답 그래프](b14-padding84.png)에 결과를 보존했다. 같은 B14/context448/full/lookahead32에서 length→padding의 sealed 중앙 처리량은124.838→125.586tok/s(+0.60%), refill은139.044→125.811tok/s(-9.52%)였다. 새padding실행620요청 전체exact/소유권위반0/정상종료를 확인했다. Padding을 처리량 기본정책으로 채택하지 않는다.

Refill의 출력32 응답시간은31.198→16.624초로 개선됐지만 출력128은31.121→36.618초로 악화했다. 요청별 우선순위 변화가 있어 처리량만으로 정책을 평가할 수 없다. All-ready 모의시험은 실제 개선을 보장하지 않았다. 다음은 같은padding정책에서 lookahead32→84만 바꾸는 시험이며, 더 넓은 배치 선택 범위와 prefill 수집 지연을 함께 평가한다. 순차 단일프로세스 비교·고정4프롬프트/출력 상관이라는 한계는 유지된다.
