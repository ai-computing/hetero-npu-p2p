# 빈 슬롯에 따른 decode quantum 조절

## 문제와 구현

기존 iteration 경로는 기본 quantum128이었다. 한 요청이128토큰을 생성하는 동안 나머지7슬롯이 비어 있어도, 뒤늦게 들어온 요청은 다음 완료 경계까지 기다릴 수 있었다. 이는 ORCA식 iteration 스케줄링을 실제 서비스에 적용할 때 실행 경계의 간격도 함께 정해야 함을 보여준다.

`QuantumPolicy(full=128, partial=4)`는 슬롯이 꽉 찼을 때128, 일부만 사용 중일 때4 step을 선택한다. 실제 step 수는 가장 빨리 끝날 요청의 잔여 길이로 한 번 더 제한한다. 새로운 슬롯 쓰기는 이전과 동일하게 worker의 명시적 장치 동기화 및 완료 응답 검증 뒤에만 허용한다. 이 정책은 이번 구현의 자체적인 제어 규칙이며 특정 논문의 알고리즘을 그대로 재현했다는 주장이 아니다.

4 step은 실측으로 약0.32초에 해당하지만 wall-clock 지연 보장은 아니다. source 준비·P2P admission·프로세스 스케줄링 비용이 추가된다. quantum128은 full batch의 불필요한 IPC를 줄이려는 선택이며, 동적 출력 길이 예측이나 preemption을 구현한 것은 아니다.

## 실험 설계

실행 ID `pd-quantum-r1`, 최초 `/tmp/hetero-pd-quantum-r1`. RNGD3/논리2 + ATOM2 한 장, KONI B8/context512, 같은 artifact·전체 logits 검증·lookahead16·collection20ms 유지. 세 정책 모두 slot refill 활성화:

- fixed128: full/partial128.
- fixed4: full/partial4.
- adaptive: full128/partial4.

세 정책 순서를 매 반복 한 칸씩 회전하고 각각3회 비교한다. burst128은16요청을 동시에 시작하며 모두128토큰을 생성한다. staggered는128토큰 요청 하나를 먼저 시작하고2초 후32토큰 요청15개를100ms 간격으로 제출한다. 각 요청의 예정·실제 시작 시각을 남겨 클라이언트 발송 오차도 검사한다. warm16 + 3정책×2조건×3회×16 = 총304요청이다.

처리량은 전체 시나리오의 출력 토큰 수/첫 제출 기준 경과시간으로 계산한다. staggered는 의도적인 도착 간격을 포함하므로 최대 포화 처리량으로 해석하지 않는다. 응답시간은 각 요청 실제 제출부터 최종 응답까지이며 non-streaming이다. 공개 trace가 아니라 제어된 도착 패턴이므로 실제 서비스 분포 전체에 일반화하지 않는다. 전력/MFU/SLO는 이번 시험에서 측정하지 않는다.

모든 slot lease·generation·출력 진행량을 replay 검증하고, 실제 quantum이 정책의 점유율별 상한을 넘지 않았는지 검사한다. 기존57개 단위시험 통과.

실행: 기존 sidecar18823 및 `serve_iteration.py --config artifacts/koni-pd-quantum-local.json`, `validate_quantum.py --config artifacts/koni-pd-quantum-local.json`.
재분석: `python3 examples/koni/analyze_quantum.py results/pd-quantum-r1 --output /tmp/quantum-analysis.json`.

관련 출발점은 [ORCA](https://www.usenix.org/conference/osdi22/presentation/yu)의 iteration-level scheduling이며, 과거 [step 경계 구현](ITERATION_SERVING_KO.md)과 별도 ablation이다. 전체 최적화 목표와 두 ATOM 확장은 계속 진행 중이다.

## 실측 결과 (2026-10-16)

| 지표 (3회 중앙값) | fixed128 | fixed4 | adaptive |
|---|---:|---:|---:|
| burst128 처리량 |94.12 tok/s|93.25 tok/s|94.08 tok/s|
| staggered 처리량 |39.51 tok/s|57.21 tok/s|56.89 tok/s|
| staggered 출력32 응답시간 |10.71초|4.82초|4.77초|
| staggered 출력128 응답시간 |10.04초|10.63초|10.69초|

적응형은 fixed128 대비 시간차 도착 처리량44.0% 증가, 짧은 요청 응답시간55.5% 감소였다. 대신 먼저 시작한 긴 요청은0.65초(6.4%) 느려졌다. 포화 처리량은−0.03%로 사실상 동일했다. fixed4도 대부분의 이득을 얻었으며 adaptive의 처리량이 fixed4보다 항상 높다는 증거는 없다. 다만 burst128의 전체 quantum 수는 fixed4가192회, adaptive가6회여서 불필요한 제어 왕복을 줄였다(각3회 합계).

적응형 staggered 반복56.87/56.89/57.50 tok/s, fixed12839.29/39.51/39.74 tok/s였다. 해당 조건의 최대 발송 지연은3.6ms 미만, burst는10.5ms 미만이었다. 유효 lane 비율39.58→59.38%이며 이는 MFU가 아니다. 요청 도착 전에 비어 있는 슬롯도 분모에 포함되므로 포화 효율과 혼동하지 않는다.

304요청 모두 전체 logits exact, DMA64/64 CQE 완료, 실행 중 KV 쓰기 없음, 모든 generation/출력 진행량 검증 통과. 다른 요청이 계속되는 가운데 슬롯 교체90건을 확인했다. worker와 서버 정상 종료, 장치 반환 확인. DMA 중앙값2.923ms.

판정: **별도 실험 서버 `serve_iteration.py`의 초기 정책을 adaptive(full128/partial4)로 변경**했다. 이전 length 서버와 vendor runtime 기본값은 그대로다. 실험에서는 동일 값을 policy control로 적용했고 원시 source snapshot을 보존했다. 긴 요청 지연 증가와 단일 모델·단일 arrival 패턴 제한은 유지된다. 실제 trace와 두 ATOM에서 재검증해야 한다.

라이브러리의 `accept_reply`가 반환하는 완료 entry에도 최종 tokens를 보존하도록 보완했다. 이전에는 살아 있는 entry만 갱신해 완료 entry의 tokens가 마지막 quantum 이전 상태였다. HTTP 응답은 항상 worker reply의 전체 tokens를 사용했으므로 위 실측 출력에는 영향이 없었다. 완료/계속 entry 모두에 대한 회귀 검사를 추가해57개 시험 통과. 이 보완과 초기 정책 변경은 하드웨어 run 종료 후 적용했으며 측정 시점의 코드와 구별한다.

[그래프](quantum-comparison.png) · [설정·각 반복·source hash](quantum-validation.json). 원본은 `results/pd-quantum-r1`에 이동했고, 측정 당시 소스 snapshot·패치·HTTP/event/worker 로그·기준 logits·장치 상태를 보존했다.

그래프 재생성: `python3 examples/koni/plot_quantum.py docs/quantum-validation.json docs/quantum-comparison.png`.
