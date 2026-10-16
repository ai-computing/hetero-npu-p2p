# ATOM 두 장의 독립적인 iteration 슬롯 교체

2026-10-16. 한 RNGD의 prefill을 ATOM2/3 두 decoder replica로 전달한다. 두 카드는 모델 복제본이며 한 요청의 decode를 카드 사이에서 나누는 tensor parallel 구성이 아니다.

## 구현과 검증 범위

`serve_iteration.py`를 설정의 ATOM 목록으로 일반화했다. 각 카드에 별도의 `IterationScheduler`, generation, slot lease를 두고 selector가 읽은 connection의 카드로만 완료를 전달한다. 한 카드가 실행 중이어도 다른 카드의 fence가 확인되면 후자의 빈 슬롯에 admission할 수 있다. 전역 request ID 중복 검사와 source owner 보존은 유지한다.

prefill source 준비보다 이미 fence된 활성 decoder 재개를 우선한다. 한 장일 때와 같은 원리를 두 장에 적용해, 한 카드가 실행 중이라는 이유만으로 다른 카드의 재개를 source 준비 뒤로 미루지 않는다. admission은 같은 카드 안의 빈 슬롯에만 수행하며 다른 카드가 준 완료를 잘못 사용해 해제하지 않는다.

가짜 adapter 두 개를 이용해 한 카드의 in-flight ticket이 유지되는 동안 다른 카드만 완료·재예약되는 상황을 검사했다. 최초58개, 초기 배치 수집 검증 추가 후60개 시험 통과. 실제 실행에서는 각 replica의 generation/slot/token 진행량을 독립 재생하고, 두 replica가 모두 사용됐는지 확인한다.

## 실험 설계

- 보완 실행 ID `pd-iter-two-r3`, RNGD 물리3/논리2 + ATOM2/3, KONI B8, 같은 모델·artifact·정밀도·llm-d HTTP control.
- 최초 `pd-iter-two-r1`은 ATOM 사전 검사에서 PID2356488의 점유가 감지돼 DMA 이전 중단. 프로세스 종료와 장치 idle을 재확인하고 새 ID로 재시도했다. 실패 원본은 `results/pd-iter-two-r1-busy`에 보존했다. 해당 프로세스를 종료하거나 그룹을 변경하지 않았다.
- sealed/refill 모두 full quantum128/partial4, lookahead16, collection20ms, 같은 server/worker를 사용한다. **슬롯 교체 여부만** 달리한다.
- probe24요청: 출력1/8/2/16/3/32/4/64를3회 반복, FIFO, full quantum16/partial4. 기준 logits와 trajectory 보존을 확인한다.
- warm16 + 3회×2정책×(출력32의16요청+출력128의16요청+혼합32요청) = probe 포함424요청.
- sealed→refill / refill→sealed / sealed→refill 순서. mixed 출력32/64/96/128, 입력140/167/227/271, non-streaming, 고정 생성 길이, 전체 logits 검사 유지.
- 두 카드의 전체 처리량·길이별 응답시간·유효 lane 비율·카드별 요청 배분을 보고한다. lane 비율은 MFU가 아니다. 전력·SLO를 이번에 재측정하지 않는다.
- 과거 한 장/두 장 서버 수치와 직접 비교하면 policy·실행 시점 차이가 섞인다. 이번 같은 두 장 서버의 sealed/refill 비교가 슬롯 교체 효과의 기준이다.

재분석: `python3 examples/koni/analyze_iteration_replicas.py results/pd-iter-two-r3 --output /tmp/two-analysis.json`.
실행: `serve_iteration.py --config artifacts/koni-pd-iter-two-local.json`, `validate_iteration_replicas.py --config artifacts/koni-pd-iter-two-local.json`, 기존 sidecar18823 구성. 재실행 시 config out은 새 경로로 변경한다.

전체 최적화 목표는 계속 진행 중이며 다양한 도착률·trace, token interval p99, 전력 및 단독 NPU 동일 조건 재평가는 남아 있다.


## 보존한 초기 실행과 보완

`pd-iter-two-r2-initial`은424요청 exact와 독립 fence 검사를 통과했으나, 새 batch 수집 대기가 없어 고정 길이16요청이8+6+2로 나뉘었다. sealed 출력12894.39, refill173.70 tok/s의 큰 차이는 이 불완전한 기준에 영향을 받으므로 대표 개선치로 쓰지 않는다. 혼합은 sealed132.12/refill129.34 tok/s로 오히려 악화했다.

초기 혼합에서 refill의 요청 수는 ATOM2/3에64/32였으나, 출력 토큰은4096/3584였다(3회 합계). 요청 수의2배 차이를 연산 부하의2배 차이로 해석하지 않는다. 카드별 token work와 종료 시점도 함께 봐야 한다.

보완은 새 부분 batch의 초기100ms 수집 대기, 아직 시작하지 않은 batch의 refill 허용, 초기 수집 중 prefill 진행 허용이다. 이미 생성된 토큰이 있는 trajectory는 수집 대기를 적용하지 않는다. `IterationScheduler.should_dispatch`의 독립 시험은 DMA 미완료 금지·부분 batch 대기·full 즉시 시작·계속되는 trajectory 즉시 재개를 검사한다. r3은 이 보완을 sealed/refill 모두에 동일하게 적용한다.

## 보완 후 최종 결과

| 조건 (각3회 중앙값) | sealed | refill |
|---|---:|---:|
| 출력32,16요청 |136.35 tok/s|136.49 tok/s|
| 출력128,16요청 |177.73 tok/s|176.66 tok/s|
| 혼합32요청 |147.01 tok/s|117.31 tok/s|

고정 길이는 모두8+8로 배치되어 유효 lane 비율100%였다. sealed 출력128은176.97/177.73/177.97 tok/s 범위다. 따라서 초기8+6+2 문제를 고쳤음을 원시 slot 배치와 처리량으로 확인했다.

혼합은 sealed147.01/147.85/114.75, refill130.28/116.24/117.31 tok/s다. 중앙값은 refill이20.2% 낮지만 범위가 겹치고3회뿐이므로 모든 부하에서의 통계적 우열을 주장하지 않는다. 유효 lane 비율은sealed81.08%, refill75.00%였다. refill의 카드별 token work는4384/3296, sealed는3712/3968로, 단순 카드 순회와 길이 선택이 multi-replica의 전역적인 배치 낭비를 최소화하지 못한다.

혼합에서 출력32 응답시간 중앙값은3.77→5.04초로 악화했고, 출력64는14.41→11.58초, 출력96은9.63→9.50초, 출력128은16.70→12.12초로 개선됐다. 전체 응답시간 중앙값12.13→11.64초만 보면 처리량과 짧은 요청 손해가 숨는다.

판정: 실험 서버는 한 ATOM일 때 기존 adaptive refill을 유지하고, **두 ATOM에서는 초기값을 sealed**로 정했다. 두 장의 refill은 policy control로 명시적으로 켜서 실험할 수 있다. 이 기본값 변경은 r3 종료 후 이루어졌으며 r3에서 같은 값을 policy control로 측정했다. 시간차 도착의 두 장 성능까지 검증한 것은 아니다.

최종424요청 전체 logits exact, 요청마다64/64 DMA CQE, 진행 중 같은 카드로 DMA 없음, 모든 generation/slot 완료 확인. 다른 요청이 진행되는 가운데 실제 refill50건을 확인했다. 초기424요청도 별도 보존하여 이번 정상 하드웨어 검증 총848요청이다. 서버·두 worker 정상 종료와 장치 반환 확인. 최종 요청당 DMA 중앙값3.136ms.

새 partial batch의 수집 대기는 한 장 구성에도 적용될 수 있으므로 소수 요청에서는 최대100ms 추가 대기가 생길 수 있다. 이전 한 장 수치를 현재 코드의 모든 부하에 그대로 적용하지 않는다. 기존 trajectory는 이 수집 대기로 멈추지 않는다.

[그래프](iteration-replicas-comparison.png) · [최종 수치·설정·소스 해시](iteration-replicas-validation.json). 원본 `results/pd-iter-two-r3`, 초기 `results/pd-iter-two-r2-initial`, 시작 실패 `results/pd-iter-two-r1-busy`를 보존했다.

다음 후보는 remaining-work 기반 카드 배치, 시간차 도착·다양한 길이에서의 goodput 검증, 더 큰 batch의 메모리 적합성이다. B8 실행 중 ATOM 사용 메모리가약15.0/15.7GiB였으므로 B16은 cache 증가와 scratch 메모리를 확인하는 별도 feasibility 시험부터 필요하다. 목표 전체는 아직 완료되지 않았다.
