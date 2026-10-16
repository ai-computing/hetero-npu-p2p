# 1 RNGD + 1 ATOM PD 실측 (2026-10-16)

KONI B8, RNGD prefill→P2P KV→ATOM decode, llm-d HTTP 요청 전체 구간을 측정했다. 한 장 결과를 두 장 수치에서 환산하지 않고 실제 ATOM2만 사용했다.

| 부하 | 1 RNGD+1 ATOM 중앙값 (3회 범위) | 기존 1 RNGD+2 ATOM 중앙값 | 두 장/한 장 |
|---|---:|---:|---:|
| 16요청, 출력32 |80.00 (79.04–80.36) tok/s|134.83 tok/s|1.69배|
| 16요청, 출력128 |91.27 (90.51–91.31) tok/s|176.50 tok/s|1.93배|
| 32요청, 출력32/64/96/128 혼합 |76.62 (72.13–77.21) tok/s|127.47 tok/s|1.66배|

한 장의 HTTP 응답시간 중앙값은 각각5.04/17.05/21.06초다. 요청 대기시간을 포함하며 TPOT가 아니다. 단회 진단은 단일 요청 출력32에서9.32 tok/s, 3요청 출력32에서33.74 tok/s였다. 위80–91 tok/s는 배치를 채운 다중 요청의 합산 처리량이다.

## 조건과 검증

- 모델 revision `0172bf7e73977ae213613529f3d2d5400a33f678`, 입력140/167/227/271토큰, context512, greedy, non-streaming, EOS와 관계없이 고정 출력 길이.
- 물리 RNGD3=`furiosa:2`, ATOM2 한 장. 기존 두 장 결과는 같은 RNGD와 ATOM2/3. B8 artifact `c244ef77d4e90da8.rbln`.
- lookahead16, collection20ms, partial batch wait100ms, HTTP backlog128. RNGD SDK2026.4.0, rebel0.11.0, ATOM driver3.0.0.
- 3회 반복의 중앙값과 전 범위를 보고했다. 기존 두 장 결과와 순차 비교이며 공유 호스트의 외부 부하·실행 순서를 통제한 교차 실험은 아니다.
- 212요청 모두 host-admitted 기준과 전체 logits 정확히 일치, DMA64/64 CQE 완료, 실행 중 replica로 KV 덮어쓰기 없음. 모든 batch 최종 완료 확인 후 정상 종료했다.
- 요청당 DMA 구간 중앙값2.944ms. 이는 helper 호출·계획 생성·대기 등을 모두 합친 end-to-end KV 전달 시간과 다르다.
- 전체 logits CPU 전송과 검증 비용이 포함된다. 전력·MFU·TPOT p99 및 SLO는 이번에 측정하지 않았다. 단독 NPU 대비 새 B8 동일조건 우위도 이 실험만으로 확정하지 않는다.
- 라이브러리 단위시험45개 통과. example의 ATOM replica 수를 설정의 `[2]`, `[3]`, `[2,3]`으로 선택할 수 있게 변경했다. 허용 카드 범위는 유지했고, 사전 idle 검사는 보수적으로2/3 모두 검사한다.

## 개선 우선순위

1. **혼합 길이 배치 구성**: 한 장의 유효 lane 비율81.08%, 두 장76.92%로 낭비가 남는다. 출력 길이별 bucket, lookahead, 대기시간을 바꾸면서 throughput과 짧은 요청 지연시간을 함께 비교한다. 이 비율은 MFU가 아니다. 실제 EOS 길이는 max_tokens와 다를 수 있다.
2. **decode와 배치 사이 비용**: 같은 ATOM의 이전 B8 decode-only 측정은103.54 tok/s, 이번 긴 출력 전체 PD는91.27 tok/s다. 조건·시점·포함 비용이 다른 참고 비교이므로12% 개선을 보장하는 상한으로 쓰지 않는다. batch 사이 admission/helper 호출과 CPU logits 처리 시간을 분리 계측한 뒤 최적화한다. 정확성 검증을 없애서 수치만 올리지 않고 검증 모드와 서비스 모드의 비용을 분리 보고한다.
3. **저부하 배치 정책**: B8은 소수 요청에서 빈 lane 비용이 크다. B4/B8 시작 설정을 부하에 맞춰 선택하고 TTFT/TPOT를 검증한다. 가중치 두 벌의 동시 상주나 무조건적인 동적 전환은 현재 구현하지 않았다.
4. **안전한 continuous batching**: 현재 컴파일된 kernel은 모든 slot을 만지므로 실행 중 KV 덮어쓰기는 금지한다. step 경계의 명시적 fence와 slot별 cache 갱신 지원·메모리 여유를 먼저 확인해야 한다. 임의 DMA로 우회하지 않는다.
5. **서비스 검증 확대**: 같은 B8의 단독 NPU 기준, 실제 요청 도착률·긴 문맥, 전력·MFU·SLO 재측정과 취소/TTL 처리 보강이 필요하다.

현재 증거는 DMA 전송속도 하나보다 decode 배치 활용과 관리 비용을 우선 볼 근거가 된다. 다만 모든 end-to-end 비용을 분리한 프로파일이 없어 각 최적화의 이득을 확정하지 않는다.

## 재현 자료

- [비교 그래프](pd-replica-throughput.png)
- [원시 수치·설정·소스 해시](pd-replica-validation.json)
- 로컬 `results/pd-one-r1`: HTTP 결과, events, reference parity, config, source manifest와 patch, 실행 명령, 장치 매핑 및 정상 종료 로그. 최초 `/tmp/hetero-pd-one-r1`에서 종료 후 이동했다.
- 비교 원본 `results/pd-b8-r4`, [두 장 시험 설명](PD_B8_RESULTS_KO.md).
- 재분석: `python3 examples/koni/analyze_pd_scaling.py results/pd-one-r1 --output /tmp/one-analysis.json`. 동일 B8 두 run을 이 분석기에 한 번에 넣으면 batch 키가 겹치므로 각각 분석하여 replica별로 비교한다.
- 그래프 재생성: `python3 examples/koni/plot_pd_replicas.py docs/pd-replica-validation.json docs/pd-replica-throughput.png`.
