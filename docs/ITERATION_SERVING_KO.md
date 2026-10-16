# Step 경계 슬롯 교체 구현과 검증

## 적용한 논문 원리와 범위

[ORCA (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)의 request 단위 대신 iteration 단위로 실행을 스케줄링하는 원리를 고정 폭 ATOM decoder에 적용했다. ORCA의 selective batching 구현 전체를 이식한 것은 아니다. 모델 graph/가중치/정밀도는 그대로이고, kernel 실행이 끝난 경계에서만 완료 슬롯의 KV와 토큰 상태를 교체한다.

기존 worker는 한 batch의 가장 긴 요청이 끝날 때까지 모든 KV 슬롯을 점유했다. 새 worker는 quantum 또는 가장 빨리 끝나는 요청까지 남은 step 수 중 작은 만큼만 실행한다. 결과 logits가 CPU에 도착하고 `torch.rbln.synchronize()`가 끝난 후 generation과 모든 요청의 진행 상태를 parent에 보낸다. parent는 응답을 검증하고 완료 슬롯만 해제한다. 계속 진행할 요청의 토큰·KV·owner는 유지하며, 빈 슬롯에만 P2P admission을 수행한 후 다음 quantum을 시작한다.

이는 실행 중인 kernel에 DMA를 겹치는 구현이 아니다. 모든 slot을 만지는 기존 kernel 특성 때문에 실제 compute fence와 DMA 완료 확인을 유지한다. 모델별 캐시 형식과 ABI 제한도 그대로다.

## 코드

- `src/hetero_p2p/iteration.py`: generation ticket, 슬롯 임대, quantum 선택, 응답 검증, 실패 시 owner 보존.
- `src/hetero_p2p/iteration_worker.py`: 슬롯별 생성 trajectory, mandatory full-logit reference 검사, adapter synchronize 이후에만 완료 응답. 실험용 NumPy 의존 모듈이며 기본 package import 경로에는 포함하지 않는다.
- `examples/koni/serve_length_worker.py`: 기존 batch 경로를 유지하면서 `decode_iteration` 명령 지원.
- `examples/koni/serve_iteration.py`: RNGD prefill과 P2P를 연결한 실험 HTTP 서버. 최초 검증은 ATOM 한 장으로 제한한다. 현재 별도 example이며 기존 서버 기본 동작은 바꾸지 않는다.
- `validate_iteration.py`, `analyze_iteration.py`: 비교 시험 및 모든 slot/generation/DMA 사건 재생 검사.

잘못된 generation, request set, 진행량, 기존 출력 변경, logits 불일치, 불확실한 device completion은 reject/poison 처리한다. 타이머·조기 HTTP 응답은 device fence를 대신하지 않는다. NumPy fake adapter로 이어가는 요청과 교체 요청의 진행량, 동기화 시점, stale response, DMA 이전 launch 금지를 검증했다. 현재55개 단위시험 통과.

## 최초 실험 설계

실행 ID `pd-iter-r1`, 최초 `/tmp/hetero-pd-iter-r1`. 물리 RNGD3=`furiosa:2`, ATOM2, KONI B8/context512, 입력140/167/227/271, 기존 모델·artifact·전력 설정·driver 유지.

먼저 출력1/8/2/16/3/32/4/64/5/128의10요청, FIFO, quantum16으로 겹치는 trajectory의 슬롯 재사용을 의도적으로 만든다. 전체 logits가 정확히 일치해야 성능 시험으로 넘어간다. 이후 warm16 + 조건별3회×2정책×(16+16+32) = 총410요청이다.

`sealed`는 활성 요청이 모두 끝나야 새 요청을 배치에 넣고, `refill`은 각 완료 경계에서 빈 슬롯을 채운다. 두 정책 모두 같은 새 worker, length 정책, lookahead16, collection20ms, quantum128을 사용한다. sealed→refill / refill→sealed / sealed→refill 순서로 비교한다. 생성은 non-streaming, EOS와 무관한 고정 길이이며 전체 logits 검증 비용을 포함한다.

새 worker에서는 정확히 같은 logits임을 비교한 뒤 별도의 절대차 배열 계산을 생략했다. 새 서버는 활성 decode 재개를 우선하도록 prefill 제어 순서를 바꾸었고, 기존의 부분 배치100ms 대기와 동일한 경로가 아니다. 따라서 이전 서버 수치와의 차이 전체를 슬롯 교체의 효과로 해석하지 않는다. 같은 새 서버 내 sealed/refill 비교가 해당 효과의 기준이다.

공유 호스트 순차 비교이며 실제 도착률 trace·EOS 길이·긴 문맥을 대표하지 않는다. 이번에는 전력/MFU/SLO를 재측정하지 않는다. 목표는 계속 진행 중이며 두 ATOM 확장과 실제 요청 분포 검증은 남아 있다.

## 실측 결과 (2026-10-16)

| 조건 | sealed 중앙값 | refill 중앙값 | 변화 |
|---|---:|---:|---:|
| 출력32,16요청 |81.58 tok/s|81.22 tok/s|−0.44%|
| 출력128,16요청 |94.13 tok/s|93.50 tok/s|−0.67%|
| 혼합32요청 |73.49 tok/s|79.50 tok/s|+8.17%|

혼합의 전체 HTTP 응답시간 중앙값20.52→16.74초(18.44% 감소), 유효 lane 비율76.92→85.71%였다. lane 비율은 MFU가 아니다. 반복 처리량은 sealed73.49/73.83/73.45, refill79.42/79.50/85.99 tok/s로 새 정책에서도 도착 순서·배치 조합에 따른 변동이 남는다. 최대값85.99를 대표값으로 사용하지 않는다.

| 혼합 내 출력 길이 | sealed 응답시간 중앙값 | refill 응답시간 중앙값 |
|---|---:|---:|
| 32 |21.89초|8.95초|
| 64 |14.08초|12.88초|
| 96 |24.41초|16.74초|
| 128 |19.18초|24.62초|

긴 요청128의 지연은 증가한다. 따라서 모든 요청의 지연 개선이나 항상 우월한 정책으로 주장하지 않는다. 고정 길이는 빈 lane 개선 여지가 없어 성능 이득도 없었다. 새 서버는 별도 실험 example으로 유지하며 기존 length 서버 기본값은 변경하지 않았다.

410요청 모두 같은 B8 host-admitted 기준의 전체 logits exact, 각 요청64/64 DMA CQE 완료, execution 중 DMA 없음, 종료 후 살아 있는 slot 없음. 진행 중인 다른 요청을 유지한 실제 refill70건을 사건 재생으로 확인했다. worker가 정상 종료했고 장치를 반환했다. DMA 중앙값2.959ms.

내부 worker 타임스탬프로 계산한 **요청별 평균 TPOT**의 p99는 혼합 sealed81.55ms, refill81.93ms였다. 이는 non-streaming HTTP 클라이언트가 받은 inter-token 지연이나 모든 token 간격의 p99가 아니다. 공개 trace의 SLO 달성 시험으로 대체할 수 없다.

원본: `results/pd-iter-r1` (기존 `/tmp`에서 이동), [분석 JSON](iteration-validation.json). config, source SHA256와 소스 snapshot, patch, worker/HTTP/event 로그, host reference를 보존했다. raw 파일은 git 제외이며 요약·코드는 git 관리한다.

재분석: `python3 examples/koni/analyze_iteration.py results/pd-iter-r1 --output /tmp/iteration-analysis.json`.
서버: `PYTHONPATH=src:../rngd_llmd/src /home/gspark/venvs/furiosa-2026.4/bin/python examples/koni/serve_iteration.py --config artifacts/koni-pd-iter-local.json` (config의 out은 존재하지 않는 새 디렉터리로 변경).
클라이언트: `python3 examples/koni/validate_iteration.py --config artifacts/koni-pd-iter-local.json`, sidecar는 기존18823→18822 shared-storage control 구성.

다음은 staggered arrival에서 quantum4/16/128 및 빈 슬롯에 따른 adaptive quantum을 비교하고, 두 ATOM으로 확장하는 단계다. 긴 요청의 지연·p99를 함께 보며, 검증을 생략해서 처리량을 올리지 않는다.

[처리량·길이별 지연 비교 그래프](iteration-comparison.png). 재생성: `python3 examples/koni/plot_iteration.py docs/iteration-validation.json docs/iteration-comparison.png`.
