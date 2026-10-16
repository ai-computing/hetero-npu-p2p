# B8 이종 PD 통합 및 라이브러리 개선 (2026-10-16)

## 결과

B8을 실제 RNGD→ATOM P2P 및 llm-d HTTP 요청 경로에 연결했다. 같은 하드웨어3장, 같은 요청 묶음, 같은 lookahead16과 부분 배치 대기100ms에서 B4/B8을 각각3회 측정했다.

| 조건 | B4 PD 중앙값 | B8 PD 중앙값 | B8/B4 |
|---|---:|---:|---:|
| 16요청, 출력32 |85.66 tok/s|134.83 tok/s|1.57배|
| 16요청, 출력128 |100.52 tok/s|176.50 tok/s|1.76배|
| 32요청, 출력32/64/96/128 혼합 |93.10 tok/s|127.47 tok/s|1.37배|

HTTP 응답시간 중앙값도 각각4.75→3.77초,15.41→11.59초,14.52→11.84초로 줄었다. 혼합 길이는 B4 85.23–100.27, B8 113.03–132.02 tok/s의 반복 변동이 있어 중앙값과 범위를 함께 보아야 한다. 단회 부분 배치3요청 점검은 B4 **35.80**, B8 **33.78 tok/s**로 B8이 느렸다. 배치가 채워질 때의 이득을 저부하에도 일반화하지 않는다.

이 배율은 **동일 카드 수의 PD B4 대비 PD B8 개선**이다. 새 B8 단독 NPU 기준을 재측정한 비교가 아니며, 전력·MFU·SLO 달성률은 이번 시험에서 평가하지 않았다.

## 조건과 검증

- KONI-Llama3.1-8B-Instruct-20241024, revision `0172bf7e73977ae213613529f3d2d5400a33f678`, context512.
- 물리 RNGD3 = SDK `furiosa:2`에서 prefill, ATOM2/3에서 decode. RNGD SDK2026.4.0, rebel0.11.0, ATOM driver3.0.0.
- 입력140/167/227/271토큰. 출력 길이 고정, greedy non-streaming, EOS 종료 없이 지정 길이까지 생성.
- B4 artifact `6463347b9a988e0e.rbln`, B8 artifact `c244ef77d4e90da8.rbln`.
- 동일 llm-d shared-storage control sidecar, lookahead16, collection quiet window20ms, 부분 배치 dispatch wait100ms, HTTP listen backlog128. 꽉 찬 배치는 즉시 dispatch한다.
- B4 먼저, B8 나중에 실행했다. 공유 호스트에서의 순차 비교이며 실행 순서 무작위화나 독점 호스트 실험은 아니다.
- 최종 비교 **424요청(각212)** 모두 같은 배치의 host-admitted 기준과 **전체 logits 정확히 일치**. 초기 B8 탐색212요청도 별도 통과하여 이번 작업의 정상 PD 요청 총636개를 보존했다.
- 모든 요청64/64 DMA CQE 완료, decode 중인 replica로 DMA 없음, 최종 batch 완료 전 버퍼 재사용 없음 확인. 요청당 DMA 중앙값은 B4 3.024ms, B8 3.042ms.
- 라이브러리 단위시험45개 통과. producer·ATOM worker 정상 종료와 해당 NPU 자원 반환 확인. 다른 작업의 프로세스·드라이버는 변경하지 않았다.
- 타이밍에는 전체 logits를 CPU로 가져와 비교하는 검증 비용이 포함된다. 실제 서비스의 EOS 길이와 요청 분포에서 동일 수치를 보장하지 않는다.

## 라이브러리 변경

1. `BatchOptions`: 배치 크기·lookahead·수집 대기 설정을 검증한다. KONI example은 B4/B8 artifact shape를 확인하고 슬롯 주소 계산·스케줄링·worker에 같은 폭을 적용한다.
2. `CompletionLedger`: 조기 응답과 최종 완료를 분리한다. 중복·오래된 세대·다른 요청·길이 오류·변경된 최종 출력은 거부하며, 조기 응답만으로 버퍼를 해제하지 않는다.
3. `RngdBinding`: 물리 DMA 노드와 SDK 논리 번호를 분리하고, 작은 실제 할당이 연 채널 파일로 대응을 검증한다. 검증된 물리 경로를 source allocation과 backend에 함께 적용한다.
4. 부분 배치가 아직 시작되지 않았다면 빈 슬롯을 추가로 채울 수 있도록 개선했다. HTTP listen backlog도 확대했다. `serve_batch_worker.py`는 공통 worker를 사용해 중복 구현을 제거했다.

## 보존한 실패와 변동

- `pd-b8-r1-rejected`: 물리 노드2가 없어 사전 검사에서 종료. 실제 장치는0,1,3이고 SDK 논리 번호는0,1,2였다. DMA 없음.
- `pd-b8-r2-busy`: SMI에 나타나지 않은 컨테이너가 물리0을 사용 중이었다. 런타임 EBUSY가 OutOfMemory로 감싸져 실패했다. DMA 전에 종료했고 이후 실제 compute acquisition을 ATOM worker 시작 이전으로 옮겼다.
- `pd-b8-r3-initial`: 정상 출력212요청이지만16개 요청이8+7+1로 나뉜 반복에서 출력128 처리량이95.98 tok/s로 떨어졌다. 버퍼 오류가 아니라 마지막1개가 다음 decode까지 기다린 문제였다. HTTP backlog, 부분 배치 refill, quiet window, dispatch wait를 함께 보강했다. 개별 변경의 독립적 기여를 분리한 실험은 아니다.
- 보강 후 최종 B8의 출력128은173.73/176.50/176.72 tok/s, 유효 batch lane 비율100%였다. 혼합 길이는 여전히76.92%로 낭비가 남는다. 이 비율은 MFU나 하드웨어 utilization이 아니다.

## 남은 개선 후보

저부하에서는 B8의 긴 step과 빈 슬롯 비용이 불리하다. 배치 크기는 현재 시작 시 선택하며 카드 위에 두 artifact를 동시에 상주시켜 동적으로 전환하지 않는다. 혼합 길이의 실행 순서·replica 부하 균형, 단독 B8과 다중 ATOM 기준 재측정, 전력 재측정이 다음 검증 대상이다. 라이브러리의 model-independent 계약과 별개로 모델 KV 변환·driver ABI·실험 서버의 취소/TTL 처리는 여전히 제한되어 있다.

## 재현 자료

- [결과 그래프](pd-b8-throughput.png)
- [원시 수치·설정·소스 SHA256](pd-b8-validation.json)
- [라이브러리 API·실행 명령](SERVING_LIBRARY.md)
- `results/pd-b4-r1`, `results/pd-b8-r4`: 최종 원시 HTTP 응답, 사건 로그, 실행 로그, config, source manifest, compute binding, 기준 logits (로컬, git 제외).
- 초기·실패 run도 `results/pd-b8-r1-rejected`, `results/pd-b8-r2-busy`, `results/pd-b8-r3-initial`에 보존했다. 로그의 `/tmp/...`는 최초 실행 경로이며 종료 후 저장소로 이동했다.
