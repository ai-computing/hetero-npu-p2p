# 검증 기록 — 2026-10-15

## 실제 하드웨어 + llm-d

새 `hetero_p2p` 라이브러리와 새로 빌드한 native helper로 수행했다. 기존 benchmark helper를 그대로 호출한 결과가 아니다.

- RNGD1 PE0 → ATOM3 eager KV, KONI 8B B1, 입력9·출력32.
- upstream llm-d shared-storage sidecar 경유3요청, 직접 prefill/decode 호출1요청.
- 요청마다2MiB prefix payload,64개 DMA 명령 모두 CQE0 완료.
- 이후32개 decode step의 full logits가 같은 batch host 전송 기준과 정확히 일치. 최대 절대 오차0.
- request_id별 P2P 완료 → prefill ready → decode 시작 → decode 완료 순서를 로그에서 검사.
- prefill 없는 decode는409, streaming/미지원 sampling 옵션은400으로 거부.
- stop 파일을 통한 정상 종료, ATOM owner의4요청 성공 및 종료를 확인.

검증된 llm-d 경로는 기존 로컬 loopback 빌드(v0.9.0, upstream commit `5f4e762f341a5196393ce79f8a57c3e1900c4a6b`)이다. Kubernetes/EPP/원격 노드 검증 완료를 의미하지 않는다.

`hardware-validation.json`에 소스 hash·helper 빌드 명령·완료 검사 결과를 기록했다. 원시 logs/JSON은 로컬 `results/hardware-r1/`에 있으며 대용량·환경별 출력은 Git에서 제외한다. Git에는 요약과 재현 코드를 포함한다.

## 장치 없는 검사

- Python22개: 범위/정렬/형식/FD alias, lease, 부분완료, notify 실패, stale completion, TTL, fingerprint, capacity, sidecar output-limit rewrite.
- native rejection11개: malformed plan은 장치를 열기 전에 exit2로 거부. CPU memfd 사용; 이 검사를 실제 DMA 시험으로 집계하지 않음.
- C는 `-Wall -Wextra -Werror -O2`로 빌드.
- Python wheel 빌드 성공. Core는 vendor SDK를 import하지 않음.

## 비교 실험과 구분

[단독/독립 서빙/PD 비교](benchmark/RESULTS_KO.md)는 이 라이브러리 분리 전에 동일 모델·길이 조건으로 수행한 별도 성능 시험이다. 초기 B1 짧은 입력에서는 RNGD 네이티브 단독이 커스텀 PD보다 빨랐으며 전력효율도 높았다. 이후 B4 및 긴 입력 결과는 [새 비교 기록](benchmark-b4/RESULTS_KO.md)에 별도로 보존했다. B4 PD는 처리량이 높았지만 ATOM 단독보다 에너지 효율은 낮았다. 라이브러리 검증4요청의 HTTP 시간으로 benchmark 수치를 대체하지 않는다.

같은 batch host/P2P full-logit 일치는 **전송 경로의 정확도** 증거다. RNGD NativeLLMEngine·ATOM 단독·커스텀 PD 사이의 모델 출력 품질 동등성을 입증하지 않는다.

## 최적화된 B4 라이브러리 경로

라이브러리 exporter/layout/PDMA backend와 BoundedPrefixPacker를 사용하는 batch_pipeline.py로 입력140/167/227/271, 출력32/128의128요청을 완료했다. 모든 full logits가 해당 batch host 기준과 exact, 동일 버퍼 admission/decode 겹침0이었다. 초기 실패(config 변수 shadowing)는 DMA 전에 중단되어 별도 보존했다. 결과와 소스 hash는 batch-hardware-validation.json, 원시는 results/batch-r2에 있다. HTTP llm-d 예제의 B4 batching 검증은 아니다.
