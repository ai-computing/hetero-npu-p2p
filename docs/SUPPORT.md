# 지원 범위와 확장 계약

| 계층 | 현재 구현 | 제한 |
|---|---|---|
| 플랫폼 | Linux x86_64, 같은 호스트 | 원격 DMA/Windows 미지원 |
| source | RNGD1 PE0, 2026.4.0 | 다른 source 경로는 설정 가능하나 실제 장치 검증 필요 |
| destination | ATOM eager exportable allocation, driver3.0.0, 검증 thunk SHA256 | compiled resident allocation, flags 변조 미지원 |
| 방향 | RNGD가 ATOM-export DMA-BUF에 기록 | RNGD-export → ATOM-import 역방향 금지; 기존 실패 경로를 재시도하지 않음 |
| planner | 같은 physical format, 4KiB scatter, 64-byte source base alignment | source≤64MiB, ≤512 descriptors, command당≤8 descriptors |
| 변환 | 모델 adapter가 수행 | FP16/BF16/DLFloat16을 같은 형식으로 취급하지 않음 |
| 동기화 | 모든 CQE 완료 후 notify/decode | 미완료·오류는 poisoned, 자동 재활용 없음 |
| 실행 | owning process의 main thread | HTTP thread는 RPC를 통해 owner에 작업 요청 |
| llm-d | local shared-storage prefill→decode 제어 계약 | 분산 클러스터 배포/전체 serving 기능과 구분 |

## 소유권 계약

1. source를 생성한 SDK 객체가 live 상태이고 모든 source 연산이 synchronize된 뒤만 전송한다.
2. destination export FD 및 SDK 객체를 유지한다. live allocation 주소를 재시작 후 재사용하지 않는다.
3. 같은 destination cache에 여러 CopySession을 만들어 동시에 쓰지 않는다. 하나의 owner가 슬롯별 session을 독점한다.
4. transfer 중 SIGINT/SIGTERM은 main thread에서 지연하며 native helper 완료를 기다린다. SIGKILL, 프로세스/장치 강제 재설정, 외부 코드의 수동 해제에 대한 복구 보장은 없다.
5. `notify` 성공 이후 decode를 시작한다. 별도 ATOM owner 프로세스라면 완료 확인 후 명시적인 decode 명령으로 notify를 전달한다.
6. failed/poisoned 상태는 운영자가 원인을 확인하기 전 재사용하지 않는다. 강제 timeout으로 DMA를 잘라낸 뒤 버퍼를 돌려주는 기능은 제공하지 않는다.

## 모델 독립성과 예제 의존성

core/planner/backend는 KONI 가중치나 layer 수를 내장하지 않는다. `TokenMajorLayout`에 실제 cache 수·head 수·context·head dimension을 제공한다. 다른 layout은 별도 planner/adapter를 작성해야 한다.

KONI 예제는 기존 `rngd_pd` 모델 구현과 NumPy/PyTorch/safetensors/Transformers 및 두 vendor의 분리된 Python 환경을 사용한다. source 변환은 custom grouped prefill graph이며 Furiosa NativeLLMEngine에서 직접 얻은 KV라고 표시하지 않는다. 예제의 모델 경로, artifact, tables, reference, Python 실행파일, 포트, ATOM 카드는 JSON 설정이다. 현재 예제의 RNGD source는1번으로 검증 범위를 고정한다.

새 SDK/장치 지원은 버전 문자열만 바꾸지 말고 allocation/export ABI, physical layout, 비정렬 source, 모든 CQE, 다음 attention/full-logit 정확도, 정상 종료까지 순서대로 검증한 뒤 추가한다. CPU memfd 테스트 통과를 실제 DMA 성공으로 간주하지 않는다.
