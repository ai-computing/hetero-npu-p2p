# hetero-npu-p2p

같은 Linux 호스트의 **RNGD → ATOM KV 전송**을 모델 실행 코드와 분리한 실험용 라이브러리입니다. 전송 계획, 버퍼 소유권, 완료 확인, 물리 KV layout, llm-d stage 연동을 제공합니다. 임의 주소 탐색이나 드라이버 패치를 하지 않습니다.

**모든 환경에서 동작하는 완성형 런타임은 아닙니다.** 현재 검증 범위는 Furiosa 2026.4.0 / ATOM driver 3.0.0 및 고정 thunk ABI입니다. 모르는 ABI는 거부합니다. 같은 ABI의 다른 장치 경로는 설정 가능하지만 추가 하드웨어 검증이 필요합니다.

## 현재 성능 판단

동일9토큰 입력·32토큰 출력의 별도 비교에서 RNGD 네이티브 단독은61.30 tok/s·0.417 tok/J, 커스텀 이종 PD는27.84 tok/s·0.183 tok/J였습니다. 현재 조건에서 PD의 성능·전력효율 우위는 없습니다. 런타임 간 greedy 출력도 달라 품질 동등성을 주장하지 않습니다. 이 프로젝트는 검증된 데이터 이동 경로를 재사용하기 위한 것이며, PD를 항상 더 빠르게 만드는 기능은 아닙니다.

## 구조

- `core.py`: source/destination owner 보존, 범위·정렬·FD alias 검사, admission/decode 상태 전이. 실패한 버퍼는 자동 재사용하지 않습니다.
- `backend.py`, `native/rngd_pdma.c`: RNGD PDMA의 completion-fenced scatter 쓰기. 호스트로 KV payload를 복사하거나 조용히 host-copy로 대체하지 않습니다.
- `layout.py`: 명시적인 token-major 물리 KV 배치로 prefix 전송 계획 생성. dtype 변환은 모델 adapter가 담당합니다.
- `atom_experimental.py`: 동일 프로세스의 live allocation trace와 검증된 thunk ABI를 사용하는 export adapter. `experimental=True`가 필요하며 정적 resident cache는 지원하지 않습니다.
- `llmd.py`: llm-d shared-storage stage 계약용 broker. 요청 ID·모델·prompt fingerprint, 용량, TTL, 순서, 실패 격리를 검사합니다. HTTP에는 요청/응답과 제어 정보만 오가며 KV는 P2P로 이동합니다.
- `examples/koni/`: 기존 KONI 모델 adapter를 연결한 실제 장치 검증 예제. 고정9토큰 prompt / 출력32 / B1에 제한됩니다. 일반 production serving API로 표시하지 않습니다.

## 설치와 검사

```bash
python -m pip install -e .
python native/build.py --liburing-root /path/to/liburing-2.6
hetero-p2p doctor --device /dev/rngd/npu1ch0
PYTHONPATH=src python -m unittest discover -s tests -v
python tests/native_rejection.py
```

빌드에는 GCC, Furiosa2026.4.0 드라이버 개발 헤더, 빌드된 liburing이 필요합니다. Python core는 표준 라이브러리만 사용합니다. wheel에는 proprietary SDK, 드라이버 헤더, 모델 가중치, C 실행파일을 포함하지 않습니다. C helper는 검증된 로컬 드라이버 헤더로 별도 빌드합니다.

`doctor`는 버전/ABI/장치 파일 호환성 검사입니다. 장치가 비었거나 실제 PCIe P2P가 성공함을 보증하는 시험은 아닙니다.

## 최소 사용 형태

```python
from hetero_p2p import (SourceAllocation, DestinationAllocation, CopySpan,
                        TransferPlan, CopySession, RngdPdmaBackend)

# raw_dram_offset, dma_buf_fd는 살아 있는 SDK 소유 객체에서 얻어야 합니다.
source = SourceAllocation(raw_dram_offset, source_bytes, source_owner,
                          "dlfloat16-token-major-v1", "/dev/rngd/npu1ch0")
destination = DestinationAllocation(dma_buf_fd, destination_owner,
                                     "dlfloat16-token-major-v1")
plan = TransferPlan(source, (destination,),
                    (CopySpan(0, 0, 0, 4096),), descriptors_per_command=1)
session = CopySession(RngdPdmaBackend("build/rngd-pdma"))
session.admit(request_id, plan, notify_destination_cache_updated)
session.start_decode(request_id)
# 실제 decode를 끝내고 정확도/상태를 확인한 뒤:
session.finish_decode(request_id, passed=True)
```

주소와 크기는 예제 값으로 추측해서 넣지 마세요. Source는 raw RNGD DRAM offset이며 CPU 주소가 아닙니다. destination은 ATOM이 소유한 exportable eager allocation의 DMA-BUF입니다. source와 모든 destination owner는 완료와 decode까지 유지해야 하며, FD를 수동으로 닫거나 소유 객체를 해제하면 안 됩니다. `CopySession`은 자신의 참조를 보존하지만 외부 코드의 잘못된 수동 해제를 방지할 수는 없습니다.

## llm-d 연결

[llm-d 연동 문서](docs/LLMD.md), [지원 범위](docs/SUPPORT.md), [검증 기록](docs/VALIDATION.md)을 참고하세요. 이 구현은 llm-d를 KV 복사 경로로 사용하지 않습니다. llm-d shared-storage 계약으로 prefill→decode를 호출하고, 같은 노드의 owner 프로세스가 DMA와 cache lease를 관리합니다.

이 프로젝트는 NIXL/Mooncake connector나 원격 노드 RDMA 구현이 아닙니다. Kubernetes 배포, 토큰 streaming, chat template, 자동 모델 컴파일, 일반 prompt 처리는 현재 검증 예제의 범위 밖입니다.

Git 저장소는 로컬에 생성했습니다. 원격 저장소 생성·push·공개 배포는 하지 않았으며 배포 라이선스는 별도 결정이 필요합니다.
