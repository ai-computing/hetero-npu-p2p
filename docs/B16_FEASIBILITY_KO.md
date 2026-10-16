# ATOM B16 실행 가능성 및 PD 연결 준비

2026-10-16. KONI revision 0172bf7e73977ae213613529f3d2d5400a33f678, ATOM2, rebel0.11.0, context512, 입력140/167/227/271을 반복한 고정128토큰 생성이다. 기존 44c3b4d 이후 작업 트리에서 측정했다. 원본은 experiments/native_pd_2026_4/b16_feasibility_20261016/ 아래 보존한다.

| decode-only | 3회 중앙값 tok/s | 반복 범위 | step 중앙값 |
|---|---:|---:|---:|
| B8 |104.7132|104.6195–104.7575|76.036ms|
| B16 |177.8052|176.9601–177.8661|89.791ms|

B16 컴파일155.69초 후 실제 실행 가능함을 확인했다. 관측 메모리15.1/15.7GiB. KV host admission은 타이밍 밖이며, prefill/P2P/HTTP는 포함하지 않는다. 따라서 1.698배는 전체 PD 성능 개선치가 아니다. 전력 snapshot은 에너지 측정이 아니며 MFU도 측정하지 않았다.

## 정확성 결과와 실패 보존

B8 기준과 bitwise logits 일치를 요구한 B16 시험은 실패했다. 자유 생성 토큰 일치는1152/2048, 같은 B8 이력으로 teacher forcing하면2032/2048이었다. 반복된4개 입력이므로 독립된2048개 표본이 아니다.

HF FP32 CPU 기준으로 같은 B8 이력의512개 위치를 비교하면 B8 top1은505개, B16은507개 일치했다. B16 자체 생성 이력에서는506개였다. 이는 제한된 수치 진단이며 응용 품질 동등성이나 B8 대비 우수성을 증명하지 않는다. 최초 strict failure는 유지한다. batch별 kernel 산술 차이는 가능한 원인이지만 원인으로 확정하지 않았다.

## PD 구현 상태

serve_iteration.py에 experimental_batch16=true인 경우만 B16을 허용했다. worker는 B16에서 라이브러리의 atom_koni_external 어댑터를 사용하며 B4/B8 경로는 유지한다. 같은 B16 host reference를 새로 생성해 P2P 결과와 비교하므로 B8 logits를 전송 정합성 기준으로 사용하지 않는다. 소유권, 장치 동기화, 완료 검증은 유지했다.

이 변경은 단위시험61개 통과 및 구문 검사 단계다. **B16 P2P 하드웨어 검증과 전체 HTTP 측정은 아직 수행하지 않았다.** 기본 배치를 변경하지 않는다. 다음은 같은 B16 host/P2P exact, 서로 다른 출력 길이의 슬롯 재사용 검증 후 동일32요청 B8/B16 비교다. 네 입력의 반복으로 숨을 수 있는 슬롯 aliasing은 추가로 검증해야 한다.

[원시 수치 요약](b16-feasibility-validation.json)

## B16 PD 초기화 실패 (02:19 UTC)

실행 pd-b16-r1에서 eager KV cache 할당 도중 RuntimeError로 종료했다. worker 등록 전이며 export/P2P DMA는 시작하지 않았다. parent/worker 모두 종료 및 장치 반환을 확인했다. 실패 원본은 results/pd-b16-r1-allocation-failed에 보존했다(종료된 Unix socket 제외).

GDB live-capture에 성공한 flags0x22,32MiB 할당52개가 기록됐다. 별도 runtime 큰 할당은15,011,414,016 bytes다. 누적 할당 기록은15.7207GiB지만 free 이력을 추적하지 않으므로 peak/상주 메모리로 해석하지 않는다. eager에서 float32 논리 KV 한 개의 요구량은16×8×512×128×4=32MiB,64개 합계2GiB다. 단독 lazy 실행의 B16 fit은 eager export 경로의 적합성을 보장하지 않는다.

다음 후보는 context512를 유지한 B12다. eager KV 총1.5GiB이며 메모리 적합성·출력·P2P는 검증 전이다. 임의 주소/flags 우회는 하지 않는다.
