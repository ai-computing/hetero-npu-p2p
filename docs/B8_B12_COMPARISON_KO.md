# B8 → B12 동일32요청 비교

1 RNGD + 1 ATOM2, KONI/context512, 입력140/167/227/271. 같은lookahead32/collection20ms/initialwait100ms, non-streaming·EOS무시·전체logits검증. B12 이후 B8을 순차실행했으며 batch 간 교차순서는 통제하지 않았다. 각조건3회 중앙값.

| 조건 | 정책 | B8 tok/s | B12 tok/s | 변화 |
|---|---|---:|---:|---:|
| fixed32 | sealed | 81.47 | 93.11 | +14.28% |
| fixed128 | sealed | 94.55 | 109.93 | +16.27% |
| mixed32 | sealed | 91.36 | 82.64 | -9.55% |
| fixed32 | refill | 81.74 | 93.07 | +13.86% |
| fixed128 | refill | 94.30 | 110.17 | +16.82% |
| mixed32 | refill | 90.75 | 100.13 | +10.34% |

B8 618요청/B12 640요청 전부 같은배치 host 기준 logits exact 및 사건재생 통과. 측정요청은 각576개로 동일하고 probe 수만 다르다. 두 서버와 sidecar 정상종료.

B12는 refill에서 개선했으나 혼합sealed는악화했다. 자동기본값변경은보류한다. B12 32요청은12+12+8 tail을 포함한다. 메모리여유가작고 독립 FP32 수치 검사는 아래에 기록했다. 전력/MFU/SLO나 일반서비스품질의동등성을입증하지않는다.

[반복별 원시 요약](b8-b12-comparison.json). Raw results/pd-b8-control32-r1 및 results/pd-b12-r1.

## 독립 FP32 수치 검사 완료

B12 자체 생성 이력에서 HF FP32 eager CPU와 다음토큰506/512개 일치(4입력×128위치). 최대 logit regret0.17534256. 같은이력에서의 수치진단이며 응용정확도·품질비열등성 증거는 아니다. 다른batch의 서로다른이력 top1을 직접 품질순위로 비교하지 않는다. [결과](b12-fp32-quality.json). 원본 experiments/native_pd_2026_4/b12_feasibility_20261016/hf-quality-own.
