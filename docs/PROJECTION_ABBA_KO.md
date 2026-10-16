# 출력 투영 통합: 동일 워밍업 ABBA 검증

KONI/context512/B12, 1 RNGD+1 ATOM2,48요청×출력128, sealed, 각실행에서동일48/128워밍업 후3회측정. full→grouped→grouped→full 순차실행한공유호스트 시험이다.

| 순서 | 그래프 | 중앙값tok/s | 3회범위 |
|---|---|---:|---:|
|1|full|132.070|131.897–132.495|
|2|grouped|126.377|126.250–126.423|
|3|grouped|126.621|126.336–126.791|
|4|full|131.390|131.228–131.555|

비교쌍의full/grouped 변화는+4.50%,+3.77%. 실행별중앙값의요약은grouped126.499→full131.730tok/s(+4.14%). 프로세스블록은그래프당2개이며반복을독립표본처럼해석하거나통계적유의성을주장하지않는다.

전896요청 same-artifact host/P2P logits exact, 소유권위반0, 실제lane100%, 모든DMA완료. 전서버/worker/sidecar정상종료. 이전strict구그래프bitwise실패는보존한다. FP32자체이력510/512는4입력수치검사이며응용품질비열등성을증명하지않는다.

## 적용 판단

고정긴출력·포화부하의다음실험 기준으로full projection을채택한다. compile_b16.py의HETERO_COMPILE_BATCH=12/HETERO_FULL_OUTPUT_PROJECTION=1로재현한다. 기본컴파일옵션은그대로두고명시적인실험선택으로관리한다. 새graph의혼합부하성능·저부하·전력·SLO는추가검증대상이다.

기존짧은워밍업full3회119.727/126.833/132.645와중앙값-0.33%결과도유지한다. 이번시험은긴워밍업후이득이재현됐음을보이지만이전변동의원인을워밍업하나로단정하지않는다.

[원시분석](projection-abba-validation.json). Raw results/proj-abba-1-full, proj-abba-2-grouped, proj-abba-3-grouped, proj-abba-4-full. 모든설정·소스·장치전후snapshot보존. Snapshot은연속전력계측이아니다. 명령: ../../experiments/2026-10-16-projection-abba.md.
