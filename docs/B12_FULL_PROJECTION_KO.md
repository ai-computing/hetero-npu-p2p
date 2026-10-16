# B12 attention 출력 투영 통합 시험

기존8개 group 부분곱과 순차합산을 하나의 출력투영으로 바꿨다. KONI/context512/B12, 같은SDK·입력·가중치cast. Runtime graph와 반올림순서는 달라진다.

| 구간 | 기존 grouped | full projection | 판단 |
|---|---:|---:|---|
| eager decode-only 중앙값 |138.394tok/s|144.151tok/s|+4.16%, 별도순차실행 |
| 전체 PD48요청 출력128 중앙값 |127.249tok/s|126.833tok/s|-0.327%, 개선미확인 |

새 PD3회119.727/126.833/132.645, 기존127.249/127.077/127.447tok/s. 최대132.645를 대표값으로 선택하지 않는다. 첫실행의 실제배치도12×4로 가득차 있었다. 첫반복 worker구간4배치가11.697/12.494/11.821/11.847초로, 기존11.400/11.424/11.416/11.387초보다 느렸다. 원인은미확정이며 sharedhost변동·워밍업가능성을분리해야한다.

224요청 same-newgraph host/P2P logits exact, 소유권위반0, 실제lane100%, 전체이벤트재생통과, 서버·worker·sidecar정상종료. 기존graph와의 strictbitwise 실패는 그대로보존한다. FP32 자체이력top1 510/512, 최대regret0.03691673이며4입력의 수치진단일뿐 응용품질우위가아니다. 기존506/512는 서로다른이력이다.

**기본경로 채택보류.** 다음은 측정과같은48요청/128출력 워밍업 후, 실행순서를교차한 grouped/full 비교와 반복별구간계측이다. 기존실험은삭제하거나 새워밍업기준으로소급제외하지않는다.

[요약](b12-full-projection-validation.json) · [전체PD비교](b12-projection-pd-comparison.json). Raw results/pd-b12-full-sat48-r1 및 experiments/native_pd_2026_4/b12_full_projection_20261016.
