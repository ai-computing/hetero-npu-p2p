# 서로 다른 한국어 질의의 PD / 단독 비교

동일 모델 revision, 입력140/167/227/271토큰의 실제 chat-template 질의 4개를 각4회 처리한다. 각 wave16요청, 출력32/128, warmup1+측정3회. PD는 RNGD1+ATOM2/3(B4), RNGD 단독은 Offline+profile-guided+concurrency8, ATOM 단독은 제조사 Optimum 1장 B4 / chunk64 prefill이다.

| 방식 | 출력 길이 | tokens/s | tokens/J | 평균 보드 W |
|---|---:|---:|---:|---:|
| PD | 32 | 89.654 | 0.5527 | 162.2 |
| PD | 128 | 98.285 | 0.6070 | 161.4 |
| RNGD solo | 32 | 61.565 | 0.4023 | 153.0 |
| RNGD solo | 128 | 66.284 | 0.4326 | 152.3 |
| ATOM vendor solo | 32 | 40.154 | 0.6255 | 64.2 |
| ATOM vendor solo | 128 | 52.255 | 0.8713 | 59.6 |

PD는3장, 각 단독은1장으로 자원 예산이 다르다. 보드 에너지만 적분했고 CPU/DRAM은 제외했다. 고정4질의 반복이며 실제 서비스 요청 분포 전체를 대표하지 않는다. 모든 출력은 지정 길이까지 생성하므로 EOS 이후 토큰도 포함할 수 있다.

동일 batch host/P2P logits exact 검증과 FP32 teacher-forced 모델 비교를 분리한다. 전송 exact만으로 모델 품질 동등성을 주장하지 않는다. quality.json 및 실패 기록을 함께 확인해야 한다.

첫 r61은 초기화 중 SIGSEGV로 실패했으며 DMA 전에 정리했다. r62는 BoundedPrefixPacker를 사용한다.
