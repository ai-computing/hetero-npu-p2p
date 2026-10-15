# llm-d 연결

검토한 upstream은 llm-d inference-scheduler/router v0.9.0, commit `5f4e762f341a5196393ce79f8a57c3e1900c4a6b`이다. [공식 저장소](https://github.com/llm-d/llm-d-inference-scheduler)의 shared-storage sidecar는 `x-prefiller-host-port`로 prefill 요청을 보내고, output limit을1로 바꾼다. 성공 후 원래 요청에 `cache_hit_threshold=0`을 넣어 decode로 보낸다. prefill 응답에 담긴 KV descriptor가 자동 전달되는 계약이 아니다.

따라서 두 stage endpoint는 동일한 `StageBroker`와 owner RPC를 공유한다. key는 `x-request-id`이고 모델·prompt fingerprint를 검증한다. output limit은 sidecar가 변경하므로 fingerprint에서 제외하며 decode 시 따로 검증한다. KV는 host shared-memory에 저장하지 않고 destination NPU에 둔다. stage간 공유되는 것은 cache lease와 제어 상태다.

현재 HTTP adapter는 명시적인 token-ID prompt와 greedy/non-streaming completions만 지원한다. 지원하지 않는 필드를 조용히 무시하지 않는다. 새 모델은 vocab/context와 tokenizer 처리를 adapter에 맞게 확장해야 한다. validation fixture는 KONI9토큰·출력32만 받는다.

## 로컬 검증 예제

1. `examples/koni/config.example.json`을 복사하고 로컬 모델/SDK/아티팩트 경로를 채운다. 출력 경로는108바이트 미만의 UNIX socket을 만들 수 있는 짧은 절대 경로로 설정한다.
2. main Python은 Furiosa2026.4 환경을 사용한다. `PYTHONPATH`에 이 프로젝트 `src`와 기존 `rngd_pd`의 `src`를 넣는다. ATOM Python은 별도 설정한다.
3. `python examples/koni/serve.py --config /path/to/config.json`을 실행한다. 예제는 live allocation 관찰용 GDB를 사용하며 kernel driver를 변경하지 않는다.
4. 기존 upstream `pd-sidecar` 실행파일을 다음처럼 실행한다. 검증에서는 수신 주소만 loopback으로 제한한 기존 로컬 빌드를 사용한다. 새 저장소에 upstream 코드/바이너리를 복사하지 않는다.

```bash
/path/to/pd-sidecar --port 18723 --vllm-port 18722 \
  --kv-connector shared-storage --secure-proxy=false
python examples/koni/validate_http.py --output /path/to/validation.json
```

5. `events.jsonl`에서 동일 request_id의 `p2p_complete → prefill_ready → decode_start → decode_complete`를 확인한다. `decode.jsonl`은 모든32단계의 logits가 같은 batch host reference와 일치했는지 기록한다.
6. 검증이 끝나면 설정한 output 디렉터리에 `stop` 파일을 만들고 서버가 정상 종료할 때까지 기다린다. 실행 중인 DMA를 강제 종료하지 않는다.

두 HTTP endpoint는 같은 프로세스의 제어 broker이며 실제 ATOM 런타임은 별도 프로세스다. 이 로컬 검증은 Kubernetes/EPP의 부하분산·멀티노드·장애 복구 시험을 대신하지 않는다. 두 stage를 서로 다른 Pod로 분리하려면 단일 cache owner로의 인증된 RPC와 Unix FD 전달 경로를 추가로 설계해야 한다.
