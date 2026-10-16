"""Controlled FIFO/early response/length-aware HTTP comparison, rotated order."""
import argparse,concurrent.futures,json,time,urllib.request,hashlib,subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args()
cfg=json.loads(a.config.read_text());out=Path(cfg['out']);prompts=[r['prompt_token_ids'] for r in json.loads(Path(cfg['workload']).read_text())['requests']]
policies={'fifo':dict(policy='fifo',early_response=False),'early':dict(policy='fifo',early_response=True),'length':dict(policy='length',early_response=True)}
def policy(name):
 setting=policies[name];tmp=out/'policy.tmp';tmp.write_text(json.dumps(setting));tmp.replace(out/'policy.json')
 deadline=time.monotonic()+120
 while time.monotonic()<deadline:
  try:
   if json.loads((out/'policy-ack.json').read_text())==setting:return
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(.05)
 raise TimeoutError('policy not acknowledged')
def request(i,n,tag):
 rid=f'{tag}-{i}';body=dict(model='koni-p2p',prompt=prompts[i%4],max_tokens=n,temperature=0,stream=False)
 req=urllib.request.Request('http://127.0.0.1:18823/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','x-request-id':rid,'x-prefiller-host-port':f"127.0.0.1:{cfg['prefill_port']}"})
 start=time.perf_counter()
 with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
 assert len(result['choices'][0]['token_ids'])==n
 return dict(id=rid,input_tokens=len(prompts[i%4]),output_tokens=n,start=start,end=time.perf_counter(),response=result)
rows=[]
def run(name,kind,counts,repeat):
 policy(name);tag=f'{name}-{kind}-r{repeat}';start=time.perf_counter()
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(counts)) as pool:
  futures=[pool.submit(request,i,n,tag) for i,n in enumerate(counts)];answers=[f.result() for f in futures]
 elapsed=time.perf_counter()-start
 row=dict(policy=name,kind=kind,repeat=repeat,seconds=elapsed,tokens_per_second=sum(counts)/elapsed,requests=answers);rows.append(row)
 (out/'http-results.json').write_text(json.dumps(dict(passed=False,in_progress=True,rows=rows),indent=2));print(tag,round(row['tokens_per_second'],3),flush=True)
try:
 run('fifo','warm',[32]*8,-1)
 for repeat in range(3):
  order=['fifo','early','length'];order=order[repeat:]+order[:repeat]
  for name in order:run(name,'mixed32',[32,64,96,128]*8,repeat)
 for name in ['fifo','length']:
  run(name,'fixed128',[128]*16,0);run(name,'partial',[32]*3,0)
 (out/'http-results.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
except BaseException as error:
 (out/'http-results.json').write_text(json.dumps(dict(passed=False,error=repr(error),rows=rows),indent=2));raise
