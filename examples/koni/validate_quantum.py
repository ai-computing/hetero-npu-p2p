"""Rotated fixed/adaptive quantum controls under burst and delayed arrivals."""
import argparse,concurrent.futures,json,time,urllib.request
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['out']);prompts=[r['prompt_token_ids'] for r in json.loads(Path(cfg['workload']).read_text())['requests']]
def request(i,count,tag,target):
 time.sleep(max(0,target-time.perf_counter()))
 rid=f'{tag}-{i}';body=dict(model='koni-p2p',prompt=prompts[i%4],max_tokens=count,temperature=0,stream=False)
 req=urllib.request.Request('http://127.0.0.1:18823/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','x-request-id':rid,'x-prefiller-host-port':f"127.0.0.1:{cfg['prefill_port']}"})
 start=time.perf_counter()
 with urllib.request.urlopen(req,timeout=180) as response:r=json.load(response)
 assert len(r['choices'][0]['token_ids'])==count
 return dict(id=rid,scheduled_start=target,start=start,end=time.perf_counter(),output_tokens=count,response=r)
def switch_policy(name):
 desired=dict(policy='length',refill=True,quantum=4 if name=='fixed4' else 128,partial_quantum=128 if name=='fixed128' else 4)
 ack=out/'policy-ack.json'
 ack.unlink(missing_ok=True)
 temp=out/'policy.tmp';temp.write_text(json.dumps(desired));temp.replace(out/'policy.json')
 deadline=time.monotonic()+180
 while time.monotonic()<deadline:
  try:
   if json.loads(ack.read_text())==desired:return
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(.02)
 raise TimeoutError('idle policy transition')
rows=[]
try:
 scenarios=[('fixed128','warm',[32]*16,[0]*16)]
 policies=['fixed128','fixed4','adaptive']
 for rep in range(3):
  for policy in policies[rep:]+policies[:rep]:
   scenarios.append((policy,f'{policy}-burst128-r{rep}',[128]*16,[0]*16))
   scenarios.append((policy,f'{policy}-staggered-r{rep}',[128]+[32]*15,[0]+[2+.1*i for i in range(15)]))
 for policy,tag,counts,arrivals in scenarios:
  switch_policy(policy)
  start=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(counts)) as pool:
   futures=[pool.submit(request,i,n,tag,start+arrivals[i]) for i,n in enumerate(counts)];answers=[f.result() for f in futures]
  elapsed=time.perf_counter()-start;row=dict(policy=policy,scenario=tag,started=start,scheduled_arrivals=arrivals,seconds=elapsed,tokens_per_second=sum(counts)/elapsed,requests=answers);rows.append(row)
  (out/'http-results.json').write_text(json.dumps(dict(passed=False,in_progress=True,rows=rows),indent=2));print(tag,round(row['tokens_per_second'],3),flush=True)
 (out/'http-results.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
except BaseException as error:
 (out/'http-results.json').write_text(json.dumps(dict(passed=False,error=repr(error),rows=rows),indent=2));raise
