"""Counterbalanced iteration slot-refill ablation with hardware overlap probe."""
import argparse,concurrent.futures,json,time,urllib.request,subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--requests',type=int,default=16,choices=(16,32,48,84));ap.add_argument('--saturation',action='store_true');ap.add_argument('--mixed-only',action='store_true');ap.add_argument('--warm-output',type=int,choices=(32,128),default=32);ap.add_argument('--device-snapshots',action='store_true');a=ap.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['out']);prompts=[r['prompt_token_ids'] for r in json.loads(Path(cfg['workload']).read_text())['requests']]
if a.saturation and a.mixed_only:raise ValueError('choose saturation or mixed-only')
if not (a.saturation or a.mixed_only) and a.requests not in (16,32):raise ValueError('legacy design requires16 or32 requests')
if a.requests==84 and cfg.get('queue_capacity',64)<84:raise ValueError('84-request design requires explicit queue_capacity>=84')
def request(i,count,tag):
 rid=f'{tag}-{i}';body=dict(model='koni-p2p',prompt=prompts[i%4],max_tokens=count,temperature=0,stream=False)
 req=urllib.request.Request('http://127.0.0.1:18823/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','x-request-id':rid,'x-prefiller-host-port':f"127.0.0.1:{cfg['prefill_port']}"})
 start=time.perf_counter()
 with urllib.request.urlopen(req,timeout=180) as response:r=json.load(response)
 assert len(r['choices'][0]['token_ids'])==count
 return dict(id=rid,start=start,end=time.perf_counter(),output_tokens=count,response=r)
def switch_policy(name):
 desired=dict(policy='fifo' if name=='probe' else 'length',refill=name!='sealed',quantum=16 if name=='probe' else 128)
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
 scenarios=[('probe','probe',([1,8,2,16,3,32,4,64,5,128] if cfg.get('batch_size',8)<=8 else list(range(1,17))+[32,64,96,128]*4)),('sealed','warm',[a.warm_output]*a.requests)]
 if a.saturation:
  if a.requests not in (48,84):raise ValueError('saturation design requires48 or84 requests')
  scenarios += [('sealed',f'sealed-fixed128-r{rep}',[128]*a.requests) for rep in range(3)]
 for rep in range(0 if a.saturation else 3):
  policies=['sealed','refill'] if rep%2==0 else ['refill','sealed']
  for policy in policies:
   for case,counts in ([('mixed',[32,64,96,128]*(a.requests//4))] if a.mixed_only else [('fixed32',[32]*a.requests),('fixed128',[128]*a.requests),('mixed32',[32,64,96,128]*8)]):
    scenarios.append((policy,f'{policy}-{case}-r{rep}',counts))
 for policy,tag,counts in scenarios:
  switch_policy(policy)
  if a.device_snapshots:
   (out/f'{tag}-device-before.txt').write_text(subprocess.check_output(['rbln-smi'],text=True))
  start=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(counts)) as pool:
   futures=[pool.submit(request,i,n,tag) for i,n in enumerate(counts)];answers=[f.result() for f in futures]
  elapsed=time.perf_counter()-start;row=dict(policy=policy,scenario=tag,seconds=elapsed,tokens_per_second=sum(counts)/elapsed,requests=answers);rows.append(row)
  if a.device_snapshots:
   (out/f'{tag}-device-after.txt').write_text(subprocess.check_output(['rbln-smi'],text=True))
  (out/'http-results.json').write_text(json.dumps(dict(passed=False,in_progress=True,rows=rows),indent=2));print(tag,round(row['tokens_per_second'],3),flush=True)
 (out/'http-results.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
except BaseException as error:
 (out/'http-results.json').write_text(json.dumps(dict(passed=False,error=repr(error),rows=rows),indent=2));raise
