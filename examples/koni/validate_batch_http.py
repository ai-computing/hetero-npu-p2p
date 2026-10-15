"""Exercise burst, ragged output, staggered arrivals, and partial B4 requests."""
import argparse,concurrent.futures,json,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--sidecar-port',type=int,default=18823);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=json.loads(a.config.read_text());prompts=[r['prompt_token_ids'] for r in json.loads(Path(cfg['workload']).read_text())['requests']]
def request(index,count,tag,delay):
 time.sleep(delay);start=time.perf_counter();rid=f'{tag}-{index}'
 body=dict(model='koni-p2p',prompt=prompts[index%4],max_tokens=count,temperature=0,stream=False)
 req=urllib.request.Request(f'http://127.0.0.1:{a.sidecar_port}/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','x-request-id':rid,'x-prefiller-host-port':f"127.0.0.1:{cfg['prefill_port']}"})
 with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
 assert len(result['choices'][0]['token_ids'])==count,result
 return dict(id=rid,start=start,end=time.perf_counter(),output_tokens=count,response=result)
scenarios=[('warm',[32]*8,0),('burst32',[32]*16,0),('burst128',[128]*16,0),('ragged',[32,64,96,128]*2,0),('stagger',[32]*8,.05),('partial',[32]*3,0)]
rows=[]
try:
 for tag,counts,spacing in scenarios:
  start=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(counts)) as pool:
   futures=[pool.submit(request,i,n,tag,i*spacing) for i,n in enumerate(counts)]
   requests=[f.result() for f in futures]
  elapsed=time.perf_counter()-start
  row=dict(scenario=tag,seconds=elapsed,tokens_per_second=sum(counts)/elapsed,requests=requests);rows.append(row)
  print(tag,round(row['tokens_per_second'],3),flush=True)
  a.output.write_text(json.dumps(dict(passed=False,in_progress=True,scenarios=rows),indent=2))
 a.output.write_text(json.dumps(dict(passed=True,scenarios=rows),indent=2))
except BaseException as error:
 a.output.write_text(json.dumps(dict(passed=False,error=repr(error),scenarios=rows),indent=2));raise
