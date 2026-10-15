from pathlib import Path
import argparse,json,time,urllib.request,urllib.error
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prefill-port',type=int,default=18721);p.add_argument('--decode-port',type=int,default=18722);p.add_argument('--sidecar-port',type=int,default=18723);a=p.parse_args()
body=dict(model='koni-p2p',prompt=[128000,9906,11,358,1097,264,18328,13,128009],max_tokens=32,temperature=0,stream=False)
def post(port,body,rid,sidecar=False):
 headers={'Content-Type':'application/json','x-request-id':rid}
 if sidecar:headers['x-prefiller-host-port']=f'127.0.0.1:{a.prefill_port}'
 req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps(body).encode(),headers=headers)
 with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)
rows=[]
for i in range(3):
 rid=f'library-sidecar-{i}';t=time.perf_counter();r=post(a.sidecar_port,body,rid,True);assert len(r['choices'][0]['token_ids'])==32
 rows.append(dict(id=rid,route='upstream-llmd-shared-storage',seconds=time.perf_counter()-t,response=r))
rid='library-direct-0';post(a.prefill_port,dict(body,max_tokens=1),rid);r=post(a.decode_port,body,rid);assert r['choices'][0]['token_ids']==rows[0]['response']['choices'][0]['token_ids'];rows.append(dict(id=rid,route='direct-two-stage',response=r))
assert all(r['response']['choices'][0]['token_ids']==rows[0]['response']['choices'][0]['token_ids'] for r in rows)
a.output.write_text(json.dumps({'passed':True,'requests':rows},indent=2)+'\n');print('PASS: 3 upstream sidecar requests + 1 direct request; identical token IDs')
