from pathlib import Path
import json,traceback,numpy as np,time,socket,array,os
from rngd_pd.native_v4.atom_external import ExternalCacheDecoder
from hetero_p2p.atom_experimental import export_live_caches
import sys
p=Path(sys.argv[1]).resolve();root=Path(__file__).resolve().parent.parent
batch=int(sys.argv[2]);device=int(sys.argv[3]);cfg=json.loads(Path(os.environ["HETERO_BATCH_CONFIG"]).read_text())
result={'kind':'real_prompt_p2p_host_parity','batch':batch,'device':device,'input_lengths':[140,167,227,271],'output_lengths':[32,128]}
try:
 artifact=Path(cfg['atom_artifact'])
 engine=ExternalCacheDecoder(artifact,cfg['tables'])
 if engine.batch_size!=batch:raise ValueError('artifact/config batch mismatch')
 engine.step_batch([128000]*batch,[0]*batch)
 import os,ctypes as C,hashlib
 state=json.loads((p/'live-capture.json').read_text());assert state['pid']==os.getpid()
 exports=export_live_caches(engine,engine.cache_device_addresses(),batch*1048576,state,experimental=True)
 mappings,records=exports.mappings,exports.records
 result['mappings']=mappings;result['exports']=records
 import torch,hashlib
 (p/'capture-complete').write_text('disable breakpoints before benchmark')
 C.CDLL(None).getppid();assert (p/'breakpoints-disabled').exists()
 workload=json.loads(Path(cfg['workload']).read_text())
 prompts=[r['prompt_token_ids'] for r in workload['requests']]*(batch//4);positions=[len(x)-1 for x in prompts]
 with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as conn:
  conn.connect(os.path.relpath(p.parent.parent/'bridge.sock'))
  fds=array.array('i',[r['fd'] for r in records])
  conn.sendmsg([json.dumps({'device':device,'batch':batch,'mappings':mappings,'blocks':records}).encode()],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,fds)])
  count=0;reference=None
  while True:
   message=conn.recv(65536)
   if not message:raise RuntimeError('Producer disconnected before stop')
   command=json.loads(message)
   if command['cmd']=='stop':break
   if command['cmd']=='decode_batch':
    assert reference is not None
    jobs=command['requests'];assert 1<=len(jobs)<=batch
    assert len({j['slot'] for j in jobs})==len(jobs)==len({j['id'] for j in jobs})
    for j in jobs:
     assert 0<=j['slot']<batch and 1<=j['output']<=128 and 2<=len(j['tokens']) and len(j['tokens'])+j['output']<=512
    engine.notify_cache_write(range(64))
    generated={j['id']:[] for j in jobs};checks={j['id']:True for j in jobs};diffs={j['id']:0.0 for j in jobs}
    lookup={tuple(prompt):reference[i] for i,prompt in enumerate(prompts)}
    current=[128000]*batch;start=time.perf_counter();first=None
    for j in jobs:current[j['slot']]=j['tokens'][-1]
    for step in range(max(j['output'] for j in jobs)):
     pos=[0]*batch
     for j in jobs:
      if step<j['output']:pos[j['slot']]=len(j['tokens'])-1+step
      else:current[j['slot']]=128000
     out=engine.step_batch(current,pos)
     if first is None:first=time.perf_counter()
     for j in jobs:
      if step>=j['output']:continue
      row=out[j['slot']];rid=j['id'];token=int(row.argmax());generated[rid].append(token);current[j['slot']]=token
      checks[rid]&=bool(np.isfinite(row).all())
      expected=lookup.get(tuple(j['tokens']))
      if expected is not None:
       checks[rid]&=bool(np.array_equal(row,expected[step]));diffs[rid]=max(diffs[rid],float(np.max(abs(row-expected[step]))))
     completed=[j for j in jobs if step+1==j['output']]
     if completed and command.get('early_response',False):
      rows=[dict(request_id=j['id'],tokens=generated[j['id']],passed=checks[j['id']],reference_checked=tuple(j['tokens']) in lookup,max_abs_diff=diffs[j['id']],start=start,first=first,end=time.perf_counter()) for j in completed]
      partial=dict(kind='request_complete',batch_id=command['batch_id'],passed=all(r['passed'] for r in rows),requests=rows)
      conn.send(json.dumps(partial).encode());assert partial['passed'],partial
    end=time.perf_counter()
    rows=[dict(request_id=j['id'],tokens=generated[j['id']],passed=checks[j['id']],reference_checked=tuple(j['tokens']) in lookup,max_abs_diff=diffs[j['id']],start=start,first=first,end=end) for j in jobs]
    reply=dict(batch_id=command['batch_id'],passed=all(checks.values()),requests=rows)
    with (p/'rows.jsonl').open('a') as f:f.write(json.dumps(reply)+'\n')
    conn.send(json.dumps(reply).encode());assert reply['passed'],reply
    count+=1
    continue
   is_reference=command['cmd']=='reference'
   if is_reference:
    assert reference is None
    host=[]
    for file in command['files']:
     with np.load(file) as z:host.append(np.stack((z['keys'].transpose(0,1,3,2),z['values']),axis=1).reshape(64,8,512,128))
    for i,cache in enumerate(engine.caches):cache.copy_(torch.from_numpy(np.stack([h[i] for h in host])))
    torch.rbln.synchronize();del host
   else:
    assert command['cmd']=='decode' and command['completed_cqes']==64*batch and reference is not None
    engine.notify_cache_write(range(64))
   current=[x[-1] for x in prompts];start=time.perf_counter();steps=[];outputs=[];tokens=[]
   for step in range(command['output_tokens']):
    t=time.perf_counter();out=engine.step_batch(current,[v+step for v in positions]);steps.append(time.perf_counter()-t)
    if step==0:first=time.perf_counter()
    current=out.argmax(axis=1).tolist();tokens.append(current);outputs.append(out.copy())
   end=time.perf_counter();actual=np.stack(outputs).transpose(1,0,2)
   if is_reference:
    reference=actual;np.savez(p/'host-reference.npz',logits=actual,tokens=np.array(tokens).T)
   expected=reference[:,:command['output_tokens']];exact=bool(np.array_equal(actual,expected))
   row={'wave_id':command.get('wave_id','host-reference'),'device':device,'batch':batch,'passed':exact,'max_abs_diff':float(np.max(abs(actual-expected))),'start':start,'first':first,'end':end,'decode_seconds':end-start,'step_seconds':steps,'tokens':tokens,'logits_sha256':hashlib.sha256(actual.tobytes()).hexdigest()}
   with (p/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   conn.send(json.dumps(row).encode());assert exact,row
   count+=not is_reference
 exports.close()
 result={'passed':True,'device':device,'batch':batch,'waves':count,'reference':'same batch, per-slot prompt, host admission'}
except Exception as e:result.update(passed=False,error=repr(e),traceback=traceback.format_exc())
(p/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if not result['passed']:raise SystemExit(1)
