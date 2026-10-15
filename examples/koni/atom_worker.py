"""Experimental export owner; use capture.gdb and configured model adapter."""
from pathlib import Path
import os,json,sys,socket,array,ctypes,time,traceback
import numpy as np
from hetero_p2p.atom_experimental import export_live_caches
from rngd_pd.native_v4.atom_external import ExternalCacheDecoder
config=json.loads(Path(sys.argv[1]).read_text());out=Path(config['out']);conn=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET)
exports=None;result={'passed':False}
try:
 engine=ExternalCacheDecoder(config['atom_artifact'],config['tables'])
 engine.step(128000,0)
 capture=json.loads((out/'live-capture.json').read_text())
 exports=export_live_caches(engine,engine.cache_device_addresses(),1048576,capture,experimental=True)
 (out/'capture-complete').write_text('disable before inference')
 ctypes.CDLL(None).getppid();assert (out/'breakpoints-disabled').exists()
 with np.load(config['reference']) as z:reference=z['logits'][0]
 conn.connect(config['socket'])
 conn.sendmsg([json.dumps({'pid':os.getpid(),'mappings':exports.mappings,'blocks':[dict(size=r['size'],dva=r['dva']) for r in exports.records]}).encode()],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[r['fd'] for r in exports.records]))])
 requests=0
 while True:
  command=json.loads(conn.recv(65536))
  if command['cmd']=='stop':break
  assert command['cmd']=='decode' and command['tokens']==[128000,9906,11,358,1097,264,18328,13,128009] and command['output']==32
  engine.notify_cache_write(range(64));current=command['tokens'][-1];tokens=[];logits=[];start=time.perf_counter()
  for pos in range(8,40):
   value=engine.step(current,pos);current=int(value.argmax());tokens.append(current);logits.append(value.reshape(-1).copy())
  actual=np.stack(logits);passed=bool(np.array_equal(actual,reference));row=dict(request_id=command['id'],passed=passed,tokens=tokens,max_abs_diff=float(np.max(abs(actual-reference))),start=start,end=time.perf_counter())
  with (out/'decode.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  conn.send(json.dumps(row).encode());assert passed;requests+=1
 result.update(passed=True,requests=requests)
except Exception as e:result.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 if exports:exports.close()
 conn.close();(out/'worker-result.json').write_text(json.dumps(result,indent=2)+'\n')
