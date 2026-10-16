"""Host-admitted B8/B16 decoder feasibility, never exports/imports DMA buffers."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch,torch_rbln
from hetero_p2p.adapters.atom_koni_external import ExternalCacheDecoder
ap=argparse.ArgumentParser();ap.add_argument('--artifact',required=True);ap.add_argument('--reference-root',required=True);ap.add_argument('--reference-batch',type=int,choices=[8,12,16],default=8);ap.add_argument('--profile',action='store_true');ap.add_argument('--batch',type=int,choices=[8,12,16],required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();N=a.batch;p=a.out.resolve();p.mkdir(parents=True,exist_ok=False);root=Path(a.reference_root);result={'kind':'host admission only','batch':N,'passed':False}
import os,subprocess
if os.environ.get('RBLN_DEVICES')!='2':raise RuntimeError('authorized physical ATOM2 required')
from batch_control import assert_atom_idle
assert_atom_idle(p)
try:
 torch.set_num_threads(8);e=ExternalCacheDecoder(a.artifact,'experiments/npu_pd_extended_20261014/pd_atom/tables.npz',experimental_batch16=N==16,experimental_batch12=N==12,profile_steps=a.profile);assert e.batch_size==N
 workload=json.loads(Path('experiments/native_pd_2026_4/real_workload_20261015/workload.json').read_text());prompts=[r['prompt_token_ids'] for r in workload['requests']]*(N//4)
 host=[]
 for i in range(4):
  with np.load(root/f'b{a.reference_batch}/host-kv-{i}.npz') as z:host.append(np.stack((z['keys'].transpose(0,1,3,2),z['values']),axis=1).reshape(64,8,512,128))
 ref=np.load(root/f'b{a.reference_batch}/atom2/host-reference.npz')['logits'];expected=ref[np.arange(N)%ref.shape[0]];rows=[]
 for repeat in range(4):
  for i,cache in enumerate(e.caches):cache.copy_(torch.from_numpy(np.stack([host[j%4][i] for j in range(N)])))
  torch.rbln.synchronize();current=[x[-1] for x in prompts];times=[];actual=[]
  for step in range(128):
   begin=time.perf_counter();out=e.step_batch(current,[len(x)-1+step for x in prompts]);times.append(time.perf_counter()-begin);actual.append(out.copy());current=out.argmax(axis=1).tolist()
  actual=np.stack(actual,axis=1);row=dict(repeat=repeat,warmup=repeat==0,seconds=sum(times),tokens_per_second=(128*N)/sum(times),median_step_ms=float(np.median(times)*1000),finite=bool(np.isfinite(actual).all()),exact_to_reference=bool(np.array_equal(actual,expected)),top1_matches=int(np.sum(actual.argmax(-1)==expected.argmax(-1))),top1_total=128*N,max_abs_diff=float(np.max(abs(actual-expected))));rows.append(row);(p/'measure-progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True);(p/f'device-repeat{repeat}.txt').write_text(subprocess.check_output(['rbln-smi'],text=True))
 result.update(reference_batch=a.reference_batch,profile_scope='host wall intervals; async work may finish in output materialization',step_profile=list(e.step_profile),passed=all(r['finite'] and r['exact_to_reference'] for r in rows),rows=rows,parity_passed=all(r['exact_to_reference'] for r in rows),artifact=a.artifact)
except Exception as error:result.update(error=repr(error),traceback=traceback.format_exc());raise
finally:(p/'measure-result.json').write_text(json.dumps(result,indent=2))

if not result["passed"]:raise SystemExit(1)
