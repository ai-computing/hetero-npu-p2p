"""Runtime fit probe with the same eager allocation as DMA export; no DMA."""
import argparse,json,os,time,traceback,subprocess
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--artifact',required=True)
ap.add_argument('--batch',type=int,choices=(8,12,14,16),required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--context',type=int,choices=(400,416,448,512),default=512)
a=ap.parse_args()
if os.environ.get('RBLN_DEVICES')!='2' or os.environ.get('TORCH_RBLN_EAGER_MALLOC')!='1':
 raise RuntimeError('explicit authorized ATOM2 and eager allocation required')
p=a.out.resolve();p.mkdir(parents=True,exist_ok=False)
from batch_control import assert_atom_idle
assert_atom_idle(p)
result=dict(batch=a.batch,artifact=a.artifact,passed=False,dma_started=False,eager=True,context=a.context)
try:
 import torch,numpy as np
 from hetero_p2p.adapters.atom_koni_external import ExternalCacheDecoder
 torch.set_num_threads(8)
 begin=time.perf_counter()
 engine=ExternalCacheDecoder(a.artifact,'experiments/npu_pd_extended_20261014/pd_atom/tables.npz',experimental_batch12=a.batch==12,experimental_batch14=a.batch==14,experimental_batch16=a.batch==16,context=a.context)
 if engine.batch_size!=a.batch:raise ValueError('artifact batch mismatch')
 rows=[]
 for step in range(2):
  logits=engine.step_batch([128000]*a.batch,[step]*a.batch)
  torch.rbln.synchronize()
  rows.append(dict(shape=list(logits.shape),finite=bool(np.isfinite(logits).all())))
 (p/'rbln-smi.txt').write_text(subprocess.check_output(['rbln-smi'],text=True))
 result.update(passed=all(r['finite'] for r in rows),rows=rows,seconds=time.perf_counter()-begin)
except Exception as error:
 result.update(error=repr(error),traceback=traceback.format_exc())
finally:
 (p/'result.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result),flush=True)
if not result['passed']:raise SystemExit(1)
