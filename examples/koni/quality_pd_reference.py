"""HF FP32 check on each backend's own token history; not task accuracy.

Run after throughput experiments to avoid shared CPU/memory interference.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--measurement-config',type=Path);a=ap.parse_args()
if a.measurement_config:
 cfg=json.loads(a.measurement_config.read_text())
 measurement=json.loads((a.run/'measure-result.json').read_text())
 rows=measurement.get('rows',[])
 if len(rows)!=4 or {r['repeat'] for r in rows}!={0,1,2,3} or not all(r['finite'] for r in rows):
  raise RuntimeError('completed four-repeat finite measurement required')
 batch=measurement['batch'];reference=a.run/'own-trajectory.npz'
 origin=dict(kind='decode-only numerical diagnosis',artifact=measurement['artifact'],strict_reference_parity=measurement['parity_passed'],transport_validated=False)
else:
 cfg=json.loads((a.run/'config.json').read_text())
 if not json.loads((a.run/'result.json').read_text())['passed']:raise RuntimeError('completed PD run required')
 batch=cfg['batch_size'];device=cfg['atom_devices'][0];reference=a.run/f'b{batch}/atom{device}/host-reference.npz'
 origin=dict(kind='completed PD own reference',artifact=cfg['atom_artifact'])
w=json.loads(Path(cfg['workload']).read_text())
a.out.mkdir(parents=True,exist_ok=False)
with np.load(reference) as data:actual=data['logits'][:4].copy()
if len(w['requests'])!=4 or actual.shape[:2]!=(4,128):raise ValueError('four prompts and128 reference steps required')
torch.set_num_threads(16)
model=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
rows=[]
with torch.inference_mode():
 for lane,job in enumerate(w['requests']):
  logits=torch.from_numpy(actual[lane]);tokens=logits.argmax(-1);prompt=job['prompt_token_ids']
  gold=model(torch.tensor([prompt+tokens[:-1].tolist()]),use_cache=False).logits[0,len(prompt)-1:].float()
  regret=gold.max(-1).values-gold[torch.arange(len(tokens)),tokens]
  row=dict(lane=lane,count=len(tokens),top1_match=int((tokens==gold.argmax(-1)).sum()),max_logit_regret=float(regret.max()),mean_logit_regret=float(regret.mean()),mean_logit_abs_error=float((logits-gold).abs().mean()))
  rows.append(row);(a.out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
(a.out/'result.json').write_text(json.dumps(dict(scope='HF FP32 eager CPU on backend own history; four prompts, not task accuracy/non-inferiority',batch=batch,origin=origin,model=cfg['model'],run=str(a.run),rows=rows,top1_match=sum(r['top1_match'] for r in rows),count=sum(r['count'] for r in rows)),indent=2))
