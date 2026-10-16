"""HF FP32 check on each backend's own token history; not task accuracy.

Run after throughput experiments to avoid shared CPU/memory interference.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
cfg=json.loads((a.run/'config.json').read_text())
if not json.loads((a.run/'result.json').read_text())['passed']:raise RuntimeError('completed PD run required')
w=json.loads(Path(cfg['workload']).read_text());batch=cfg['batch_size'];device=cfg['atom_devices'][0]
a.out.mkdir(parents=True,exist_ok=False)
with np.load(a.run/f'b{batch}/atom{device}/host-reference.npz') as data:actual=data['logits'][:4].copy()
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
(a.out/'result.json').write_text(json.dumps(dict(scope='HF FP32 eager CPU on backend own history; four prompts, not task accuracy/non-inferiority',batch=batch,model=cfg['model'],run=str(a.run),rows=rows,top1_match=sum(r['top1_match'] for r in rows),count=sum(r['count'] for r in rows)),indent=2))
