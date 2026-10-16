"""Independent FP32 numerical checks; not application accuracy or non-inferiority."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
ap=argparse.ArgumentParser();ap.add_argument('--b16-root',type=Path,required=True);ap.add_argument('--b8-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
w=json.loads(Path('experiments/native_pd_2026_4/real_workload_20261015/workload.json').read_text());torch.set_num_threads(16)
m=AutoModelForCausalLM.from_pretrained(w['model'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
for _ in range(300):
 if (a.b16_root/'measure-result.json').exists():break
 time.sleep(1)
else:raise TimeoutError('B16 diagnostic results not ready')
base=np.load(a.b8_root/'b8/atom2/host-reference.npz')['logits'][:4]
free=np.load(a.b16_root/'logits-0.npz')['actual'][:4]
forced=np.load(a.b16_root/'logits-1.npz')['actual'][:4]
rows=[]
def evaluate(name,lane,actual,gold,scope):
 actual=torch.from_numpy(actual);chosen=actual.argmax(-1);best=gold.argmax(-1)
 regret=gold.max(-1).values-gold[torch.arange(len(chosen)),chosen]
 row=dict(backend=name,lane=lane,scope=scope,count=len(chosen),top1_match=int((chosen==best).sum()),max_logit_regret=float(regret.max()),mean_logit_regret=float(regret.mean()),mean_logit_abs_error=float((actual-gold).abs().mean()),mean_kl_gold_to_backend=float((gold.softmax(-1)*(gold.log_softmax(-1)-actual.log_softmax(-1))).sum(-1).mean()))
 rows.append(row);(a.out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
with torch.inference_mode():
 for lane,job in enumerate(w['requests']):
  prompt=job['prompt_token_ids'];tokens=base[lane].argmax(-1).tolist();gold=m(torch.tensor([prompt+tokens[:-1]]),use_cache=False).logits[0,len(prompt)-1:].float()
  evaluate('B8',lane,base[lane],gold,'common B8 trajectory')
  evaluate('B16',lane,forced[lane],gold,'common B8 trajectory')
  tokens=free[lane].argmax(-1).tolist();gold=m(torch.tensor([prompt+tokens[:-1]]),use_cache=False).logits[0,len(prompt)-1:].float()
  evaluate('B16',lane,free[lane],gold,'B16 own trajectory')
(a.out/'result.json').write_text(json.dumps(dict(reference='HF FP32 eager CPU, fixed four prompts, 128 positions; not task accuracy',model=w['model'],rows=rows),indent=2))
