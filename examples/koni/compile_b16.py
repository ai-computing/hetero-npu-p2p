"""KONI graph-mode external KV feasibility; full decoder diagnostic, not full PD."""
import json,time,traceback,os
if os.environ.get("RBLN_DEVICES")!="2":raise RuntimeError("This feasibility run requires authorized physical ATOM2")
class CompileOnlyDone(BaseException):pass
N=int(os.environ.get('HETERO_COMPILE_BATCH','16'))
if N not in (8,12,14,16):raise ValueError('feasibility compiler supports B8/B12/B14/B16')
S=int(os.environ.get('HETERO_COMPILE_CONTEXT','512'))
if S not in (400,416,448,512):raise ValueError('feasibility context must be400,416,448 or512')
projection=os.environ.get('HETERO_FULL_OUTPUT_PROJECTION','0')
if projection not in ('0','1'):raise ValueError('HETERO_FULL_OUTPUT_PROJECTION must be0 or1')
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch,torch_rbln,rebel
from safetensors import safe_open
from rngd_pd.native_v4.build_resident import ResidentDecode
p=Path(os.environ['HETERO_BUILD_DIR']).resolve();p.mkdir(parents=True,exist_ok=False)
from batch_control import assert_atom_idle
assert_atom_idle(p)
modeldir=Path('/home/gspark/.cache/huggingface/hub/models--KISTI-KONI--KONI-Llama3.1-8B-Instruct-20241024/snapshots/0172bf7e73977ae213613529f3d2d5400a33f678')
result={'kind':'koni_external_full_decoder','layers':32,'batch':N,'context':S,'full_output_projection':projection=='1','revision':modeldir.name}
start=time.time()
try:
 torch.set_num_threads(8)
 config=json.loads((modeldir/'config.json').read_text())
 cfg=SimpleNamespace(hidden=config['hidden_size'],inter=config['intermediate_size'],n_heads=config['num_attention_heads'],n_kv=config['num_key_value_heads'],head_dim=config['hidden_size']//config['num_attention_heads'],gqa=config['num_attention_heads']//config['num_key_value_heads'],eps=config['rms_norm_eps'])
 sd={}
 for shard in sorted(modeldir.glob('model-*.safetensors')):
  with safe_open(shard,framework='pt') as f:
   for key in f.keys():
    if key.startswith('model.layers.') or key in ['model.norm.weight','lm_head.weight']:
     sd[key]=f.get_tensor(key).float().numpy()
 model=ResidentDecode(cfg,sd,S,32,N,wdt=torch.bfloat16).eval();model.full_output_projection=projection=='1';del sd
 model=model.to('rbln:0')
 print('WEIGHTS_READY',flush=True)
 def move(x):return x.to('rbln:0')
 caches=[move(torch.zeros(N,8,S,128)) for _ in range(64)]
 original_create=rebel.RBLNCompiledModel.create_runtime
 def stop_after_save(self,*args,**kwargs):
  if list((p/'compiled').glob('*.rbln')):raise CompileOnlyDone('COMPILE_ONLY_ARTIFACT_SAVED')
  return original_create(self,*args,**kwargs)
 rebel.RBLNCompiledModel.create_runtime=stop_after_save
 holder=[]
 run=torch.compile(model,backend='rbln',dynamic=False,options={'cache_dir':str(p/'compiled'),'_runtime_holder':holder})
 with np.load('experiments/npu_pd_extended_20261014/pd_atom/tables.npz') as z:
  embeddings=z['embed'].astype(np.float32);inv=z['inv'].copy()
 rows=[];saved_logits=[]
 with torch.no_grad():
  for pos,tok in [(0,128000),(1,9906)]:
   angles=pos*inv
   mask=torch.zeros(N,S);mask[:,:pos+1]=1
   args=[move(torch.from_numpy(np.repeat(embeddings[tok:tok+1],N,axis=0))),move(torch.from_numpy(np.repeat(np.cos(angles).astype(np.float32)[None,:],N,axis=0))),move(torch.from_numpy(np.repeat(np.sin(angles).astype(np.float32)[None,:],N,axis=0))),move(mask),move(torch.full((N,1),pos,dtype=torch.int32)),move(torch.arange(N,dtype=torch.int16).reshape(N,1)),*caches]
   t=time.perf_counter();out=run(*args);cpu=out.cpu().float()
   saved_logits.append(cpu.numpy())
   rows.append({'pos':pos,'argmax':int(cpu.argmax()),'finite':bool(torch.isfinite(cpu).all()),'shape':list(cpu.shape),'seconds':time.perf_counter()-t})
 np.savez(p/'logits.npz',*saved_logits)
 result.update(rows=rows,runtime_count=len(holder),cache_addresses=[rebel._C.vmem.get_device_addrs(t.data_ptr()) for t in caches],passed=all(r['finite'] for r in rows))
except CompileOnlyDone:
 artifacts=list((p/'compiled').glob('*.rbln'))
 result.update(passed=bool(artifacts),compiled_only=True,artifacts=[str(x) for x in artifacts],runtime_validated=False)
except Exception as e:
 artifacts=list((p/'compiled').glob('*.rbln'))
 compiled=bool(artifacts) and 'COMPILE_ONLY_ARTIFACT_SAVED' in str(e)
 result.update(passed=compiled,compiled_only=compiled,artifacts=[str(x) for x in artifacts],error=None if compiled else repr(e),traceback=None if compiled else traceback.format_exc())
result['seconds']=time.time()-start
(p/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if not result['passed']:raise SystemExit(1)
