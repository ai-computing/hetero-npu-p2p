"""CPU-only compiler diagnostic; isolated subprocess per candidate shape."""
import argparse,json,os,subprocess,sys,time,resource
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--context',type=int);a=ap.parse_args()
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
if a.context is not None:
 import torch,rebel
 from rngd_pd.native_v4.build_resident import register_attention
 register_attention();torch.set_num_threads(8);S=a.context;B=16
 class Attention(torch.nn.Module):
  def forward(self,q,k,v,mask,kcache,vcache,seq,blocks):
   return torch.ops.rbln_custom_ops.paged_attn_decode(q,k,v,mask,kcache,vcache,seq,torch.tensor(128**-.5),blocks,S)
 inputs=[torch.zeros(B,8,4,1,128),torch.zeros(B,8,1,1,128),torch.zeros(B,8,1,1,128),torch.ones(B,1,1,1,S),torch.zeros(B,8,1,S,128),torch.zeros(B,8,1,S,128),torch.zeros(B,1,dtype=torch.int32),torch.arange(B,dtype=torch.int16).reshape(B,1)]
 compiled=rebel.compile_from_torch(Attention().eval(),example_inputs=inputs,npu='RBLN-CA22')
 compiled.save(str(a.out/'attention.rbln'))
else:
 a.out.mkdir(parents=True,exist_ok=False);rows=[]
 for context in (400,416,448,512):
  p=a.out/f's{context}';p.mkdir();start=time.monotonic()
  with (p/'compile.log').open('w') as log:
   rc=subprocess.call([sys.executable,__file__,'--out',str(p),'--context',str(context)],stdout=log,stderr=subprocess.STDOUT)
  row=dict(context=context,returncode=rc,compiled=rc==0 and (p/'attention.rbln').exists(),seconds=time.monotonic()-start,runtime_created=False,dma_started=False)
  rows.append(row);(a.out/'result.json').write_text(json.dumps(dict(scope='single attention compile, not full model support or runtime correctness',rows=rows),indent=2));print(json.dumps(row),flush=True)
