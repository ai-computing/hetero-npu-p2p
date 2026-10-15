"""End-to-end hardware validation example, not a general production LLM server.

All physical DMA runs on the main owning thread. HTTP only queues control RPCs.
"""
from pathlib import Path
import argparse,os,json,socket,array,subprocess,time,queue,concurrent.futures,threading,traceback
from types import SimpleNamespace
from hetero_p2p import SourceAllocation,DestinationAllocation,CopySession,RngdPdmaBackend
from hetero_p2p.layout import TokenMajorLayout
from hetero_p2p.llmd import StageBroker,StageError,serve_stage
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False);cfg['out']=str(out.resolve());cfg['socket']=str(out.resolve()/'owner.sock')
# UNIX sun_path is bounded; fail before any model allocation.
if len(cfg['socket'].encode())>=108:raise ValueError('Choose a shorter output path for UNIX socket')
(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');commands=queue.Queue();running=True;servers=[];fds=[];worker=None;result={'passed':False}
prompt=[128000,9906,11,358,1097,264,18328,13,128009]
def event(row):
 with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def rpc(kind,*args):
 future=concurrent.futures.Future();commands.put((kind,args,future));return future.result()
def prepare(rid,tokens):
 if tokens!=prompt:raise StageError(400,'hardware validation fixture supports only its recorded 9-token prompt')
 return rpc('prepare',rid,tokens)
def decode(handle,tokens,count):
 if count!=32:raise StageError(400,'hardware validation fixture requires output32')
 return rpc('decode',handle,tokens,count)
try:
 if cfg.get('rngd_device_path','/dev/rngd/npu1ch0')!='/dev/rngd/npu1ch0':raise ValueError('this KONI adapter only validates RNGD1 PE0')
 atom=str(cfg['atom_device'])
 if Path(f'/sys/class/rebellions/rbln{atom}/group_id').read_text().strip()!='0':raise RuntimeError('ATOM must be in default RSD group')
 smi=json.loads(subprocess.check_output(['rbln-smi','--json'],text=True))
 selected=next(d for d in smi['devices'] if d['device']==f'rbln{atom}')
 if int(selected['memory']['used'])!=0:raise RuntimeError('ATOM card is already in use')
 if 'npu1' in subprocess.check_output(['furiosa-smi','ps'],text=True):raise RuntimeError('RNGD1 is already in use')
 backend=RngdPdmaBackend(cfg['helper'],cfg.get('rngd_device_path','/dev/rngd/npu1ch0'));session=CopySession(backend)
 listener=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET);listener.bind(cfg['socket']);listener.listen(1);listener.settimeout(1)
 env=dict(os.environ,RBLN_DEVICES=str(cfg['atom_device']),RBLN_NPUS_PER_DEVICE='1',HETERO_CAPTURE_DIR=str(out.resolve()),TORCH_RBLN_EAGER_MALLOC='1');env.pop('RBLN_DEVICE_MAP',None);env.pop('RBLN_DUMMY_DEVICE',None)
 log=(out/'worker.log').open('w');example=Path(__file__).resolve().parent
 worker=subprocess.Popen(['gdb','-q','-batch','-x',str(example/'capture.gdb'),'--args',cfg['atom_python'],str(example/'atom_worker.py'),str(out/'config.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
 deadline=time.monotonic()+300
 while True:
  if worker.poll() is not None:raise RuntimeError('ATOM owner initialization failed; inspect worker.log')
  if time.monotonic()>deadline:raise RuntimeError('ATOM owner did not register before DMA')
  try:conn,_=listener.accept();break
  except socket.timeout:pass
 data,anc,flags,_=conn.recvmsg(65536,socket.CMSG_SPACE(64*4));assert not flags&(socket.MSG_TRUNC|socket.MSG_CTRUNC)
 received=array.array('i')
 for level,kind,content in anc:
  if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:received.frombytes(content)
 fds=list(received);metadata=json.loads(data);assert len(fds)==len(metadata['blocks']) and len(metadata['mappings'])==64
 for fd,block in zip(fds,metadata['blocks']):assert os.fstat(fd).st_size==block['size']
 (out/'exports.json').write_text(json.dumps(metadata,indent=2))
 destinations=tuple(DestinationAllocation(fd,conn,'dlfloat16-token-major-v1') for fd in fds)
 import torch,numpy as np
 from safetensors import safe_open
 from rngd_pd.engines.model import ModelConfig
 from rngd_pd.engines.prefill_full import FullPrefillEngine
 from rngd_pd.engines.fsdk import sync
 from rngd_pd.native_v4.rngd_payload import PackPrefixKVPayload
 import furiosa.torch as ft
 from furiosa.torch._C.tensor import copy_from_tensor
 from furiosa.torch._C.ir import FlatDramShape
 class KVOnly(FullPrefillEngine):
  def _build_lmhead(self,cache_dir):pass
  def _next_token(self,x,P):return None
 torch.set_num_threads(8);modeldir=Path(cfg['model']);model_cfg=ModelConfig(SimpleNamespace(**json.loads((modeldir/'config.json').read_text())));sd={}
 for shard in sorted(modeldir.glob('model-*.safetensors')):
  with safe_open(shard,framework='pt') as f:
   for key in f.keys():sd[key]=f.get_tensor(key).float().numpy()
 engine=KVOnly(model_cfg,sd,C=16,group=2,chip=1,cache_dir=None,S=512)
 pack=ft.CompileModule.from_module(PackPrefixKVPayload(8),(torch.zeros((32,8,128,512),dtype=torch.bfloat16),torch.zeros((32,8,512,128),dtype=torch.bfloat16)));pack.to('furiosa:1')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
 layout=TokenMajorLayout(64,8,512,128)
 broker=StageBroker('koni-p2p',prepare,decode,lambda h:rpc('discard',h),capacity=1,event_sink=event)
 servers=[serve_stage(broker,'prefill',cfg['prefill_port']),serve_stage(broker,'decode',cfg['decode_port'])]
 (out/'ready.json').write_text(json.dumps({'ready':True,'pid':os.getpid(),'atom_device':cfg['atom_device']}));print('READY',flush=True)
 while running:
  try:kind,args,future=commands.get(timeout=.25)
  except queue.Empty:
   if (out/'stop').exists():break
   continue
  try:
   if kind=='prepare':
    rid,tokens=args;kp,vp,_=engine.prefill_device(tokens[:-1]);sync(1)
    payload,maximum=pack(kp,vp);sync(1);assert np.isfinite(float(maximum.cpu().item())) and float(maximum.cpu().item())<2**31
    source=copy_from_tensor(payload,FlatDramShape.new_as_broadcast(list(payload.shape),torch.float16),'furiosa:1');sync(1)
    allocation=SourceAllocation(source.data_ptr(),2097152,(source,payload,kp,vp),'dlfloat16-token-major-v1')
    plan=layout.prefix_plan(allocation,destinations,metadata['mappings'],16)
    receipt=session.admit(rid,plan,lambda:None);event(dict(stage='p2p_complete',request_id=rid,**receipt));future.set_result(rid)
   elif kind=='decode':
    rid,tokens,count=args;session.start_decode(rid);conn.send(json.dumps(dict(cmd='decode',id=rid,tokens=tokens,output=count)).encode());reply=json.loads(conn.recv(65536));assert reply['request_id']==rid
    session.finish_decode(rid,reply['passed']);future.set_result({'id':rid,'object':'text_completion','model':'koni-p2p','choices':[{'index':0,'text':tokenizer.decode(reply['tokens'],skip_special_tokens=True),'token_ids':reply['tokens'],'finish_reason':'length'}],'usage':{'prompt_tokens':9,'completion_tokens':32,'total_tokens':41}})
   elif kind=='discard':session.discard_ready(args[0]);future.set_result(None)
   else:raise ValueError('unknown command')
  except BaseException as e:
   future.set_exception(e)
   if session.state=='poisoned':raise
 result.update(passed=True)
except Exception as e:result.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 for server,thread in servers:server.shutdown();server.server_close();thread.join()
 if worker is not None and worker.poll() is None and 'conn' in locals():
  conn.send(json.dumps({'cmd':'stop'}).encode());worker.wait()
 for fd in fds:os.close(fd)
 if 'conn' in locals():conn.close()
 if 'listener' in locals():listener.close()
 if 'log' in locals():log.close()
 if (out/'worker-result.json').exists():
  result['worker']=json.loads((out/'worker-result.json').read_text());result['passed']=result['passed'] and result['worker']['passed']
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
