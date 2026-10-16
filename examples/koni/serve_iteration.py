"""Experimental KONI HTTP microbatch server with fenced P2P admission."""
from pathlib import Path
from types import SimpleNamespace
import json,time,traceback,subprocess,os,socket,array,hashlib,selectors
from batch_control import assert_atom_idle, worker_environment, accept_worker
import argparse
from hetero_p2p import SourceAllocation, DestinationAllocation, RngdPdmaBackend
from hetero_p2p.layout import TokenMajorLayout
from hetero_p2p import CopySession
from hetero_p2p.iteration import IterationScheduler,QuantumPolicy
from hetero_p2p.length_queue import LengthQueue
from hetero_p2p.serving import BatchOptions,CompletionLedger,RngdBinding,validate_koni_batch
from hetero_p2p.llmd import StageBroker,StageError,serve_stage
import queue,concurrent.futures,threading
ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());p=Path(cfg["out"]).resolve();p.mkdir(parents=True,exist_ok=False)
example=Path(__file__).resolve().parent
options=BatchOptions(cfg.get('batch_size',4),cfg.get('lookahead',2*cfg.get('batch_size',4)),cfg.get('collection_ms',20)/1000)
validate_koni_batch(options.width,cfg.get('experimental_batch16',False),cfg.get('experimental_batch12',False))

if len(str(p/"bridge.sock").encode())>=108:raise ValueError("output socket path is too long")
binding=RngdBinding.from_path(cfg.get('rngd_device_path','/dev/rngd/npu1ch0'),cfg.get('rngd_logical_index'))
devices=cfg["atom_devices"]
if not isinstance(devices,list) or not devices or any(type(d) is not int or d not in (2,3) for d in devices) or len(set(devices))!=len(devices):raise ValueError("select a nonempty unique subset of authorized ATOM2/3")
if f'npu{binding.index}:' in subprocess.check_output(['furiosa-smi','ps'],text=True):raise RuntimeError('selected RNGD is in use')
(p/"config.json").write_text(json.dumps(cfg,indent=2));os.environ["HETERO_BATCH_CONFIG"]=str((p/"config.json").resolve())
backend=RngdPdmaBackend(cfg["helper"],cfg.get("rngd_device_path","/dev/rngd/npu1ch0"));layout=TokenMajorLayout(64,8,512,128)
assert_atom_idle(p)
import torch,numpy as np
from safetensors import safe_open
from rngd_pd.engines.model import ModelConfig
from rngd_pd.engines.prefill_full import FullPrefillEngine
from rngd_pd.engines.fsdk import sync
from hetero_p2p.adapters.rngd_dlfloat import BoundedPrefixPacker
import furiosa.torch as ft
from furiosa.torch._C.tensor import copy_from_tensor
from furiosa.torch._C.ir import FlatDramShape
# Acquire the actual compute device before exporting ATOM memory. SMI alone
# can omit container-owned contexts. Keep this probe alive for the run.
try:
 binding_probe=torch.zeros(16,dtype=torch.bfloat16).to(binding.torch_device);sync(binding.compute_index)
 channel_paths=[]
 for descriptor in Path('/proc/self/fd').iterdir():
  try:channel_paths.append(os.readlink(descriptor))
  except OSError:pass
 binding.verify_open_channels(channel_paths)
 (p/'compute-binding.json').write_text(json.dumps(dict(physical=binding.device_path,logical=binding.torch_device,paths=[x for x in channel_paths if '/dev/rngd/' in x]),indent=2))
except Exception as error:
 (p/'result.json').write_text(json.dumps(dict(passed=False,phase='compute acquisition',hardware_dma_started=False,error=repr(error)),indent=2));raise
class KVOnly(FullPrefillEngine):
 def _build_lmhead(self,cache_dir):pass
 def _next_token(self,x,P):return None
result={'kind':'PD_HTTP_batch','passed':False};wave_rows=[]
commands=queue.Queue();rpc_lock=threading.Lock();accepting=True;servers=[];decode_futures={};pending=[];live_batches={};sessions={};tokenizer=None;staged={};waiting=LengthQueue(options.width,options.lookahead,options.collection_seconds);policy={'policy':'length','refill':len(devices)==1,'quantum':128,'partial_quantum':4}
def event(row):
 with (p/'events.jsonl').open('a') as f:f.write(json.dumps(dict({'time':time.perf_counter()},**row))+'\n')
def rpc(kind,*args):
 future=concurrent.futures.Future()
 with rpc_lock:
  if not accepting:raise StageError(503,'server stopping')
  commands.put((kind,args,future))
 return future.result()
def prepare(rid,tokens):
 if not 2<=len(tokens)<=max(prefixes)+1:raise StageError(400,'prompt exceeds configured prefill bucket')
 return rpc('prepare',rid,tokens)
def decode(handle,tokens,count):
 if count>128:raise StageError(400,'output limit is 128')
 return rpc('decode',handle,tokens,count)
workload=json.loads(Path(cfg['workload']).read_text())
prompts=[r['prompt_token_ids'] for r in workload['requests']]
if len(prompts)!=4 or any(not 2<=len(x)<=384 or any(type(t) is not int or not 0<=t<128256 for t in x) for x in prompts):raise ValueError('require four valid KONI token prompts of length2..384')
if not cfg.get('output_lengths',[32,128]) or any(n not in (32,128) for n in cfg.get('output_lengths',[32,128])):raise ValueError('validated output lengths are 32 and 128')
if type(cfg.get('repeats',3)) is not int or cfg.get('repeats',3)<1:raise ValueError('positive measured repeat count required')
prompts=prompts*(options.width//4)
prefixes=[len(x)-1 for x in prompts]
head_bytes=[((n+15)//16)*4096 for n in prefixes]
try:
 torch.set_num_threads(8)
 for batch in [options.width]:
  assert_atom_idle(p)
  phase=p/f'b{batch}';phase.mkdir();workers={};owners={};streams=[]
  with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as server:
   server.bind(os.path.relpath(p/'bridge.sock'));server.listen(len(devices))
   for device in devices:
    d=phase/f'atom{device}';d.mkdir()
    g=(example/'capture.gdb').read_text()
    (d/'capture.gdb').write_text(g);log=(d/'worker.log').open('w');streams.append(log)
    workers[device]=subprocess.Popen(['gdb','-q','-batch','-x',str(d/'capture.gdb'),'--args',cfg['atom_python'],str(example/'serve_length_worker.py'),str(d),str(batch),str(device)],stdout=log,stderr=subprocess.STDOUT,env=dict(worker_environment(os.environ,device),HETERO_CAPTURE_DIR=str(d.resolve())))
   for _ in devices:
    conn,_=accept_worker(server,workers);data,anc,flags,_=conn.recvmsg(65536,socket.CMSG_SPACE(64*4));assert not flags&(socket.MSG_TRUNC|socket.MSG_CTRUNC)
    fds=array.array('i')
    for level,kind,content in anc:
     if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:fds.frombytes(content)
    meta=json.loads(data);device=meta['device'];assert device in workers and device not in owners;assert meta['batch']==batch and len(meta['mappings'])==64 and len(fds)==len(meta['blocks'])
    for fd,block in zip(fds,meta['blocks']):assert os.fstat(fd).st_size==block['size']
    (phase/f'atom{device}'/'export-metadata.json').write_text(json.dumps(meta,indent=2))
    owners[device]={'conn':conn,'fds':fds,'metadata':meta}
    print('WORKER_READY',batch,device,flush=True)
   if batch in (4,8,12,16):
    modeldir=Path(cfg['model'])
    model_cfg=ModelConfig(SimpleNamespace(**json.loads((modeldir/'config.json').read_text())));sd={}
    for shard in sorted(modeldir.glob('model-*.safetensors')):
     with safe_open(shard,framework='pt') as f:
      for key in f.keys():sd[key]=f.get_tensor(key).float().numpy()
    print('WEIGHTS_LOADED',flush=True)
    engine=KVOnly(model_cfg,sd,C=((max(prefixes)+15)//16)*16,group=2,chip=binding.compute_index,cache_dir=None,S=512)
    packer=BoundedPrefixPacker(binding.torch_device);print('BOUNDED_PACKER_READY',flush=True)
    print('RNGD_READY',flush=True)
    files=[]
    for slot,prompt in enumerate(prompts):
     kp,vp,_=engine.prefill_device(prompt[:-1]);sync(binding.compute_index)
     file=phase/f'host-kv-{slot}.npz';np.savez(file,keys=kp.cpu().float().numpy(),values=vp.cpu().float().numpy());files.append(str(file))
    for owner in owners.values():owner['conn'].send(json.dumps({'cmd':'reference','files':files,'output_tokens':128}).encode())
    refs=[json.loads(owner['conn'].recv(65536)) for owner in owners.values()];assert all(r['passed'] for r in refs) and len({r['logits_sha256'] for r in refs})==1
    (phase/'reference-parity.json').write_text(json.dumps(refs,indent=2));print('REFERENCES_READY',flush=True)
   from transformers import AutoTokenizer
   tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
   schedulers={device:IterationScheduler(batch) for device in devices};initial_since={}
   broker=StageBroker('koni-p2p',prepare,decode,lambda h:rpc('discard',h),capacity=cfg.get('queue_capacity',64),ttl=cfg.get('admission_ttl_seconds',60),event_sink=lambda row:event(row))
   servers=[serve_stage(broker,'prefill',cfg['prefill_port']),serve_stage(broker,'decode',cfg['decode_port'])]
   selector=selectors.DefaultSelector()
   for device,owner in owners.items():selector.register(owner['conn'],selectors.EVENT_READ,device)
   def stage_source(rid,tokens):
    if rid in staged or any(r.request_id==rid for scheduler in schedulers.values() for r in scheduler.slots.values()):raise StageError(409,'request buffers still leased')
    started=time.perf_counter()
    kp,vp,_=engine.prefill_device(tokens[:-1]);sync(binding.compute_index)
    payload,maximum=packer.encode(kp,vp,len(tokens)-1);sync(binding.compute_index)
    source=copy_from_tensor(payload,FlatDramShape.new_as_broadcast(list(payload.shape),torch.float16),binding.torch_device);sync(binding.compute_index)
    padded=((len(tokens)-1+15)//16)*16
    allocation=SourceAllocation(source.data_ptr(),64*8*padded*128*2,(source,payload),'dlfloat16-token-major-v1',device=binding.device_path)
    staged[rid]=(tuple(tokens),allocation,padded)
    event(dict(stage='source_ready',request_id=rid,bytes=allocation.size,seconds=time.perf_counter()-started))
   def admit(entry,device):
    scheduler=schedulers[device];owner=owners[device]
    assert scheduler.active is None
    tokens,allocation,padded=staged[entry.request_id]
    destinations=tuple(DestinationAllocation(fd,owner['conn'],'dlfloat16-token-major-v1') for fd in owner['fds'])
    plan=layout.prefix_plan(allocation,destinations,owner['metadata']['mappings'],padded,slot=entry.slot,batch=batch)
    session=CopySession(backend);sessions[entry.request_id]=session
    started=time.perf_counter();receipt=session.admit(entry.request_id,plan,lambda:None);scheduler.mark_admitted(entry.request_id)
    session.start_decode(entry.request_id);del staged[entry.request_id]
    event(dict(stage='p2p_complete',request_id=entry.request_id,replica=device,slot=entry.slot,admission_seconds=time.perf_counter()-started,**receipt))
   def respond(row,entry,device):
    rid=entry.request_id;future=decode_futures.pop(rid)
    response={'id':rid,'object':'text_completion','model':'koni-p2p','choices':[{'index':0,'text':tokenizer.decode(row['tokens'],skip_special_tokens=True),'token_ids':row['tokens'],'finish_reason':'length'}],'usage':{'prompt_tokens':len(entry.owner),'completion_tokens':len(row['tokens']),'total_tokens':len(entry.owner)+len(row['tokens'])}}
    future.set_result(response);event(dict(stage='response_sent',request_id=rid,replica=device))
   (p/'ready.json').write_text(json.dumps(dict(ready=True,pid=os.getpid(),prefill_port=cfg['prefill_port'],decode_port=cfg['decode_port'])))
   print('HTTP_READY',flush=True)
   while True:
    for key,_ in selector.select(timeout=0):
     device=key.data;owner=owners[device];scheduler=schedulers[device]
     message=owner['conn'].recv(65536)
     if not message:raise RuntimeError('decode owner disconnected')
     reply=json.loads(message);ticket=scheduler.active
     finished=scheduler.accept_reply(ticket,reply)
     event(dict(stage='iteration_complete',replica=device,**reply))
     rows={r['request_id']:r for r in reply['requests']}
     for entry in finished:
      sessions.pop(entry.request_id).finish_decode(entry.request_id,True)
      respond(rows[entry.request_id],entry,device)
    for _ in range(64):
     try:kind,args,future=commands.get_nowait()
     except queue.Empty:break
     try:
      if kind=='prepare':pending.append((args,future))
      elif kind=='decode':
       rid,tokens,count=args
       if list(staged[rid][0])!=tokens:raise RuntimeError('decode prompt differs from source')
       waiting.add(rid,count,time.monotonic());decode_futures[rid]=future
      elif kind=='discard':
       rid=args[0];del staged[rid];future.set_result(None)
      else:raise RuntimeError('unknown owner command')
     except Exception as error:future.set_exception(error)
    if not pending and not staged and not any(s.slots for s in schedulers.values()) and not waiting.jobs:
     control=p/'policy.json'
     if control.exists():
      candidate=json.loads(control.read_text())
      if candidate.get('policy') not in ('fifo','length','padding') or type(candidate.get('refill')) is not bool or type(candidate.get('quantum')) is not int or not 1<=candidate['quantum']<=128:raise ValueError('invalid iteration policy')
      QuantumPolicy(candidate['quantum'],candidate.get('partial_quantum',candidate['quantum']))
      if candidate!=policy:policy=candidate;event(dict(stage='policy',**policy))
      (p/'policy-ack.json').write_text(json.dumps(policy))
    # Prepare sources during decode, but prioritize resuming a fenced active set.
    if pending and not any(s.active is None and any(r.produced>0 for r in s.slots.values()) for s in schedulers.values()):
     args,future=pending.pop(0);rid,tokens=args
     try:stage_source(rid,tokens);future.set_result(rid)
     except StageError as error:future.set_exception(error)
     except BaseException as error:future.set_exception(error);raise
    for device,scheduler in schedulers.items():
     owner=owners[device]
     if scheduler.active is None:
      free=batch-len(scheduler.slots)
      if free and (policy['refill'] or not scheduler.slots or all(r.produced==0 for r in scheduler.slots.values())):
       selected=waiting.take(time.monotonic(),more_prefills=bool(pending),policy=policy['policy'],limit=free)
       for job in selected:
        if not scheduler.slots:initial_since[device]=time.monotonic()
        entry=scheduler.reserve(job.request_id,job.output_tokens,staged[job.request_id][0]);assert entry is not None
        admit(entry,device)
      if scheduler.slots:
       if not scheduler.should_dispatch(time.monotonic(),initial_since[device],cfg.get('batch_wait_ms',100)/1000):continue
       quantum=QuantumPolicy(policy['quantum'],policy.get('partial_quantum',policy['quantum'])).choose(len(scheduler.slots),batch)
       ticket=scheduler.begin(quantum)
       jobs=[dict(slot=slot,id=rid,tokens=list(scheduler.slots[slot].owner),output=scheduler.slots[slot].output_tokens,produced=scheduler.slots[slot].produced,admitted=True) for slot,rid in ticket.requests]
       owner['conn'].send(json.dumps(dict(cmd='decode_iteration',generation=ticket.generation,steps=ticket.steps,requests=jobs)).encode())
       event(dict(stage='iteration_start',replica=device,generation=ticket.generation,steps=ticket.steps,requests=jobs,compiled_batch=batch,policy=dict(policy)))
    if (p/'stop').exists() and not any(s.slots or s.active is not None for s in schedulers.values()) and not pending and not staged and not waiting.jobs:break
    time.sleep(.001)
   selector.close();result.update(passed=True)
except Exception as error:
 result.update(passed=False,error=repr(error),traceback=traceback.format_exc());raise
finally:
 with rpc_lock:accepting=False
 for args,future in pending:
  if not future.done():future.set_exception(StageError(503,'server stopped'))
 for future in decode_futures.values():
  if not future.done():future.set_exception(StageError(503,'server stopped'))
 while not commands.empty():
  _,_,future=commands.get_nowait()
  if not future.done():future.set_exception(StageError(503,'server stopped'))
 for server,thread in servers:server.shutdown();server.server_close();thread.join()
 for owner in locals().get('owners',{}).values():
  try:owner['conn'].send(json.dumps({'cmd':'stop'}).encode())
  except OSError:pass
 for worker in locals().get('workers',{}).values():worker.wait()
 for owner in locals().get('owners',{}).values():
  owner['conn'].close()
  for fd in owner['fds']:os.close(fd)
 for log in locals().get('streams',[]):log.close()
 (p/'result.json').write_text(json.dumps(result,indent=2))
