"""Configurable KONI B4 pipeline using library planning/export/transport.

Requires the rngd_pd model adapter and the pinned SDKs; this is a benchmark,
not an HTTP microbatch scheduler. Full logits are checked against a host reference.
"""
from pathlib import Path
from types import SimpleNamespace
import json,time,traceback,subprocess,os,socket,array,hashlib,selectors
from batch_control import assert_atom_idle, worker_environment, accept_worker
import argparse
from hetero_p2p import SourceAllocation, DestinationAllocation, RngdPdmaBackend
from hetero_p2p.layout import TokenMajorLayout
ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());p=Path(cfg["out"]).resolve();p.mkdir(parents=True,exist_ok=False)
example=Path(__file__).resolve().parent
if len(str(p/"bridge.sock").encode())>=108:raise ValueError("output socket path is too long")
if cfg.get("rngd_device_path","/dev/rngd/npu1ch0")!="/dev/rngd/npu1ch0":raise ValueError("KONI adapter validates RNGD1 only")
if cfg["atom_devices"]!=[2,3]:raise ValueError("this batch example validates ATOM2/3")
if "npu1" in subprocess.check_output(["furiosa-smi","ps"],text=True):raise RuntimeError("RNGD1 is in use")
(p/"config.json").write_text(json.dumps(cfg,indent=2));os.environ["HETERO_BATCH_CONFIG"]=str((p/"config.json").resolve())
backend=RngdPdmaBackend(cfg["helper"],cfg.get("rngd_device_path","/dev/rngd/npu1ch0"));layout=TokenMajorLayout(64,8,512,128)
assert_atom_idle(p)
from rngd_pd.native_v4.replica_lease import ReplicaLease
from rngd_pd.native_v4.power import PowerSampler
import torch,numpy as np
from safetensors import safe_open
from rngd_pd.engines.model import ModelConfig
from rngd_pd.engines.prefill_full import FullPrefillEngine
from rngd_pd.engines.fsdk import sync
from hetero_p2p.adapters.rngd_dlfloat import BoundedPrefixPacker
import furiosa.torch as ft
from furiosa.torch._C.tensor import copy_from_tensor
from furiosa.torch._C.ir import FlatDramShape
class KVOnly(FullPrefillEngine):
 def _build_lmhead(self,cache_dir):pass
 def _next_token(self,x,P):return None
result={'kind':'PD_overlap_pipeline','passed':False};wave_rows=[]
workload=json.loads(Path(cfg['workload']).read_text())
prompts=[r['prompt_token_ids'] for r in workload['requests']]
if len(prompts)!=4 or any(not 2<=len(x)<=384 or any(type(t) is not int or not 0<=t<128256 for t in x) for x in prompts):raise ValueError('require four valid KONI token prompts of length2..384')
if not cfg.get('output_lengths',[32,128]) or any(n not in (32,128) for n in cfg.get('output_lengths',[32,128])):raise ValueError('validated output lengths are 32 and 128')
if type(cfg.get('repeats',3)) is not int or cfg.get('repeats',3)<1:raise ValueError('positive measured repeat count required')
prefixes=[len(x)-1 for x in prompts]
head_bytes=[((n+15)//16)*4096 for n in prefixes]
try:
 torch.set_num_threads(8)
 for batch in [4]:
  assert_atom_idle(p)
  phase=p/f'b{batch}';phase.mkdir();workers={};owners={};streams=[]
  with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as server:
   server.bind(os.path.relpath(p/'bridge.sock'));server.listen(2)
   for device in [2,3]:
    d=phase/f'atom{device}';d.mkdir()
    g=(example/'capture.gdb').read_text()
    (d/'capture.gdb').write_text(g);log=(d/'worker.log').open('w');streams.append(log)
    workers[device]=subprocess.Popen(['gdb','-q','-batch','-x',str(d/'capture.gdb'),'--args',cfg['atom_python'],str(example/'batch_worker.py'),str(d),str(batch),str(device)],stdout=log,stderr=subprocess.STDOUT,env=dict(worker_environment(os.environ,device),HETERO_CAPTURE_DIR=str(d.resolve())))
   for _ in range(2):
    conn,_=accept_worker(server,workers);data,anc,flags,_=conn.recvmsg(65536,socket.CMSG_SPACE(64*4));assert not flags&(socket.MSG_TRUNC|socket.MSG_CTRUNC)
    fds=array.array('i')
    for level,kind,content in anc:
     if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:fds.frombytes(content)
    meta=json.loads(data);device=meta['device'];assert device in workers and device not in owners;assert meta['batch']==batch and len(meta['mappings'])==64 and len(fds)==len(meta['blocks'])
    for fd,block in zip(fds,meta['blocks']):assert os.fstat(fd).st_size==block['size']
    plans=[]
    for slot in range(batch):
     plan=phase/f'atom{device}'/f'plan-slot{slot}.txt';lines=[]
     for i,m in enumerate(meta['mappings']):
      assert m['index']==i
      for head in range(8):lines.append(f"{fds[m['block_index']]} {(i*8+head)*head_bytes[slot]} {m['offset']+(slot*8+head)*131072} {head_bytes[slot]}\n")
     plan.write_text(''.join(lines));plans.append(plan)
    (phase/f'atom{device}'/'export-metadata.json').write_text(json.dumps(meta,indent=2))
    owners[device]={'conn':conn,'fds':fds,'plans':plans,'metadata':meta,'lease':ReplicaLease()}
    print('WORKER_READY',batch,device,flush=True)
   if batch==4:
    modeldir=Path(cfg['model'])
    model_cfg=ModelConfig(SimpleNamespace(**json.loads((modeldir/'config.json').read_text())));sd={}
    for shard in sorted(modeldir.glob('model-*.safetensors')):
     with safe_open(shard,framework='pt') as f:
      for key in f.keys():sd[key]=f.get_tensor(key).float().numpy()
    print('WEIGHTS_LOADED',flush=True)
    engine=KVOnly(model_cfg,sd,C=((max(prefixes)+15)//16)*16,group=2,chip=1,cache_dir=None,S=512)
    sample=(torch.zeros((32,8,128,512),dtype=torch.bfloat16),torch.zeros((32,8,512,128),dtype=torch.bfloat16))
    packer=BoundedPrefixPacker('furiosa:1');print('BOUNDED_PACKER_READY',flush=True)
    print('RNGD_READY',flush=True)
    files=[]
    for slot,prompt in enumerate(prompts):
     kp,vp,_=engine.prefill_device(prompt[:-1]);sync(1)
     file=phase/f'host-kv-{slot}.npz';np.savez(file,keys=kp.cpu().float().numpy(),values=vp.cpu().float().numpy());files.append(str(file))
    for owner in owners.values():owner['conn'].send(json.dumps({'cmd':'reference','files':files,'output_tokens':128}).encode())
    refs=[json.loads(owner['conn'].recv(65536)) for owner in owners.values()];assert all(r['passed'] for r in refs) and len({r['logits_sha256'] for r in refs})==1
    (phase/'reference-parity.json').write_text(json.dumps(refs,indent=2));print('REFERENCES_READY',flush=True)
   active=[3,2]
   def admit(device,request_id,admissions):
    owner=owners[device];owner['lease'].begin_admission(request_id);completed=0
    for slot in range(batch):
     begin=time.perf_counter();t=begin
     kp,vp,_=engine.prefill_device(prompts[slot][:-1]);sync(1);prefill=time.perf_counter()-t;t=time.perf_counter()
     payload,maximum=packer.encode(kp,vp,prefixes[slot]);sync(1);assert np.isfinite(maximum) and maximum<2**31
     source=copy_from_tensor(payload,FlatDramShape.new_as_broadcast(list(payload.shape),torch.float16),'furiosa:1');sync(1);conversion=time.perf_counter()-t;t=time.perf_counter()
     allocation=SourceAllocation(source.data_ptr(),512*head_bytes[slot],(source,payload,kp,vp),'dlfloat16-token-major-v1')
     destinations=tuple(DestinationAllocation(fd,owner['conn'],'dlfloat16-token-major-v1') for fd in owner['fds'])
     plan=layout.prefix_plan(allocation,destinations,owner['metadata']['mappings'],head_bytes[slot]//256,slot=slot,batch=batch)
     dma=backend.transfer(plan);assert dma['completed']==64 and dma['cqe_result']==0;completed+=dma['completed']
     admissions.append({'request_id':request_id,'device':device,'slot':slot,'start':begin,'end':time.perf_counter(),'prefill_seconds':prefill,'conversion_seconds':conversion,'transport_seconds':time.perf_counter()-t,'dma':dma})
    owner['lease'].mark_ready(completed,64*batch)
   def dispatch(device):
    owner=owners[device];owner['lease'].start_decode()
    owner['conn'].send(json.dumps({'cmd':'decode','wave_id':owner['lease'].request_id,'completed_cqes':64*batch,'output_tokens':output_tokens}).encode())
   def receive(device,replies):
    owner=owners[device];reply=json.loads(owner['conn'].recv(65536))
    assert reply['device']==device
    owner['lease'].finish_decode(reply['wave_id'],reply['passed']);replies.append(reply)
   sampler=PowerSampler(p,'npu1',[2,3],interval=.2);sampler.start();time.sleep(.3)
   for repetition in range(-cfg.get('warmups',1),cfg.get('repeats',3)):
    modes=[f'pipeline-{length}' for length in cfg.get('output_lengths',[32,128])]
    for mode in modes:
     output_tokens=int(mode.split('-')[-1]);wave_id=f'b{batch}-r{repetition}-{mode}';start=time.perf_counter();admissions=[];replies=[];rounds=2
     assert all(o['lease'].state=='idle' for o in owners.values())
     if mode=='barrier':
      for cycle in range(rounds):
       for device in active:admit(device,f'{wave_id}-d{device}-c{cycle}',admissions)
       for device in active:dispatch(device)
       for device in active:receive(device,replies)
     else:
      submitted={device:0 for device in active}
      with selectors.DefaultSelector() as selector:
       for device in active:selector.register(owners[device]['conn'],selectors.EVENT_READ,device)
       for device in active:
        admit(device,f'{wave_id}-d{device}-c0',admissions);dispatch(device);submitted[device]=1
       while len(replies)<rounds*len(active):
        for key,_ in selector.select():
         device=key.data;receive(device,replies)
         if submitted[device]<rounds:
          admit(device,f"{wave_id}-d{device}-c{submitted[device]}",admissions);dispatch(device);submitted[device]+=1
         else:selector.unregister(key.fileobj)
     assert all(o['lease'].state=='idle' for o in owners.values())
     # Timestamp audit is independent of the state-machine checks.
     own_overlap=[];other_overlap=0.
     for admission in admissions:
      for reply in replies:
       overlap=max(0.,min(admission['end'],reply['end'])-max(admission['start'],reply['start']))
       if admission['device']==reply['device'] and overlap>0:own_overlap.append((admission['request_id'],reply['wave_id'],overlap))
       elif admission['device']!=reply['device']:other_overlap+=overlap
     assert not own_overlap,own_overlap
     end=max(r['end'] for r in replies);requests=len(active)*batch*rounds
     row={'start':start,'end':end,'wave_id':wave_id,'mode':mode,'batch':batch,'cards':len(active),'rounds':rounds,'repetition':repetition,'warmup':repetition<0,'passed':True,'requests':requests,'output_length':output_tokens,'output_tokens':requests*output_tokens,'wall_seconds':end-start,'output_tokens_per_second':requests*output_tokens/(end-start),'first_completion_seconds':min(r['end'] for r in replies)-start,'first_token_seconds':min(r['first'] for r in replies)-start,'own_buffer_overlap_count':len(own_overlap),'cross_replica_overlap_pair_seconds':other_overlap,'admissions':admissions,'workers':replies}
     with (p/'waves.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
     wave_rows.append(row);print(wave_id,round(row['output_tokens_per_second'],2),'tokens/s',flush=True)
   sampler.finish(end)
   for owner in owners.values():
    owner['conn'].send(json.dumps({'cmd':'stop'}).encode());owner['conn'].close()
    for fd in owner['fds']:os.close(fd)
   for device,worker in workers.items():
    worker.wait();assert json.loads((phase/f'atom{device}'/'result.json').read_text())['passed']
   for log in streams:log.close()
  os.unlink(p/'bridge.sock');print('PHASE_COMPLETE',batch,flush=True)
 result.update(passed=True,waves=len(wave_rows),measured_waves=sum(not r['warmup'] for r in wave_rows))
except Exception as e:result.update(passed=False,error=repr(e),traceback=traceback.format_exc())
(p/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if not result['passed']:raise SystemExit(1)
