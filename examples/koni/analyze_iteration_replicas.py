"""Replay per-replica generations, leases and completion fences independently."""
import argparse,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p=a.run
http=json.loads((p/'http-results.json').read_text());assert http['passed'];cfg=json.loads((p/'config.json').read_text())
events=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
states={d:dict(slots={},active=None,generation=0) for d in cfg['atom_devices']};done={};copies=[];starts=[];refills=[];completions={};max_active=0
for e in events:
 stage=e['stage']
 if stage not in ('p2p_complete','iteration_start','iteration_complete','response_sent'):continue
 device=e['replica'];state=states[device];slots=state['slots'];active=state['active']
 if stage=='p2p_complete':
  assert active is None and e['slot'] not in slots
  assert e['requested']==e['completed']==64 and e['cqe_result']==0
  if any(s['produced']>0 for s in slots.values()):refills.append(e)
  slots[e['slot']]=dict(id=e['request_id'],produced=0,tokens=[]);copies.append(e)
 elif stage=='iteration_start':
  assert active is None and e['generation']==state['generation']+1;state['generation']=e['generation']
  jobs=e['requests'];assert len(jobs)==len(slots) and {j['slot'] for j in jobs}==set(slots)
  for j in jobs:
   s=slots[j['slot']];assert j['id']==s['id'] and j['produced']==s['produced'];assert 0<e['steps']<=j['output']-j['produced'];s['output']=j['output']
  state['active']=e;starts.append(e);max_active=max(max_active,sum(s['active'] is not None for s in states.values()))
 elif stage=='iteration_complete':
  assert active is not None and e['generation']==active['generation'] and e['steps']==active['steps']
  assert e['device_completed'] is True and e['passed'] is True
  assert len(e['requests'])==len(slots) and {r['slot'] for r in e['requests']}==set(slots)
  for row in e['requests']:
   s=slots[row['slot']];assert row['request_id']==s['id'] and row['passed'] and row['reference_checked'] and row['max_abs_diff']==0
   assert row['produced']==s['produced']+e['steps']==len(row['tokens']) and row['tokens'][:s['produced']]==s['tokens']
   assert row['complete']==(row['produced']==s['output'])
   if row['complete']:
    assert row['request_id'] not in done;done[row['request_id']]=row;del slots[row['slot']]
   else:s.update(produced=row['produced'],tokens=row['tokens'])
  completions[(device,e['generation'])]=e;state['active']=None
 elif stage=='response_sent':assert e['request_id'] in done
assert all(s['active'] is None and not s['slots'] and s['generation']>0 for s in states.values())
assert len(done)==len(copies)==sum(len(r['requests']) for r in http['rows'])==424
assert set(done)=={r['id'] for row in http['rows'] for r in row['requests']}
assert max_active==len(states)==2
assert any(e['request_id'].startswith('probe-') for e in refills)
summary=dict(requests=len(done),exact_logits=True,ownership_violations=0,refills_with_continuing_requests=len(refills),max_outstanding_replicas=max_active,median_dma_ms=statistics.median(e['dma_seconds'] for e in copies)*1000,config=cfg,source=json.loads((p/'source-manifest.json').read_text()),cases={})
for case in ['fixed32','fixed128','mixed32']:
 summary['cases'][case]={}
 for policy in ['sealed','refill']:
  prefix=f'{policy}-{case}-r';rows=[r for r in http['rows'] if r['scenario'].startswith(prefix)];assert len(rows)==3
  groups=[e for e in starts if all(j['id'].startswith(prefix) for j in e['requests'])]
  assert all(e['policy']['refill']==(policy=='refill') for e in groups)
  req=[q for r in rows for q in r['requests']];v=[r['tokens_per_second'] for r in rows]
  summary['cases'][case][policy]=dict(median_tps=statistics.median(v),tps_repeats=v,median_response_seconds=statistics.median(q['end']-q['start'] for q in req),response_by_output={str(n):statistics.median(q['end']-q['start'] for q in req if q['output_tokens']==n) for n in sorted({q['output_tokens'] for q in req})},useful_lane_fraction=sum(e['steps']*len(e['requests']) for e in groups)/sum(e['steps']*e['compiled_batch'] for e in groups),iteration_count=len(groups),requests_by_replica={str(d):sum(1 for e in copies if e['replica']==d and e['request_id'].startswith(prefix)) for d in states})
  summary['cases'][case][policy]['output_tokens_by_replica']={str(d):sum(g['steps']*len(g['requests']) for g in groups if g['replica']==d) for d in states}
 summary['cases'][case]['refill_vs_sealed']=summary['cases'][case]['refill']['median_tps']/summary['cases'][case]['sealed']['median_tps']
a.output.write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('source','config')},indent=2))
