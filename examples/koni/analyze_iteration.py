"""Replay every slot lease and synchronized generation before reporting metrics."""
import argparse,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--saturation',action='store_true');a=ap.parse_args();p=a.run
http=json.loads((p/'http-results.json').read_text());assert http['passed']
events=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
slots={};active=None;generation=0;done={};copies=[];starts=[];refills=[];completions={}
for e in events:
 stage=e['stage']
 if stage=='p2p_complete':
  assert active is None and e['slot'] not in slots
  assert e['requested']==e['completed']==64 and e['cqe_result']==0
  if any(s['produced']>0 for s in slots.values()):refills.append(e)
  slots[e['slot']]=dict(id=e['request_id'],produced=0,tokens=[]);copies.append(e)
 elif stage=='iteration_start':
  assert active is None and e['generation']==generation+1;generation=e['generation']
  jobs=e['requests'];assert len(jobs)==len(slots) and {j['slot'] for j in jobs}==set(slots)
  for j in jobs:
   s=slots[j['slot']];assert j['id']==s['id'] and j['produced']==s['produced'];assert 0<e['steps']<=j['output']-j['produced'];s['output']=j['output']
  active=e;starts.append(e)
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
  completions[e['generation']]=e;active=None
 elif stage=='response_sent':assert e['request_id'] in done
assert active is None and not slots
cfg=json.loads((p/'config.json').read_text())
assert len(http['rows'])==(5 if a.saturation else 20)
warm=[r for r in http['rows'] if r['scenario']=='warm'];assert len(warm)==1
request_count=len(warm[0]['requests']);assert request_count in ((48,) if a.saturation else (16,32))
probe_count=10 if cfg.get('batch_size',8)<=8 else 32
expected_count=probe_count+request_count+(3*48 if a.saturation else 6*(2*request_count+32))
assert len(done)==len(copies)==sum(len(r['requests']) for r in http['rows'])==expected_count
assert set(done)=={r['id'] for row in http['rows'] for r in row['requests']}
assert any(e['request_id'].startswith('probe-') for e in refills),'probe did not exercise refill of a running cohort'
summary=dict(requests=len(done),exact_logits=True,ownership_violations=0,refills_with_continuing_requests=len(refills),median_dma_ms=statistics.median(e['dma_seconds'] for e in copies)*1000,config=json.loads((p/'config.json').read_text()),source=json.loads((p/'source-manifest.json').read_text()),cases={})
for case in (['fixed128'] if a.saturation else ['fixed32','fixed128','mixed32']):
 summary['cases'][case]={}
 for policy in (['sealed'] if a.saturation else ['sealed','refill']):
  prefix=f'{policy}-{case}-r';rows=[r for r in http['rows'] if r['scenario'].startswith(prefix)];assert len(rows)==3
  groups=[e for e in starts if all(j['id'].startswith(prefix) for j in e['requests'])]
  assert all(e['policy']['refill']==(policy=='refill') for e in groups)
  req=[q for r in rows for q in r['requests']];v=[r['tokens_per_second'] for r in rows]
  summary['cases'][case][policy]=dict(median_tps=statistics.median(v),tps_repeats=v,median_response_seconds=statistics.median(q['end']-q['start'] for q in req),response_by_output={str(n):statistics.median(q['end']-q['start'] for q in req if q['output_tokens']==n) for n in sorted({q['output_tokens'] for q in req})},useful_lane_fraction=sum(e['steps']*len(e['requests']) for e in groups)/sum(e['steps']*e['compiled_batch'] for e in groups),iteration_count=len(groups),device_seconds=sum(completions[e['generation']]['seconds'] for e in groups),producer_window_seconds=sum(completions[e['generation']]['time']-e['time'] for e in groups))
  engine_tpot=[1000*(done[q['id']]['end']-done[q['id']]['first'])/(q['output_tokens']-1) for q in req if q['output_tokens']>1]
  summary['cases'][case][policy]['internal_request_avg_tpot_ms']=dict(median=statistics.median(engine_tpot),p99=statistics.quantiles(engine_tpot,n=100,method='inclusive')[98],scope='worker token timestamps; request-average TPOT distribution, not client streaming or token-interval p99')
 if not a.saturation:summary['cases'][case]['refill_vs_sealed']=summary['cases'][case]['refill']['median_tps']/summary['cases'][case]['sealed']['median_tps']
a.output.write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('source','config')},indent=2))
