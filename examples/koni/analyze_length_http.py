import argparse,collections,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();p=a.directory
result=json.loads((p/'http-results.json').read_text());assert result['passed']
events=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
active={};done=[];early=0;copies=0;batch_start={};violations=[]
for e in events:
 if e['stage']=='batch_start':
  if e['replica'] in active:violations.append('overlapping decode')
  active[e['replica']]=set(e['request_ids']);batch_start[e['batch_id']]=e
 elif e['stage']=='p2p_complete':
  if e['replica'] in active:violations.append('DMA into running replica')
  assert e['completed']==e['requested']==64 and e['cqe_result']==0;copies+=1
 elif e['stage']=='response_sent':
  assert e['request_id'] in active[e['replica']]
 elif e['stage']=='batch_complete':
  assert {r['request_id'] for r in e['requests']}==active.pop(e['replica'])
  done.extend(e['requests'])
assert not active and not violations
assert all(r['passed'] and r['reference_checked'] and r['max_abs_diff']==0 for r in done)
assert len(done)==copies==sum(len(r['requests']) for r in result['rows'])
summary={}
for name in ['fifo','early','length']:
 rows=[r for r in result['rows'] if r['policy']==name and r['kind']=='mixed32'];assert len(rows)==3
 latencies={str(n):[q['end']-q['start'] for r in rows for q in r['requests'] if q['output_tokens']==n] for n in [32,64,96,128]}
 starts=[e for e in batch_start.values() if all(rid.startswith(name+'-mixed32-') for rid in e['request_ids'])]
 useful=sum(sum(e['output_lengths']) for e in starts);allocated=sum(4*max(e['output_lengths']) for e in starts)
 summary[name]=dict(tokens_per_second=[r['tokens_per_second'] for r in rows],median_tokens_per_second=statistics.median(r['tokens_per_second'] for r in rows),median_latency_by_output={n:statistics.median(v) for n,v in latencies.items()},max_latency_by_output={n:max(v) for n,v in latencies.items()},useful_lane_fraction=useful/allocated)
extras=[{k:v for k,v in r.items() if k!='requests'} for r in result['rows'] if r['kind'] not in ['mixed32','warm']]
out=dict(passed=True,requests=len(done),variants=summary,extras=extras,ownership_violations=violations,exact_full_logit_parity=True,power_measured=False)
(p/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
