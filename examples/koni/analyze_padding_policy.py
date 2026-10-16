"""Validate full-logit/P2P ownership evidence and compare all policy repeats."""
import argparse,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
p=a.run;http=json.loads((p/'http-results.json').read_text());assert http['passed']
events=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
active={};starts=[];done=[];copies=[];completion={}
for e in events:
 if e['stage']=='batch_start':
  assert e['replica'] not in active;active[e['replica']]=e;starts.append(e)
 elif e['stage']=='p2p_complete':
  assert e['replica'] not in active and e['requested']==e['completed']==64 and e['cqe_result']==0;copies.append(e)
 elif e['stage']=='response_sent':assert e['request_id'] in active[e['replica']]['request_ids']
 elif e['stage']=='batch_complete':
  start=active.pop(e['replica']);assert {r['request_id'] for r in e['requests']}==set(start['request_ids']);completion[start['batch_id']]=e;done.extend(e['requests'])
assert not active
assert all(r['passed'] and r['reference_checked'] and r['max_abs_diff']==0 for r in done)
assert len(done)==len(copies)==sum(len(r['requests']) for r in http['rows'])==400
assert {r['request_id'] for r in done}=={r['id'] for row in http['rows'] for r in row['requests']}
summary={'requests':len(done),'exact_logits':True,'ownership_violations':0,'median_dma_ms':1000*statistics.median(e['dma_seconds'] for e in copies),'config':json.loads((p/'config.json').read_text()),'source':json.loads((p/'source-manifest.json').read_text()),'cases':{}}
for case in ['fixed32','fixed128','mixed32']:
 summary['cases'][case]={}
 for policy in ['length','padding']:
  prefix=f'{policy}-{case}-r';rows=[r for r in http['rows'] if r['scenario'].startswith(prefix)];assert len(rows)==3
  groups=[e for e in starts if all(rid.startswith(prefix) for rid in e['request_ids'])]
  assert all(g['policy']['policy']==policy for g in groups)
  req=[q for r in rows for q in r['requests']];v=[r['tokens_per_second'] for r in rows]
  summary['cases'][case][policy]=dict(median_tps=statistics.median(v),tps_repeats=v,median_response_seconds=statistics.median(q['end']-q['start'] for q in req),response_by_output={str(n):statistics.median(q['end']-q['start'] for q in req if q['output_tokens']==n) for n in sorted({q['output_tokens'] for q in req})},useful_lane_fraction=sum(sum(g['output_lengths']) for g in groups)/sum(g['compiled_batch']*max(g['output_lengths']) for g in groups),batch_wall_seconds=sum(completion[g['batch_id']]['time']-g['time'] for g in groups),batch_count=len(groups))
 summary['cases'][case]['padding_vs_length']=summary['cases'][case]['padding']['median_tps']/summary['cases'][case]['length']['median_tps']
a.output.write_text(json.dumps(summary,indent=2));print(json.dumps(summary['cases'],indent=2))
