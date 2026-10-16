import argparse,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('directories',type=Path,nargs='+');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();summary={}
for p in a.directories:
 result=json.loads((p/'http-results.json').read_text());assert result['passed'];cfg=json.loads((p/'config.json').read_text());batch=cfg.get('batch_size',4)
 events=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()];active={};done=[];copies=[];starts=[]
 for e in events:
  if e['stage']=='batch_start':
   assert e['replica'] not in active;active[e['replica']]=set(e['request_ids']);starts.append(e)
  elif e['stage']=='p2p_complete':
   assert e['replica'] not in active;assert e['completed']==e['requested']==64 and e['cqe_result']==0;copies.append(e)
  elif e['stage']=='response_sent':assert e['request_id'] in active[e['replica']]
  elif e['stage']=='batch_complete':
   assert {r['request_id'] for r in e['requests']}==active.pop(e['replica']);done.extend(e['requests'])
 assert not active
 assert all(r['passed'] and r['reference_checked'] and r['max_abs_diff']==0 for r in done)
 assert len(done)==len(copies)==sum(len(r['requests']) for r in result['rows'])
 cases={}
 for kind in ['fixed32','fixed128','mixed32']:
  rows=[r for r in result['rows'] if r['scenario'].startswith(kind+'-r')];assert len(rows)==3
  values=[r['tokens_per_second'] for r in rows];qs=[q for r in rows for q in r['requests']]
  groups=[e for e in starts if all(rid.startswith(kind+'-r') for rid in e['request_ids'])]
  cases[kind]=dict(median_tps=statistics.median(values),tps_repeats=values,median_response_seconds=statistics.median(q['end']-q['start'] for q in qs),max_response_seconds=max(q['end']-q['start'] for q in qs),useful_lane_fraction=sum(sum(e['output_lengths']) for e in groups)/sum(batch*max(e['output_lengths']) for e in groups))
 summary[f'b{batch}']=dict(cases=cases,diagnostics=[{k:v for k,v in r.items() if k!='requests'} for r in result['rows'] if '-r' not in r['scenario']],requests=len(done),exact_logits=True,ownership_violations=0,median_dma_ms=1000*statistics.median(e['dma_seconds'] for e in copies),device_binding=json.loads((p/'compute-binding.json').read_text()),config=cfg,raw_root=str(p.resolve()),source_manifest=json.loads((p/'source-manifest.json').read_text()))
if 'b4' in summary and 'b8' in summary:summary['speedup_b8_vs_b4']={k:summary['b8']['cases'][k]['median_tps']/summary['b4']['cases'][k]['median_tps'] for k in ['fixed32','fixed128','mixed32']}
a.output.write_text(json.dumps(summary,indent=2));print(json.dumps({k:({x:y for x,y in v.items() if x in ['cases','requests','median_dma_ms']} if k.startswith('b') else v) for k,v in summary.items()},indent=2))
