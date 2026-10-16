"""Summarize all four completed projection runs, preserving process-level blocks."""
import argparse,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('runs',nargs=4,type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
configs=[];rows=[]
for order,(path,graph) in enumerate(zip(a.runs,['full','grouped','grouped','full']),1):
 cfg=json.loads((path/'config.json').read_text());http=json.loads((path/'http-results.json').read_text());audit=json.loads((path/'analysis.json').read_text());terminal=json.loads((path/'result.json').read_text())
 assert cfg['experiment_design']==dict(order=order,graph=graph,warm_output=128,requests=48)
 assert http['passed'] and terminal['passed'] and audit['exact_logits'] and audit['ownership_violations']==0 and audit['requests']==224
 warm=[r for r in http['rows'] if r['scenario']=='warm'];assert len(warm)==1 and len(warm[0]['requests'])==48
 assert all(q['output_tokens']==128 for q in warm[0]['requests'])
 measured=[r for r in http['rows'] if r['scenario'].startswith('sealed-fixed128-r')]
 assert {r['scenario'] for r in measured}=={f'sealed-fixed128-r{i}' for i in range(3)}
 assert all(len(r['requests'])==48 and all(q['output_tokens']==128 for q in r['requests']) for r in measured)
 assert audit['cases']['fixed128']['sealed']['useful_lane_fraction']==1.0
 configs.append(cfg);v=[r['tokens_per_second'] for r in measured]
 rows.append(dict(order=order,graph=graph,path=str(path),repeats=v,median_tps=statistics.median(v),worker_seconds=audit['cases']['fixed128']['sealed']['device_seconds']))
for cfg in configs[1:]:
 for key in ('model','tables','workload','atom_devices','rngd_device_path','rngd_logical_index','batch_size','lookahead','collection_ms','batch_wait_ms'):
  assert cfg.get(key)==configs[0].get(key),f'confounded field: {key}'
assert configs[0]['atom_artifact']==configs[3]['atom_artifact']
assert configs[1]['atom_artifact']==configs[2]['atom_artifact']
assert configs[0]['atom_artifact']!=configs[1]['atom_artifact']
pairs=[rows[0]['median_tps']/rows[1]['median_tps'],rows[3]['median_tps']/rows[2]['median_tps']]
summary=dict(scope='ABBA sequential shared host, two process blocks per graph; no independence or significance claim',rows=rows,paired_full_over_grouped=pairs,all_pairs_faster=all(r>1 for r in pairs),requests=sum(224 for _ in rows),process_median_by_graph={g:statistics.median(r['median_tps'] for r in rows if r['graph']==g) for g in ('full','grouped')})
a.output.write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
