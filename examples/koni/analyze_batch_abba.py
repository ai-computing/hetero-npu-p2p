"""Audit B12/B14/B14/B12 process blocks; never treat repetitions as independent."""
import argparse,json,statistics
from pathlib import Path

def analyze(paths):
 if len(paths)!=4 or len({Path(p).resolve() for p in paths})!=4:
  raise ValueError('four distinct process runs required')
 configs=[];rows=[];windows=[]
 for order,(path,width) in enumerate(zip(map(Path,paths),(12,14,14,12)),1):
  cfg=json.loads((path/'config.json').read_text());audit=json.loads((path/'analysis.json').read_text())
  http=json.loads((path/'http-results.json').read_text());terminal=json.loads((path/'result.json').read_text())
  assert cfg['batch_size']==width and cfg.get('context',512)==(512 if width==12 else 448)
  assert http['passed'] and terminal['passed'] and audit['exact_logits'] and audit['ownership_violations']==0 and audit['requests']==368
  assert audit['cases']['fixed128']['sealed']['useful_lane_fraction']==1.0
  assert len(http['rows'])==5
  warm=[r for r in http['rows'] if r['scenario']=='warm'];assert len(warm)==1
  measured=[r for r in http['rows'] if r['scenario'].startswith('sealed-fixed128-r')]
  assert {r['scenario'] for r in measured}=={f'sealed-fixed128-r{i}' for i in range(3)}
  for r in warm+measured:
   assert len(r['requests'])==84 and all(q['output_tokens']==128 for q in r['requests'])
  events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
  windows.append((min(e['time'] for e in events),max(e['time'] for e in events)))
  copies=[e for e in events if e['stage']=='p2p_complete'];assert len(copies)==368 and all(e['requested']==e['completed']==64 and e['cqe_result']==0 for e in copies)
  vals=[r['tokens_per_second'] for r in measured]
  assert statistics.median(vals)==audit['cases']['fixed128']['sealed']['median_tps']
  configs.append(cfg);rows.append(dict(order=order,batch=width,context=cfg.get('context',512),path=str(path),repeats=vals,median_tps=statistics.median(vals)))
 for i in range(3):assert windows[i][1]<windows[i+1][0],'process ordering/overlap mismatch; same host boot required'
 for cfg in configs[1:]:
  for key in ('model','tables','workload','atom_devices','rngd_device_path','rngd_logical_index','lookahead','collection_ms','batch_wait_ms','queue_capacity'):
   assert cfg.get(key)==configs[0].get(key),f'confounded field:{key}'
 assert configs[0]['atom_artifact']==configs[3]['atom_artifact']
 assert configs[1]['atom_artifact']==configs[2]['atom_artifact']
 ratios=[rows[1]['median_tps']/rows[0]['median_tps'],rows[2]['median_tps']/rows[3]['median_tps']]
 return dict(scope='same-host ABBA; two process blocks per configuration; batch and context change together; not significance or quality proof',requests=1472,rows=rows,paired_b14_over_b12=ratios,all_pairs_faster=all(x>1 for x in ratios),process_median_by_batch={str(b):statistics.median(r['median_tps'] for r in rows if r['batch']==b) for b in (12,14)})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('runs',nargs=4,type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 result=analyze(a.runs);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
