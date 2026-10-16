"""Compare completed, replay-validated runs with equal request/card budgets."""
import argparse,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('baseline',type=Path);ap.add_argument('candidate',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
runs=[]
for directory in (a.baseline,a.candidate):
 cfg=json.loads((directory/'config.json').read_text())
 http=json.loads((directory/'http-results.json').read_text())
 audit=json.loads((directory/'analysis.json').read_text())
 terminal=json.loads((directory/'result.json').read_text())
 assert http['passed'] and terminal['passed'] and audit['exact_logits'] and audit['ownership_violations']==0
 runs.append((cfg,http,audit))
for key in ('model','tables','workload','atom_devices','rngd_device_path','rngd_logical_index','lookahead','collection_ms','batch_wait_ms'):
 assert runs[0][0].get(key)==runs[1][0].get(key),f'confounded configuration: {key}'
rows=[]
for policy in ('sealed','refill'):
 for case in ('fixed32','fixed128','mixed32'):
  samples=[]
  for cfg,http,audit in runs:
   selected=[r for r in http['rows'] if r['scenario'].startswith(f'{policy}-{case}-r')]
   assert len(selected)==3 and all(len(r['requests'])==32 for r in selected)
   wanted=[32]*32 if case=='fixed32' else [128]*32 if case=='fixed128' else [32,64,96,128]*8
   assert all(sorted(q['output_tokens'] for q in r['requests'])==sorted(wanted) for r in selected)
   values=[r['tokens_per_second'] for r in selected]
   samples.append(dict(batch=cfg['batch_size'],median=statistics.median(values),repeats=values))
  rows.append(dict(policy=policy,case=case,baseline=samples[0],candidate=samples[1],ratio=samples[1]['median']/samples[0]['median']))
a.output.write_text(json.dumps(dict(scope='HTTP 32-request scenario throughput; sequential shared host; same-batch transport parity, not application quality or MFU',baseline=str(a.baseline),candidate=str(a.candidate),rows=rows),indent=2))
