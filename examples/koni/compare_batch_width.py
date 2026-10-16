"""Compare completed, replay-validated runs with equal request/card budgets."""
import argparse,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('baseline',type=Path);ap.add_argument('candidate',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--saturation',action='store_true');ap.add_argument('--mixed-only',action='store_true');ap.add_argument('--requests',type=int,choices=(32,48,84));a=ap.parse_args();n=a.requests or (48 if a.saturation else 32)
if (a.saturation and a.mixed_only) or (a.saturation and n not in (48,84)) or (not (a.saturation or a.mixed_only) and n!=32):raise ValueError('request count incompatible with design')
runs=[]
for directory in (a.baseline,a.candidate):
 cfg=json.loads((directory/'config.json').read_text())
 http=json.loads((directory/'http-results.json').read_text())
 audit=json.loads((directory/'analysis.json').read_text())
 terminal=json.loads((directory/'result.json').read_text())
 assert http['passed'] and terminal['passed'] and audit['exact_logits'] and audit['ownership_violations']==0
 runs.append((cfg,http,audit))
for key in ('model','tables','workload','atom_devices','rngd_device_path','rngd_logical_index','lookahead','collection_ms','batch_wait_ms','queue_capacity'):
 assert runs[0][0].get(key)==runs[1][0].get(key),f'confounded configuration: {key}'
warm_outputs=[]
for cfg,http,audit in runs:
 warm=[r for r in http['rows'] if r['scenario']=='warm'];assert len(warm)==1
 warm_outputs.append(sorted(q['output_tokens'] for q in warm[0]['requests']))
assert warm_outputs[0]==warm_outputs[1] and len(warm_outputs[0])==n,'warmup workloads differ or request budget mismatches'
rows=[]
for policy in (('sealed',) if a.saturation else ('sealed','refill')):
 for case in (('fixed128',) if a.saturation else ('mixed',) if a.mixed_only else ('fixed32','fixed128','mixed32')):
  samples=[]
  for cfg,http,audit in runs:
   selected=[r for r in http['rows'] if r['scenario'].startswith(f'{policy}-{case}-r')]
   assert {r['scenario'] for r in selected}=={f'{policy}-{case}-r{i}' for i in range(3)}
   assert len(selected)==3 and all(len(r['requests'])==n for r in selected)
   wanted=[32]*n if case=='fixed32' else [128]*n if case=='fixed128' else [32,64,96,128]*(n//4)
   assert all(sorted(q['output_tokens'] for q in r['requests'])==sorted(wanted) for r in selected)
   values=[r['tokens_per_second'] for r in selected]
   samples.append(dict(batch=cfg['batch_size'],context=cfg.get('context',512),artifact=cfg['atom_artifact'],median=statistics.median(values),repeats=values))
  rows.append(dict(policy=policy,case=case,baseline=samples[0],candidate=samples[1],ratio=samples[1]['median']/samples[0]['median']))
a.output.write_text(json.dumps(dict(scope=f'HTTP {n}-request scenario throughput; sequential shared host; same-batch transport parity, not application quality or MFU',baseline=str(a.baseline),candidate=str(a.candidate),rows=rows),indent=2))
