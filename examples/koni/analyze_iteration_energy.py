"""Allocated-board energy for exact HTTP measurement windows; no peak/MFU claim."""
import argparse,json
from pathlib import Path
from hetero_p2p.energy import integrate_power

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=json.loads((a.run/'config.json').read_text());http=json.loads((a.run/'http-results.json').read_text());audit=json.loads((a.run/'analysis.json').read_text())
assert http['passed'] and audit['exact_logits'] and audit['ownership_violations']==0
sensors=['rngd']+[f'atom-{device}' for device in cfg['atom_devices']]
raw={key:[json.loads(line) for line in (a.run/f'power-{key}.jsonl').read_text().splitlines()] for key in sensors}
results=[]
for row in http['rows']:
    if row['scenario'] in ('probe','warm'):continue
    try:
        if not row.get('power_sampled'):raise ValueError('no power measurement for this scenario')
        start,end=row['window_start'],row['window_end']
        if abs((end-start)-row['seconds'])>1e-6:raise ValueError('inconsistent HTTP window')
        joules={key:integrate_power(raw[key],start,end) for key in sensors}
        total=sum(joules.values())
        if total<=0:raise ValueError('nonpositive board energy')
        results.append(dict(scenario=row['scenario'],valid=True,seconds=end-start,
            boards={key:dict(joules=value,mean_watts=value/(end-start)) for key,value in joules.items()},
            total_joules=total,output_tokens_per_joule=sum(q['output_tokens'] for q in row['requests'])/total))
    except (ValueError,KeyError) as error:
        results.append(dict(scenario=row['scenario'],valid=False,error=str(error)))
a.output.write_text(json.dumps(dict(scope='SMI board sensor samples, linear midpoint integration over HTTP windows; allocated-board idle included, host CPU/DRAM excluded; not wall-plug power, stage FLOPs or MFU',rows=results),indent=2))
print(json.dumps(results,indent=2))
