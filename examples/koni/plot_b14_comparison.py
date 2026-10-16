"""Plot completed 48-request runs without mixing decode-only measurements."""
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
series=[]
for root in (a.baseline,a.candidate):
 cfg=json.loads((root/'config.json').read_text());audit=json.loads((root/'analysis.json').read_text());http=json.loads((root/'http-results.json').read_text())
 assert json.loads((root/'result.json').read_text())['passed'] and http['passed'] and audit['exact_logits'] and audit['ownership_violations']==0
 rows=[r for r in http['rows'] if r['scenario'].startswith('sealed-fixed128-r')]
 assert len(rows)==3 and all(len(r['requests'])==48 and all(q['output_tokens']==128 for q in r['requests']) for r in rows)
 warm=[r for r in http['rows'] if r['scenario']=='warm'];assert len(warm)==1 and len(warm[0]['requests'])==48 and all(q['output_tokens']==128 for q in warm[0]['requests'])
 series.append((cfg,audit,[r['tokens_per_second'] for r in rows]))
for key in ('model','workload','atom_devices','rngd_device_path','lookahead','batch_wait_ms'):
 assert series[0][0].get(key)==series[1][0].get(key),key
fig,ax=plt.subplots(1,2,figsize=(9,4),layout='constrained')
labels=[]
for i,(cfg,audit,vals) in enumerate(series):
 label=f"B{cfg['batch_size']} / context {cfg.get('context',512)}";labels.append(label)
 ax[0].scatter([i-.08,i,i+.08],vals,s=40)
 med=statistics.median(vals);ax[0].plot([i-.23,i+.23],[med,med],color='black');ax[0].annotate(f'{med:.2f}',(i,med),xytext=(0,12),textcoords='offset points',ha='center')
 lane=100*audit['cases']['fixed128']['sealed']['useful_lane_fraction'];ax[1].bar(i,lane,width=.5);ax[1].text(i,lane+2,f'{lane:.1f}%',ha='center')
for axis in ax:axis.set_xticks(range(2),labels);axis.grid(axis='y',alpha=.2)
ax[0].set_ylabel('HTTP output tokens/s');ax[0].set_ylim(0,160);ax[0].set_title('All three repeats; line = median')
ax[1].set_ylabel('Useful decode lanes (%)');ax[1].set_ylim(0,115);ax[1].set_title('Measured lane occupancy (not MFU)')
fig.suptitle('1 RNGD + 1 ATOM | 48 requests x 128 output tokens\nSequential runs; batch and context both change')
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180)
