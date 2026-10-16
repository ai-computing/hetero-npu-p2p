"""Plot all throughput repetitions and request-response medians from audited runs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
data=[]
for path in (a.baseline,a.candidate):
    d=json.loads((path/'analysis.json').read_text())
    assert d['exact_logits'] and d['ownership_violations']==0
    assert json.loads((path/'result.json').read_text())['passed']
    data.append(d)
fig,(ax,lat)=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
labels=[]
for index,(d,policy) in enumerate((d,p) for d in data for p in ('sealed','refill')):
    row=d['cases']['mixed'][policy];cfg=d['config']
    label=f"B{cfg['batch_size']} S{cfg.get('context',512)}\n{cfg.get('queue_policy','length')} / {policy}";labels.append(label)
    color=f'C{index}'
    ax.scatter([index-.08,index,index+.08],row['tps_repeats'],color=color)
    ax.plot([index-.25,index+.25],[row['median_tps']]*2,color=color)
    ax.text(index,row['median_tps']+4,f"{row['median_tps']:.2f}",ha='center')
    lat.plot([32,64,96,128],[row['response_by_output'][str(n)] for n in (32,64,96,128)],marker='o',color=color,label=label.replace('\n',' '))
ax.set_xticks(range(4),labels);ax.set_ylim(0,170);ax.set_ylabel('HTTP output tokens/s');ax.set_title('All 3 repetitions; horizontal line = median')
lat.set_xticks([32,64,96,128]);lat.set_ylim(bottom=0);lat.set_xlabel('Output tokens');lat.set_ylabel('Median request response (seconds)');lat.legend(fontsize=8);lat.set_title('Non-streaming response time')
for axis in (ax,lat):axis.grid(axis='y',alpha=.2)
fig.suptitle('1 RNGD + 1 ATOM | 84 mixed requests\nSequential process comparison; batch/context change together when comparing B12/B14')
fig.savefig(a.output,dpi=170)
