"""Render every measured repetition in an audited batch ABBA summary."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.summary.read_text());assert d['requests']==1472 and len(d['rows'])==4
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
for i,r in enumerate(d['rows']):
 color='tab:blue' if r['batch']==12 else 'tab:orange'
 ax.scatter([i-.08,i,i+.08],r['repeats'],color=color,s=45)
 ax.plot([i-.24,i+.24],[r['median_tps']]*2,color=color,lw=2)
 ax.text(i,r['median_tps']+4,f"{r['median_tps']:.2f}",ha='center')
ax.set_xticks(range(4),[f"{r['order']}: B{r['batch']} / S{r['context']}" for r in d['rows']]);ax.set_ylim(0,180);ax.set_ylabel('HTTP output tokens/s');ax.grid(axis='y',alpha=.2)
ax.set_title('1 RNGD + 1 ATOM | 84 requests x 128 output tokens\nAll repeats, process medians; batch and context change together')
fig.savefig(a.output,dpi=180)
