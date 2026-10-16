import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('summary',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();raw=json.loads(a.summary.read_text());s=raw['cases'];atoms=len(raw['config']['atom_devices']);fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for off,key,label,color in [(-.19,'sealed','Wait for batch end','#718599'),(.19,'refill','Refill at step fence','#169a85')]:
 keys=['fixed32','fixed128','mixed32'];v=[s[k][key]['median_tps'] for k in keys];lo=[v[i]-min(s[k][key]['tps_repeats']) for i,k in enumerate(keys)];hi=[max(s[k][key]['tps_repeats'])-v[i] for i,k in enumerate(keys)]
 bars=axes[0].bar([i+off for i in range(3)],v,.36,yerr=[lo,hi],capsize=3,label=label,color=color);axes[0].bar_label(bars,fmt='%.1f',padding=3)
 v=[s['mixed32'][key]['response_by_output'][str(n)] for n in [32,64,96,128]];bars=axes[1].bar([i+off for i in range(4)],v,.36,color=color);axes[1].bar_label(bars,fmt='%.1f',padding=3)
axes[0].set_xticks(range(3),['Fixed32','Fixed128','Mixed']);axes[0].set_ylabel('Output tokens / second');axes[0].set_ylim(0,1.25*max(c[k]['median_tps'] for c in s.values() for k in ['sealed','refill']));axes[0].legend(fontsize=8)
axes[1].set_xticks(range(4),['32','64','96','128']);axes[1].set_xlabel('Output tokens per request (mixed workload)');axes[1].set_ylabel('Median HTTP response seconds');axes[1].set_ylim(0,30)
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle(f"KONI B{raw['config'].get('batch_size',8)} | 1 RNGD + {atoms} ATOM | 3 counterbalanced repeats\nThroughput and HTTP response-time tradeoffs");fig.tight_layout();fig.savefig(a.output,dpi=150)
