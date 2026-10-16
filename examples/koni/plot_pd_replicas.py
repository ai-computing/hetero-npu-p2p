"""Plot one/two ATOM PD summaries, preserving all three repeat ranges."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('summary',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
s=json.loads(args.summary.read_text());keys=['fixed32','fixed128','mixed32'];fig,ax=plt.subplots(figsize=(8.5,4.5))
for off,key,label,color in [(-.19,'one_atom','1 RNGD + 1 ATOM','#597f9c'),(.19,'two_atom','1 RNGD + 2 ATOM','#1d9b85')]:
 cs=s[key]['cases'];v=[cs[k]['median_tps'] for k in keys];lo=[x-min(cs[k]['tps_repeats']) for x,k in zip(v,keys)];hi=[max(cs[k]['tps_repeats'])-x for x,k in zip(v,keys)];bars=ax.bar([i+off for i in range(3)],v,.36,yerr=[lo,hi],capsize=3,label=label,color=color);ax.bar_label(bars,fmt='%.1f',padding=5)
ax.set_xticks(range(3),['16 requests / out32','16 requests / out128','32 requests / mixed']);ax.set_ylabel('Output tokens / second');ax.set_title('KONI B8 P2P PD through llm-d HTTP\nMedian and range of 3 repeats; sequential shared-host measurements');ax.set_ylim(0,215);ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(args.output,dpi=150)
