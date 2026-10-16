import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('summary',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();s=json.loads(a.summary.read_text())['cases'];fig,axes=plt.subplots(1,2,figsize=(10,4.5));policies=['fixed128','fixed4','adaptive'];colors=['#718599','#b89b50','#169a85']
for off,case,label in [(-.2,'burst128','Burst: 16 x 128'),(.2,'staggered','Delayed: 128 + 15 x 32')]:
 v=[s[case][p]['median_tps'] for p in policies];lo=[v[i]-min(s[case][p]['tps_repeats']) for i,p in enumerate(policies)];hi=[max(s[case][p]['tps_repeats'])-v[i] for i,p in enumerate(policies)];bars=axes[0].bar([i+off for i in range(3)],v,.38,yerr=[lo,hi],capsize=3,label=label);axes[0].bar_label(bars,fmt='%.1f',padding=3)
for off,n,label in [(-.2,'32','Late short requests'),(.2,'128','Initial long request')]:
 v=[s['staggered'][p]['response_by_output'][n] for p in policies];bars=axes[1].bar([i+off for i in range(3)],v,.38,label=label);axes[1].bar_label(bars,fmt='%.2f',padding=3)
for ax in axes:
 ax.set_xticks(range(3),['Fixed128','Fixed4','Adaptive']);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Output tokens / scenario second');axes[0].set_ylim(0,120);axes[1].set_ylabel('Median HTTP response seconds');axes[1].set_ylim(0,15)
fig.suptitle('KONI B8 | 1 RNGD + 1 ATOM | three rotated repeats\nAdaptive: 4-step partial batches, 128-step full batches');fig.tight_layout();fig.savefig(a.output,dpi=150)
