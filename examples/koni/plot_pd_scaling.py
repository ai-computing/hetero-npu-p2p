import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('summary',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();s=json.loads(a.summary.read_text());keys=['fixed32','fixed128','mixed32'];labels=['16 requests\n32 output tokens','16 requests\n128 output tokens','32 requests\nmixed outputs'];fig,ax=plt.subplots(figsize=(9,4.8))
for offset,batch,color in [(-.19,'b4','#7b929e'),(.19,'b8','#2371b5')]:
 values=[s[batch]['cases'][k]['median_tps'] for k in keys];lo=[v-min(s[batch]['cases'][k]['tps_repeats']) for k,v in zip(keys,values)];hi=[max(s[batch]['cases'][k]['tps_repeats'])-v for k,v in zip(keys,values)];bars=ax.bar([i+offset for i in range(3)],values,.36,yerr=[lo,hi],capsize=4,label=batch.upper(),color=color);ax.bar_label(bars,fmt='%.1f',padding=6)
ax.set_xticks(range(3),labels);ax.set_ylabel('Output tokens / second');ax.set_ylim(0,max(s['b8']['cases'][k]['median_tps'] for k in keys)*1.25);ax.set_title('KONI heterogeneous PD via llm-d HTTP\nSame RNGD x1 + ATOM x2 | lookahead16 | median and range of 3 repeats');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(a.output,dpi=160)
