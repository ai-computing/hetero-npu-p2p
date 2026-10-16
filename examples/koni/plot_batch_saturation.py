"""Plot separate fixed-output128 scenarios; never pool32 and48 request runs."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--docs',type=Path,required=True);a=ap.parse_args()
fig,ax=plt.subplots(figsize=(8,4.5))
for off,which,color in [(-.19,'baseline','#718599'),(.19,'candidate','#169a85')]:
 rows=[]
 for name in ['b8-b12-comparison.json','b8-b12-saturation48.json']:
  data=json.loads((a.docs/name).read_text())
  rows.append(next(r[which] for r in data['rows'] if r['case']=='fixed128' and r['policy']=='sealed'))
 v=[r['median'] for r in rows];low=[r['median']-min(r['repeats']) for r in rows];high=[max(r['repeats'])-r['median'] for r in rows]
 bars=ax.bar([i+off for i in range(2)],v,.36,yerr=[low,high],capsize=4,label=f"B{rows[0]['batch']}",color=color);ax.bar_label(bars,fmt='%.2f',padding=4)
ax.set_xticks([0,1],['32 requests','48 requests']);ax.set_ylim(0,150);ax.set_ylabel('HTTP output tokens / second');ax.set_title('KONI | 1 RNGD + 1 ATOM | output128 | sealed batches');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.text(.5,.015,'3-repeat medians; bars show min–max. Separate sequential shared-host scenarios.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(a.docs/'b8-b12-saturation.png',dpi=150)
