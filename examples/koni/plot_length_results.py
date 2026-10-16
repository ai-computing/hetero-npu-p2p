import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('summary',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();data=json.loads(a.summary.read_text());names=['fifo','early','length'];labels=['FIFO','FIFO + early reply','Length + early reply'];colors=['#7c8996','#42a5ad','#2371b5']
fig,axes=plt.subplots(1,2,figsize=(11,4.4));values=[data['variants'][n]['median_tokens_per_second'] for n in names];lo=[v-min(data['variants'][n]['tokens_per_second']) for n,v in zip(names,values)];hi=[max(data['variants'][n]['tokens_per_second'])-v for n,v in zip(names,values)];bars=axes[0].bar(labels,values,color=colors,yerr=[lo,hi],capsize=4);axes[0].bar_label(bars,fmt='%.2f',padding=7);axes[0].set_ylim(0,max(values)*1.3);axes[0].set_ylabel('Output tokens / second');axes[0].set_title('Mixed-length burst: 32 requests\nMedian and range of 3 repeats')
for n,label,color in zip(names,labels,colors):axes[1].plot([32,64,96,128],[data['variants'][n]['median_latency_by_output'][str(k)] for k in [32,64,96,128]],'o-',label=label,color=color)
axes[1].set_xlabel('Requested output tokens');axes[1].set_ylabel('Median HTTP response time (seconds)');axes[1].legend();axes[1].set_title('Latency by output length')
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(a.output,dpi=160)
