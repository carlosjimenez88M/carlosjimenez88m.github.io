"""Figures from measured trajectories; no invented frontier or uncertainty bands."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/blog-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd,numpy as np
from prepare import *
DEST=ROOT/'static/img/album-memory-v2';DEST.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f6','axes.facecolor':'#faf9f6'})
COLORS=dict(full='#283f55',recent='#bb6b45',summary='#848a72',tfidf='#448b8b',embedding='#b19238',relational='#755b93',adaptive='#b44662')
def save(fig,name):
 for ext in ['svg','png']:fig.savefig(DEST/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
def main():
 p=pd.read_csv(OUT/'policy_means.csv');p=p[(p.policy!='adaptive')|(p.threshold==3)]
 fig,ax=plt.subplots(figsize=(10,6));fig.subplots_adjust(bottom=.2)
 for policy,g in p.groupby('policy'):
  ax.scatter(g.mean_total_tokens,g.quality,s=55+g.mean_receipt_latency_s*2,color=COLORS[policy],alpha=.85,label=policy)
 frontier=p[p.pareto_primary].sort_values('mean_total_tokens')
 ax.plot(frontier.mean_total_tokens,frontier.quality,color='#333333',linestyle='--',linewidth=1.2,zorder=0)
 ax.set(xlabel='Mean input + output tokens across the entire trajectory',ylabel='Positive answers accepted by the model verifier (fraction)',ylim=(-.03,1.03))
 ax.set_title('The observed quality–token frontier',loc='left',fontsize=16);ax.legend(frameon=False,ncol=2);ax.grid(alpha=.15)
 fig.text(.12,.02,'Bubble area grows with summed provider-call latency (cached receipts included).\nDescriptive frontier: 20 positive questions, four albums, one response per setting; no human gold labels.',fontsize=9)
 save(fig,'pareto-frontier')
 fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for policy in ['tfidf','embedding','relational']:
  g=p[p.policy==policy].sort_values('cap')
  for ax,col in zip(axes,['availability','utilization']):ax.plot(g.cap,g[col],'-o',label=policy,color=COLORS[policy])
 for ax,title in zip(axes,['Designated evidence exposed','Cited given exposure']):
  ax.set(xlabel='Exposed-memory token cap',ylabel='Fraction',ylim=(-.03,1.03));ax.set_title(title,loc='left');ax.grid(alpha=.15)
 axes[0].legend(frameon=False);fig.suptitle('Availability and utilization answer different questions',fontsize=15)
 fig.text(.05,-.05,'Utilization is conditional on selected target items; it is not causal attention. Undefined denominators are omitted.',fontsize=9)
 save(fig,'budget-utilization')
 fig,axes=plt.subplots(4,1,figsize=(12,12),layout='constrained')
 palette={'return':'#448b8b','contrast':'#bb6b45','transformation':'#755b93','counterpoint':'#b19238'}
 for ax,a in zip(axes,ALBUMS):
  edges=read(OUT/f"relations-{a['id']}.json");n=len(a['tracks'])
  ax.scatter(range(1,n+1),[0]*n,color='#283f55',s=35,zorder=3)
  for i,e in enumerate(edges):
   x,y=[int(v[1:3]) for v in e['evidence']];x,y=sorted([x,y])
   rad=.1+(i%3)*.06
   ax.add_patch(FancyArrowPatch((x,0),(y,0),connectionstyle=f'arc3,rad=-{rad}',arrowstyle='-',color=palette[e['type']],alpha=.7,linewidth=1.5))
  ax.set(xlim=(.5,n+.5),ylim=(-.5,4.5),xticks=range(1,n+1),yticks=[])
  ax.set_title(f"{a['artist']} · {a['album']}",loc='left');ax.set_xlabel('Track position; edges are hypotheses with claim references')
  ax.spines['left'].set_visible(False);ax.spines['bottom'].set_visible(False)
 fig.suptitle('Narrative relation memory before any question was evaluated',fontsize=16)
 from matplotlib.lines import Line2D
 fig.legend(handles=[Line2D([0],[0],color=c,label=t) for t,c in palette.items()],loc='lower center',ncol=4,bbox_to_anchor=(.5,-.03),frameon=False)
 save(fig,'relation-memory')
 # Distance is a task/corpus covariate, not a randomized position intervention.
 rows=read(OUT/'runs.json');points=[]
 for r in rows:
  if r['policy']=='adaptive' and r['threshold']!=3:continue
  pos=[int(s[1:]) for s in r['question']['target_ids']]
  if len(pos)>=2:points.append(dict(policy=r['policy'],distance=max(pos)-min(pos),success=r['metrics']['model_verified_supported']))
 d=pd.DataFrame(points);fig,ax=plt.subplots(figsize=(10,5));fig.subplots_adjust(bottom=.23)
 for policy in ['full','tfidf','embedding','relational','adaptive']:
  g=d[d.policy==policy].groupby('distance').success.mean();ax.plot(g.index,g.values,'-o',color=COLORS[policy],label=policy)
 ax.set(xlabel='Distance between first and last designated evidence positions',ylabel='Model-verified supported rate',ylim=(-.05,1.05));ax.set_title('A descriptive distance breakdown, not a needle-position experiment',loc='left',fontsize=13);ax.legend(ncol=3,frameon=False);ax.grid(alpha=.15)
 fig.text(.12,.02,'Task type, album, and distance are confounded. Retrieval policies pool the five budgets; full and adaptive are single settings.\nSmall strata may contain one question. These lines do not identify a causal effect of distance.',fontsize=9)
 save(fig,'distance-breakdown')
def economics():
 rows=read(OUT/'runs.json')
 choices=[('full',100000),('tfidf',800),('embedding',1600),('adaptive',100000)]
 fig,ax=plt.subplots(figsize=(10,5));bottom=np.zeros(len(choices))
 for stage,color in [('sufficiency','#b19238'),('answer','#448b8b'),('verification','#755b93')]:
  means=[]
  for policy,cap in choices:
   group=[r for r in rows if r['policy']==policy and r['cap']==cap and r['threshold']==3]
   means.append(sum(r['stages'].get(stage,{}).get('input_tokens',0)+r['stages'].get(stage,{}).get('output_tokens',0) for r in group)/len(group))
  ax.bar(range(len(choices)),means,bottom=bottom,label=stage,color=color);bottom+=np.array(means)
 ax.set_xticks(range(len(choices)),['Full','TF-IDF 800','Embedding 1600','Adaptive relational'])
 ax.set_ylabel('Mean input + output tokens per complete trajectory');ax.set_title('The controller and verifier belong in the bill',loc='left',fontsize=15);ax.legend(frameon=False)
 save(fig,'trajectory-economics')
 old=pd.read_csv(OLD/'policy_means.csv');full=float(old[old.policy=='full'].input_tokens.iloc[0]);summary=float(old[old.policy=='summary'].input_tokens.iloc[0]);upfront=read(OLD/'usage_by_stage.json')['summary']['input_tokens']
 q=np.arange(0,31);cross=upfront/(full-summary)
 fig,ax=plt.subplots(figsize=(10,5));fig.subplots_adjust(bottom=.23)
 ax.plot(q,q*full,label='Full-card answer input',color=COLORS['full']);ax.plot(q,upfront+q*summary,label='Build four summaries + answer input',color=COLORS['summary'])
 ax.axvline(cross,linestyle='--',color='#888');ax.text(cross+.4,2000,f'{cross:.1f} total queries',fontsize=9)
 ax.set(xlabel='Total queries across a balanced four-album workload',ylabel='Cumulative input tokens');ax.set_title('Part I: token break-even does not restore evidence',loc='left',fontsize=15);ax.legend(frameon=False)
 fig.text(.12,.02,'Assumes all four summaries are built first. Excludes common annotation, output tokens, pricing, and evaluation.\nAbout 2.3 queries per album on average; the summary still had only 5.6% designated evidence recall.',fontsize=9)
 save(fig,'amortization')

if __name__=='__main__':
 main();economics()
