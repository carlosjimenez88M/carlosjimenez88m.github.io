"""Descriptive editorial figures from frozen results; no new model calls."""
from figures import *
def main():
 rows=read(OUT/'runs.json')
 tasks=['local','distant','transformation','contradiction','multi_hop','abstention']
 labels=['Local','Distant','Transform.','Contradiction','Multi-hop','Abstention']
 settings=[('Full','full',100000),('Embedding 1600','embedding',1600),('Relational 2400','relational',2400),('Adaptive t3','adaptive',100000)]
 scores=[];costs=[];export=[]
 for label,policy,cap in settings:
  ss=[];cc=[]
  for task in tasks:
   g=[r for r in rows if r['policy']==policy and r['cap']==cap and r['threshold']==3 and r['task']==task];assert len(g)==4
   key='correct_negative_abstention' if task=='abstention' else 'model_verified_supported'
   n=sum(r['metrics'][key] for r in g);cost=sum(r['metrics']['input_tokens']+r['metrics']['output_tokens'] for r in g)/4
   ss.append(n);cc.append(cost);export.append(dict(setting=label,task=task,accepted=n,total=4,mean_trajectory_tokens=cost))
  scores.append(ss);costs.append(cc)
 pd.DataFrame(export).to_csv(OUT/'revision_task_table.csv',index=False)
 fig,axes=plt.subplots(2,1,figsize=(11,7.8),layout='constrained')
 for ax,data,title,cmap in zip(axes,[scores,costs],['Online acceptance by task — four questions per cell','Mean trajectory tokens by task — input + output'],['YlGnBu','YlOrBr']):
  ax.imshow(data,cmap=cmap,aspect='auto');ax.set_xticks(range(6),labels);ax.set_yticks(range(4),[s[0] for s in settings]);ax.set_title(title,loc='left',pad=14)
  for i in range(4):
   for j in range(6):
    v=data[i][j];text=f'{int(v)}/4' if data is scores else f'{v:,.0f}'
    ax.text(j,i,text,ha='center',va='center',color='white' if v>np.max(data)*.72 else '#202522',fontsize=12)
 fig.supxlabel('Abstention uses correct negative refusals; other columns use positive online acceptance.\nDifferent endpoints, no human correctness labels. A one-answer change moves a cell by 25 points.',fontsize=10)
 save(fig,'task-acceptance-tokens')
 sessions=read(OUT/'sessions.json');values=[]
 for p in ['full','adaptive']:
  g=[r for r in sessions if r['policy']==p];assert len(g)==16
  values.append((sum(r['metrics']['supported'] for r in g),sum(r['metrics']['input_tokens']+r['metrics']['output_tokens'] for r in g)/16))
 fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
 for ax,ix,title in zip(axes,[0,1],['Accepted conversation turns / 16','Mean trajectory tokens per turn']):
  v=[x[ix] for x in values];ax.bar(['Full','Adaptive'],v,color=[COLORS['full'],COLORS['adaptive']]);ax.set_title(title,loc='left');ax.set_ylim(0,max(v)*1.23)
  for j,x in enumerate(v):ax.text(j,x+max(v)*.035,f'{x:,.0f}'+('/16' if ix==0 else ''),ha='center')
 fig.supxlabel('Four conversations per policy, four linked turns each. Same 250-token history cap.\nDescriptive diagnostic; no embedding conversation control or no-history ablation.',fontsize=10)
 save(fig,'conversation-tradeoff')
 r=next(r for r in rows if r['case_id']=='roach-distant-adaptive-100000-t3')
 receipts=r['receipts'];roundnum=-1;trace=[]
 for c in receipts:
  if c['kind']=='sufficiency':roundnum+=1
  trace.append(dict(cap=r['rounds'][roundnum]['cap'],stage=c['kind'],tokens=c['usage']['prompt_tokens']+c['usage']['completion_tokens']))
 assert len(trace)==13 and sum(x['tokens'] for x in trace)==19831
 write(OUT/'infest_receipt_sequence.json',dict(case_id=r['case_id'],mlflow_run_id=r['mlflow_run_id'],source='Frozen ordered provider receipts, not an MLflow UI screenshot or timing chart',calls=trace))
 fig,ax=plt.subplots(figsize=(11,7),layout='constrained');y=np.arange(len(trace))
 colors={'sufficiency':'#b19238','answer':'#448b8b','verification':'#755b93'}
 ax.barh(y,[x['tokens'] for x in trace],color=[colors[x['stage']] for x in trace]);ax.set_yticks(y,[f"{i+1:02} · cap {x['cap']:,} · {x['stage']}" for i,x in enumerate(trace)]);ax.invert_yaxis();ax.set_xlim(0,3400)
 for i,x in enumerate(trace):ax.text(x['tokens']+35,i,f"{x['tokens']:,}",va='center',fontsize=10)
 ax.set_xlabel('Provider input + output tokens for this call');ax.set_title('Infest: 13 calls, 19,831 trajectory tokens',loc='left',fontsize=16)
 fig.supxlabel('Receipt reconstruction, not a UI screenshot. Final exposed memory: 2,326 tokens.\nRecorded online acceptance contains a known T01 / T04 attribution error.',fontsize=10)
 save(fig,'infest-receipt-sequence')
 print(pd.DataFrame(export).to_string(index=False));print('Conversations:',values)
if __name__=='__main__':main()
