"""Descriptive, denominator-explicit analysis; no claim of human validation."""
from collections import defaultdict
import math,re
import pandas as pd
from prepare import *
def explicit_track_alignment(r):
 return all(set(re.findall(r'\bT\d{2}\b',c['text']))<={s.split(':')[0] for s in c['sources']} for c in r['answer']['claims'])
def setting(r):return f"{r['policy']}:{r['cap']}:t{r['threshold']}"
def summarize(rows):
 positive=[r for r in rows if r['question']['answerable']]
 negative=[r for r in rows if not r['question']['answerable']]
 sums={k:sum(r['metrics'][k] for r in rows) for k in ['input_tokens','output_tokens','claims_count','supported_claims','available_count','target_count','used_when_available_count','cited_target_count','model_verified_supported','correct_negative_abstention']}
 def ratio(a,b):return a/b if b else None
 sums['model_verified_supported']=sum(r['metrics']['model_verified_supported'] for r in positive)
 return dict(posthoc_screened_quality=ratio(sum(r['metrics']['model_verified_supported'] and explicit_track_alignment(r) for r in positive),len(positive)),cases=len(rows),positive_cases=len(positive),supported_answers=sums['model_verified_supported'],quality=ratio(sums['model_verified_supported'],len(positive)),negative_abstention=ratio(sums['correct_negative_abstention'],len(negative)),mean_input_tokens=sums['input_tokens']/len(rows),mean_total_tokens=(sums['input_tokens']+sums['output_tokens'])/len(rows),tokens_per_supported_answer=ratio(sums['input_tokens']+sums['output_tokens'],sums['model_verified_supported']),availability=ratio(sums['available_count'],sums['target_count']),utilization=ratio(sums['used_when_available_count'],sums['available_count']),target_recall=ratio(sums['cited_target_count'],sums['target_count']),model_attribution=ratio(sums['supported_claims'],sums['claims_count']),mean_receipt_latency_s=sum(r['metrics']['receipt_latency_s'] for r in rows)/len(rows),mean_rounds=sum(r['metrics']['retrieval_rounds'] for r in rows)/len(rows))
def main():
 rows=read(OUT/'runs.json');assert len(rows)==504 and len({r['case_id'] for r in rows})==504
 write(OUT/'posthoc_track_alignment.json',[dict(case_id=r['case_id'],passes=explicit_track_alignment(r),note='Necessary ID alignment only; not semantic proof.') for r in rows])
 groups=defaultdict(list)
 for r in rows:groups[setting(r)].append(r)
 policy=[];albums=[];tasks=[]
 for key,group in groups.items():
  r=group[0];meta=dict(setting=key,policy=r['policy'],cap=r['cap'],threshold=r['threshold'])
  policy.append(dict(**meta,**summarize(group)))
  for album in ALBUMS:albums.append(dict(**meta,album=album['id'],**summarize([r for r in group if r['album']==album['id']])))
  for task in ['local','distant','transformation','contradiction','multi_hop','abstention']:tasks.append(dict(**meta,task=task,**summarize([r for r in group if r['task']==task])))
 # Frontier for primary settings only; threshold variants are validation data.
 primary=[r for r in policy if r['policy']!='adaptive' or r['threshold']==3]
 for r in policy:
  r['pareto_primary']=r in primary and not any(x['mean_total_tokens']<=r['mean_total_tokens'] and x['quality']>=r['quality'] and (x['mean_total_tokens']<r['mean_total_tokens'] or x['quality']>r['quality']) for x in primary)
 for name,data in [('policy_means',policy),('album_means',albums),('task_means',tasks)]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
 folds=[];heldout=[]
 for album in ALBUMS:
  choices=[]
  for threshold in [2,3,4]:
   train=[r for r in rows if r['policy']=='adaptive' and r['threshold']==threshold and r['album']!=album['id']]
   score=summarize(train);feasible=score['quality']>=.8 and (score['availability'] or 0)>=.5 and (score['model_attribution'] or 0)>=.8
   choices.append(dict(threshold=threshold,feasible=feasible,**score))
  feasible=[s for s in choices if s['feasible']]
  chosen=min(feasible,key=lambda x:x['mean_total_tokens']) if feasible else min(choices,key=lambda x:(-x['quality'],x['mean_total_tokens']))
  test=[r for r in rows if r['policy']=='adaptive' and r['threshold']==chosen['threshold'] and r['album']==album['id']];heldout+=test
  folds.append(dict(heldout_album=album['id'],chosen_threshold=chosen['threshold'],training_choices=choices,test=summarize(test)))
 write(OUT/'leave_one_album_out.json',dict(folds=folds,aggregate=summarize(heldout),note='Threshold selection only; relation construction and Part I corpus inspection precede this study. Four albums do not establish artist-level generalization.'))
 usage=defaultdict(lambda:dict(calls=0,input_tokens=0,output_tokens=0))
 for p in PRIVATE.glob('call-*.json'):
  c=read(p);u=usage[c['kind']];u['calls']+=1;u['input_tokens']+=c['usage']['prompt_tokens'];u['output_tokens']+=c['usage']['completion_tokens']
 if usage:write(OUT/'unique_provider_usage.json',dict(usage))
 primary_ids={c['request_sha256'] for r in rows for c in r['receipts']}
 session_path=OUT/'sessions.json'
 session_ids={c['request_sha256'] for r in read(session_path) for c in r['receipts']} if session_path.exists() else set()
 scopes=defaultdict(lambda:dict(calls=0,input_tokens=0,output_tokens=0))
 for path in PRIVATE.glob('call-*.json'):
  c=read(path);scope='primary_and_threshold' if c['request_sha256'] in primary_ids else 'sessions_only' if c['request_sha256'] in session_ids else 'preparation' if c['kind'] in ['relations','questions','negative_audit'] else 'smoke_or_failed_attempts'
  v=scopes[scope];v['calls']+=1;v['input_tokens']+=c['usage']['prompt_tokens'];v['output_tokens']+=c['usage']['completion_tokens']
 if scopes:write(OUT/'unique_usage_by_scope.json',dict(scopes))
 emb=PRIVATE/'embedding-usage.json'
 if emb.exists():write(OUT/'embedding_usage.json',read(emb))
 print(pd.DataFrame(primary)[['policy','cap','quality','mean_total_tokens','availability','utilization','model_attribution','tokens_per_supported_answer','pareto_primary']].to_string(index=False))
if __name__=='__main__':main()
