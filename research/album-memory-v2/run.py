"""Run static budget curves and bounded adaptive trajectories, resumably."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import random,threading
from engine import *
import mlflow
LOCK=threading.Lock()
def setup():
 mlflow.set_tracking_uri('sqlite:///'+str(PRIVATE/'mlflow.db'))
 mlflow.set_registry_uri(mlflow.get_tracking_uri())
 name='album-memory-v2-context-allocation'
 if mlflow.get_experiment_by_name(name) is None:mlflow.create_experiment(name,artifact_location=(PRIVATE/'artifacts').as_uri())
 mlflow.set_experiment(name);mlflow.langchain.autolog();mlflow.openai.autolog()
 p=OUT/'prompt_manifest.json'
 fingerprint=hashlib.sha256(dumps([ANSWER,SUFFICIENCY,VERIFIER,RELATIONS,QUESTIONS]).encode()).hexdigest()
 fp=OUT/'prompt_fingerprint.json'
 if p.exists() and fp.exists() and read(fp)==fingerprint:return read(p)
 versions={}
 for name,template in [('answer',ANSWER),('sufficiency',SUFFICIENCY),('verifier',VERIFIER),('relations',RELATIONS),('questions',QUESTIONS)]:
  r=mlflow.genai.register_prompt(name=f'album-v2-{name}',template=template,commit_message='Part II frozen protocol')
  versions[name]=f'prompts:/{r.name}/{r.version}'
 write(p,versions);write(fp,fingerprint);return versions

def settings():
 return [('full',100000,3),('recent',100000,3),('summary',300,3)]+[(p,b,3) for p in ['tfidf','embedding','relational'] for b in BUDGETS[:-1]]+[('adaptive',100000,t) for t in [2,3,4]]

def execute(memory,q,policy,cap,threshold,prompts):
 aid=memory.aid;case=f'{aid}-{q["kind"]}-{policy}-{cap}-t{threshold}'
 path=PRIVATE/f'case-{case}.json'
 fingerprint=hashlib.sha256(dumps([ANSWER,SUFFICIENCY,VERIFIER]).encode()).hexdigest()
 if path.exists() and read(path).get('prompt_fingerprint')==fingerprint:return read(path)
 question=public_question(q,memory.n);graph=memory.graph();start=time.monotonic()
 with mlflow.start_run(run_name=case) as run:
  mlflow.log_params(dict(album=aid,task=q['kind'],policy=policy,cap=cap,threshold=threshold,generation_model=GEN,verifier_model=VERIFY))
  for uri in prompts.values():mlflow.genai.load_prompt(uri)
  with mlflow.start_span(name='context_allocation'):
   mlflow.update_current_trace(session_id=case)
   state=graph.invoke(dict(question=question,policy=policy,cap=cap,threshold=threshold,level=0,rounds=[],receipts=[]),dict(configurable=dict(thread_id=case),recursion_limit=60))
  receipts=state['receipts'];by_stage={}
  for r in receipts:
   v=by_stage.setdefault(r['kind'],dict(input_tokens=0,output_tokens=0,latency_s=0,calls=0))
   v['input_tokens']+=r['usage']['prompt_tokens'];v['output_tokens']+=r['usage']['completion_tokens'];v['latency_s']+=r['latency_s'];v['calls']+=1
  targets=set(q['target_ids']);available=targets&set(state['selected_ids']);cited={s.split(':')[0] for c in state['answer']['claims'] for s in c['sources']};used=available&cited
  # Summary cannot claim source exposure from an ID mentioned in prose.
  metrics=dict(format_error=int(state['answer'].get('format_error',False)),input_tokens=sum(v['input_tokens'] for v in by_stage.values()),output_tokens=sum(v['output_tokens'] for v in by_stage.values()),final_context_tokens=tokens(state['memory']),retrieval_rounds=len(state['rounds']),target_count=len(targets),available_count=len(available),used_when_available_count=len(used),cited_target_count=len(targets&cited),model_verified_supported=int(state['supported']),abstained=int(state['answer']['abstained']),correct_negative_abstention=int(not q['answerable'] and state['answer']['abstained']),claims_count=len(state['answer']['claims']),supported_claims=sum(c['supported'] for c in state['verification']['claims']),latency_s=time.monotonic()-start,receipt_latency_s=sum(v['latency_s'] for v in by_stage.values()))
  mlflow.log_metrics(metrics)
  result=dict(prompt_fingerprint=fingerprint,prompt_uris=prompts,case_id=case,album=aid,task=q['kind'],policy=policy,cap=cap,threshold=threshold,question=q,answer=state['answer'],verification=state['verification'],selected_ids=state['selected_ids'],selected_addresses=state['selected_addresses'],rounds=state['rounds'],metrics=metrics,stages=by_stage,receipts=receipts,mlflow_run_id=run.info.run_id)
  mlflow.log_dict(result,'result.json');write(path,result)
 return result

def main(smoke=False):
 prompts=setup();memories={a['id']:Memory(a['id']) for a in ALBUMS}
 jobs=[(memories[a['id']],q,p,b,t,prompts) for a in ALBUMS for q in read(OUT/f"questions-{a['id']}.json") for p,b,t in settings()]
 # Threshold phases run separately to avoid duplicate concurrent requests for
 # shared deterministic adaptive prefixes. Cached prefix receipts represent
 # trajectory resource usage, not a claim of duplicate provider billing.
 rows=[]
 for phase in [3,2,4]:
  selected=[j for j in jobs if j[4]==phase];random.Random(52).shuffle(selected)
  if smoke:selected=[next(j for j in jobs if j[0].aid=='rhcp' and j[1]['kind']=='local' and j[2]=='full'),next(j for j in jobs if j[0].aid=='rhcp' and j[1]['kind']=='distant' and j[2]=='adaptive' and j[4]==3)] if phase==3 else []
  with ThreadPoolExecutor(max_workers=4) as pool:
   futures=[pool.submit(execute,*job) for job in selected]
   for f in as_completed(futures):
    try:row=f.result()
    except Exception as error:
     print('CASE FAILED',type(error).__name__,str(error)[:120],flush=True)
     continue
    rows.append(row)
    with LOCK:print('CASE',len(rows),'/',2 if smoke else len(jobs),row['case_id'],'supported',row['metrics']['model_verified_supported'],'tokens',row['metrics']['input_tokens'],flush=True)
 if not smoke:
  assert len(rows)==len(jobs),'Some cases failed; rerun to resume completed cases from cache'
  write(OUT/'runs.json',rows)
 mlflow.flush_trace_async_logging();print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':
 import sys
 main('--smoke' in sys.argv)
