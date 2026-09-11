"""Four genuinely linked turns per album/policy with one persistent graph thread."""
from run import setup
from engine import *
import mlflow

def main():
 setup();rows=[]
 for album in ALBUMS:
  aid=album['id'];memory=Memory(aid)
  questions={q['kind']:q for q in read(OUT/f'questions-{aid}.json')}
  for policy in ['full','adaptive']:
   graph=memory.graph();session=f'part2-session-{aid}-{policy}';history=[]
   for turn,kind in enumerate(['local','transformation','distant','contradiction'],1):
    q=public_question(questions[kind],memory.n)
    exposed=[]
    for item in reversed(history):
     candidate=[item]+exposed
     if tokens(candidate)<=250:exposed=candidate
    if exposed:
     q['previous_verified_or_failed_claims']=exposed
     q['conversation_instruction']='Relate the current interpretation to earlier claims where evidence permits; correct earlier claims when sources conflict. Prior answers are fallible, not independent sources.'
    with mlflow.start_run(run_name=f'{session}-turn-{turn}') as run:
     mlflow.log_params(dict(session_id=session,turn=turn,album=aid,policy=policy))
     with mlflow.start_span(name='album_conversation_turn'):
      mlflow.update_current_trace(session_id=session)
      result=graph.invoke(dict(question=q,policy=policy,cap=100000,level=0,threshold=3,rounds=[],receipts=[]),dict(configurable=dict(thread_id=session),recursion_limit=60))
     metrics=dict(input_tokens=sum(r['usage']['prompt_tokens'] for r in result['receipts']),output_tokens=sum(r['usage']['completion_tokens'] for r in result['receipts']),history_tokens=tokens(exposed),history_items=len(exposed),rounds=len(result['rounds']),supported=int(result['supported']))
     mlflow.log_metrics(metrics)
     row=dict(session_id=session,turn=turn,album=aid,policy=policy,question=q,exposed_history=exposed,answer=result['answer'],verification=result['verification'],metrics=metrics,rounds=result['rounds'],receipts=result['receipts'],mlflow_run_id=run.info.run_id)
     mlflow.log_dict(row,'turn.json');rows.append(row)
    history.append(dict(task=kind,claims=result['answer']['claims'],supported=result['supported']))
    # The same checkpointer must retain this turn for the next invocation.
    assert graph.get_state(dict(configurable=dict(thread_id=session))).values['question']==q
    write(PRIVATE/'sessions.json',rows)
    print('SESSION',aid,policy,turn,'history',len(exposed),'supported',result['supported'],flush=True)
 write(OUT/'sessions.json',rows);mlflow.flush_trace_async_logging()
if __name__=='__main__':main()
