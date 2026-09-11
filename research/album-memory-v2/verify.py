"""Structural/provenance audit. Does not certify musical correctness."""
import sqlite3,itertools
from run import settings
from prepare import *
def main():
 rows=read(OUT/'runs.json');sessions=read(OUT/'sessions.json')
 expected={(a['id'],q['kind'],p,b,t) for a in ALBUMS for q in read(OUT/f"questions-{a['id']}.json") for p,b,t in settings()}
 assert len(rows)==504 and {(r['album'],r['task'],r['policy'],r['cap'],r['threshold']) for r in rows}==expected
 assert len(sessions)==32 and len({r['session_id'] for r in sessions})==8
 for group in {r['session_id'] for r in sessions}:
  assert {r['turn'] for r in sessions if r['session_id']==group}=={1,2,3,4}
  assert all(r['metrics']['history_tokens']<=250 for r in sessions if r['session_id']==group)
 budget_checks=0;receipt_checks=0
 with sqlite3.connect(PRIVATE/'mlflow.db') as db:
  for r in rows+sessions:
   assert db.execute('select status from runs where run_uuid=?',(r['mlflow_run_id'],)).fetchone()==('FINISHED',)
   for rec in r['receipts']:
    original=read(PRIVATE/f"call-{rec['request_sha256']}.json")
    assert original['usage']==rec['usage'] and original['request_id']==rec['request_id'];receipt_checks+=1
   assert sum(c['usage']['prompt_tokens'] for c in r['receipts'])==r['metrics']['input_tokens']
   assert sum(c['usage']['completion_tokens'] for c in r['receipts'])==r['metrics']['output_tokens']
   if r['policy'] not in ['full','recent','summary']:
    for rnd in r['rounds']:assert rnd['context_tokens']<=rnd['cap'];budget_checks+=1
  meta=db.execute("select value from trace_request_metadata where key='mlflow.trace.session'").fetchall()
  for ident in {r['session_id'] for r in sessions}:assert any(ident in v[0] for v in meta),ident
 for album in ALBUMS:
  aid=album['id'];addresses={f"{c['id']}:{v['claim_id']}" for c in cards_for(aid) for v in c['claims']}
  for edge in read(OUT/f'relations-{aid}.json'):assert len(edge['evidence'])==2 and set(edge['evidence'])<=addresses
 assert len(read(Path(__file__).parent/'human-review/items.json'))==72
 report=dict(primary_and_threshold_cases=504,conversation_turns=32,sessions=8,relations=48,blind_review_items=72,receipt_references_checked=receipt_checks,budget_checks=budget_checks,human_ratings_collected=0,scope='Structural, receipt, budget and session audit only; online-verifier errors are documented in the article.')
 write(OUT/'verification.json',report);print(dumps(report))
def cards_for(aid):return read(OLD/f'cards-{aid}.json')
if __name__=='__main__':main()
