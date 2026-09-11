import unittest
from unittest.mock import patch
from engine import *
class EngineTests(unittest.TestCase):
 def test_budget_and_provenance(self):
  m=Memory('rhcp',with_embeddings=False)
  q=public_question(read(OUT/'questions-rhcp.json')[0],m.n)
  for b in BUDGETS[:-1]:
   text,ids,addresses=m.select('tfidf',q,b)
   self.assertLessEqual(tokens(text),b)
   self.assertTrue(set(addresses)<=set(m.claims))
   self.assertEqual(len(ids),len(set(ids)))
 def test_structural_constraints(self):
  a={'claims':[{'sources':['T01:C1','T15:C1']}]}
  self.assertTrue(structural(a,'distant',15,['T01:C1','T15:C1']))
  self.assertFalse(structural(a,'local',15,['T01:C1','T15:C1']))
  self.assertFalse(structural(a,'distant',15,['T15:C1']))
  self.assertFalse(structural(a,'multi_hop',15,['T01:C1','T15:C1']))
 def test_adaptive_termination_and_hidden_labels(self):
  m=Memory('rhcp',with_embeddings=False)
  q=public_question(read(OUT/'questions-rhcp.json')[1],m.n)
  def fake(kind,key,system,user,**kw):
   self.assertNotIn('target_ids',user);self.assertNotIn('expected',user)
   out={'score':0,'reason':'insufficient'} if kind=='sufficiency' else {'answer':'Insufficient evidence','claims':[],'abstained':True,'limitation':'test'}
   return dict(kind=kind,output=out,usage={'prompt_tokens':1,'completion_tokens':1},latency_s=0)
  with patch.object(m,'select',return_value=('{}',[],[])),patch('engine.call',fake):
   g=m.graph();s=g.invoke(dict(question=q,policy='adaptive',cap=100000,level=0,threshold=3,rounds=[],receipts=[]),dict(configurable=dict(thread_id='a'),recursion_limit=60))
   self.assertEqual(len(s['rounds']),6);self.assertFalse(s['supported'])
   self.assertFalse(g.get_state(dict(configurable=dict(thread_id='b'))).values)
if __name__=='__main__':unittest.main()
