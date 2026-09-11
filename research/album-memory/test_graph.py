"""Offline invariants: context budgets, real graph execution, isolated checkpoints."""
import json
import unittest
from unittest.mock import patch
import study


class MemoryGraphTests(unittest.TestCase):
    def test_policies_and_thread_isolation(self):
        cards = [dict(id=f'T{i:02}', position=i,
                      stance='early rupture' if i == 2 else 'renewal')
                 for i in range(1, 13)]
        probe = dict(kind='change', question='How does early rupture change?',
                     target_ids=['T02', 'T11'])

        def fake_call(*args, **kwargs):
            return dict(output=dict(answer='', evidence_ids=[], abstained=True,
                                    uncertainty='test'), usage=dict(prompt_tokens=1,
                                    completion_tokens=1))

        with patch.object(study, 'call', fake_call):
            graph = study.build_graph(cards, 'T02 rupture; T11 renewal.')
            for policy, count in [('full', 12), ('recent3', 3),
                                  ('summary', 2), ('selective6', 6)]:
                config = {'configurable': {'thread_id': policy}, 'recursion_limit': 100}
                result = graph.invoke(dict(album='test', policy=policy, probe=probe,
                                           repetition=0, archive=[], cursor=0), config)
                self.assertEqual(len(result['archive']), 12)
                self.assertEqual(result['validation']['selected_tracks'], count)
                self.assertEqual(result['validation']['evidence_recall'], 0)
                if policy == 'recent3':
                    self.assertEqual(result['validation']['target_availability'], .5)
                if policy == 'selective6':
                    self.assertTrue({'T01', 'T02', 'T11', 'T12'} <= set(result['selected_ids']))
            # A single compiled graph must retain separate completed states.
            for policy in ['full', 'recent3', 'summary', 'selective6']:
                saved = graph.get_state({'configurable': {'thread_id': policy}})
                self.assertEqual(saved.values['policy'], policy)
            empty = graph.get_state({'configurable': {'thread_id': 'unseen'}})
            self.assertFalse(empty.values)


if __name__ == '__main__':
    unittest.main()
