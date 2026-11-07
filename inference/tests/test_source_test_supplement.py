import json
import tempfile
import unittest
from pathlib import Path

from source_test_supplement import (append_vote, check_vote, correlations,
    load_votes, metrics, read_journal, split_ids, tied_ranks, validate_rows)


class SourceTestSupplementTests(unittest.TestCase):
    def test_export_split_aliases_preserve_order_and_reject_conflicts(self):
        old={'pool_ids':[2,1],'val_ids':[3],'test_ids':[0]}
        export={'pool':[2,1],'val':[3],'test':[0]}
        self.assertEqual(split_ids(export),old)
        self.assertEqual(split_ids(old),old)
        with self.assertRaises(RuntimeError):split_ids({**old,'pool':[1,2]})

    def fixture(self):
        r={'route_fingerprint':'model-A-full-contract','test_ids':[7]}
        c={'item_id':7,'text_sha256':'text','rendered_request_sha256':'render','input_tokens':12}
        v={**c,'route_fingerprint':r['route_fingerprint'],'vote_index':1,
           'raw':'Security','label':'Security','finish_reason':'stop'}
        return r,c,v

    def test_resume_keeps_real_individual_votes(self):
        r,c,v=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'votes.jsonl';append_vote(p,v)
            loaded=load_votes(p,r,{7:c})
            self.assertEqual(set(loaded),{(7,1)})
            self.assertNotIn((7,2),loaded)
            append_vote(p,{**v,'vote_index':2})
            self.assertEqual(len(load_votes(p,r,{7:c})),2)

    def test_duplicate_votes_rejected(self):
        r,c,v=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'votes.jsonl';append_vote(p,v);append_vote(p,v)
            with self.assertRaises(RuntimeError):load_votes(p,r,{7:c})

    def test_torn_tail_preserved_without_losing_valid_vote(self):
        r,c,v=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'votes.jsonl';append_vote(p,v)
            with p.open('ab') as f:f.write(b'{"torn":')
            self.assertEqual(read_journal(p),[v])
            self.assertEqual(next(Path(d).glob('*.torn-*')).read_bytes(),b'{"torn":')
            self.assertTrue(p.read_bytes().endswith(b'\n'))

    def test_complete_json_without_newline_retained(self):
        _,_,v=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'votes.jsonl';p.write_text(json.dumps(v))
            self.assertEqual(read_journal(p),[v])
            self.assertTrue(p.read_bytes().endswith(b'\n'))

    def test_wrong_model_input_prompt_contract_rejected(self):
        r,c,v=self.fixture()
        for delta in ({'route_fingerprint':'model-B'}, {'text_sha256':'other'},
                      {'rendered_request_sha256':'different-template'}, {'input_tokens':13}):
            with self.assertRaises(RuntimeError):check_vote({**v,**delta},r,c)

    def test_invalid_and_truncation_rejected(self):
        r,c,v=self.fixture()
        for delta in ({'label':'INVALID'}, {'raw':'not a label'}, {'finish_reason':'length'}):
            with self.assertRaises(RuntimeError):check_vote({**v,**delta},r,c)

    def test_ties_constant_and_small_pools(self):
        self.assertEqual(tied_ranks([1,1+5e-13,0]),[1.5,1.5,3])
        self.assertIsNone(correlations([1,1],[0,1])['rho'])
        self.assertEqual(correlations([1,0],[0,1])['tau_b'],-1)
        self.assertAlmostEqual(correlations([3,2,2,1],[3,2,2,1])['tau_b'],1)
        self.assertAlmostEqual(correlations([3,2,2,1],[3,2,2,1])['rho'],1)

    def test_metrics_known_matrix_and_invalid(self):
        rows=[{'prediction':p,'gold':g} for p,g in
              [('Security','Security'),('Security','Non-security'),('Non-security','Non-security')]]
        self.assertAlmostEqual(metrics(rows)['macro_f1'],2/3)
        with self.assertRaises(RuntimeError):metrics([{'prediction':'INVALID','gold':'Security'}])

    def test_missing_votes_and_wrong_majority_fail(self):
        from types import SimpleNamespace
        r,c,v=self.fixture();examples={7:SimpleNamespace(label='Security')}
        row={'item_id':7,'gold':'Security','prediction':'Security','votes':[v]}
        with self.assertRaises(RuntimeError):validate_rows([row],r,{7:c},examples)
        row['votes']=[{**v,'vote_index':i} for i in (1,2,3)];row['prediction']='Non-security'
        with self.assertRaises(RuntimeError):validate_rows([row],r,{7:c},examples)


if __name__=='__main__':unittest.main()
