"""Focused offline invariants for the review-stage comparisons."""
import csv
import json
import unittest
from pathlib import Path
from review_supplements import f1

R=Path(__file__).resolve().parents[1]/'review_2026-10-22'
def rows(name):
    with (R/name).open() as f: return list(csv.DictReader(f))

class ReviewChecks(unittest.TestCase):
    def test_binary_f1(self):
        self.assertEqual(f1('0011','0011'),1)
        self.assertEqual(f1('0011','1100'),0)
        self.assertAlmostEqual(f1('0011','0000'),1/3)

    def test_policies_and_work(self):
        d=json.loads((R/'review_analysis_checks.json').read_text())
        self.assertEqual(d['new_model_calls'],0)
        self.assertEqual(d['unique_cell_candidates'],380)
        w=d['source_test_workload']
        self.assertEqual(w['gross_cell_items'],380*204)
        self.assertEqual(w['gross_cell_items']-w['existing_endpoint_cell_items'],w['remaining_cell_items'])
        self.assertEqual(w['remaining_cell_votes'],3*w['remaining_cell_items'])
        self.assertEqual(w['prediction_conflicts'],0)
        self.assertTrue(all(d['tests'].values()))
        rs=rows('selection_policy_cells.csv')
        self.assertEqual(len(rs),60)
        for r in rs:
            self.assertEqual(int(r['last_proposal_iteration']),20)
            self.assertEqual(int(r['n_generated_unique']),int(r['n_unique'])-1)
            self.assertGreaterEqual(float(r['pool_gap_pp']),-1e-12)
            if r['optimizer']=='pe2':
                self.assertEqual(r['p_source_hash'],r['final_current_hash'])

    def test_case_accounting(self):
        changed=rows('case_all_changed_items.csv')
        summary=rows('case_summary.csv')
        for s in summary:
            rs=[r for r in changed if (r['cell_id'],r['contrast'])==(s['cell_id'],s['contrast'])]
            self.assertEqual(len(rs),int(s['fixed'])+int(s['broken']))
            self.assertEqual(int(s['net']),int(s['fixed'])-int(s['broken']))
            self.assertEqual(sum(r['transition']=='Fixed' for r in rs),int(s['fixed']))
        examples=rows('case_display_examples.csv')
        self.assertEqual(len(examples),8)
        self.assertEqual(sum(r['transition']=='Fixed' for r in examples),4)

if __name__=='__main__': unittest.main()
