"""Targeted statistical checks; run with the same environment as analyze.py."""
import unittest
import numpy as np
from analyze import metric, conf, assert_votes

class StatisticalChecks(unittest.TestCase):
    def test_known_confusion(self):
        x=metric([80,10,5,5])
        self.assertAlmostEqual(x['macro_f1'],(.4+160/175)/2)
        self.assertAlmostEqual(x['weighted_f1'],.9*160/175+.1*.4)
        self.assertAlmostEqual(x['security_recall'],.5)
        self.assertAlmostEqual(x['security_precision'],1/3)
    def test_zero_division_explicit(self):
        x=metric([6,0,4,0])
        self.assertEqual(x['security_precision'],0)
        self.assertEqual(x['security_f1'],0)
        self.assertTrue(np.isfinite(list(x.values())).all())
    def test_repository_multiplicity_is_not_deduplicated(self):
        # Repository A is selected twice and B once.
        a=conf([0,1],[0,0]); b=conf([0,1,1],[1,1,1])
        expected=conf([0,1,0,1,0,1,1],[0,0,0,0,1,1,1])
        np.testing.assert_array_equal(2*a+b,expected)
        self.assertNotEqual(metric(a+b)['macro_f1'],metric(expected)['macro_f1'])
    def test_majority_and_illegal_votes(self):
        assert_votes({'vote_labels':['Security','Non-security','Security'],'prediction':'Security','gold':'Non-security'})
        with self.assertRaises(AssertionError):
            assert_votes({'vote_labels':['INVALID']*3,'prediction':'Non-security','gold':'Security'})
    def test_paired_zero_effect(self):
        x=metric(np.array([[80,10,5,5],[160,20,10,10]]))['macro_f1']
        np.testing.assert_array_equal(x-x,[0,0])

if __name__=='__main__':unittest.main()
