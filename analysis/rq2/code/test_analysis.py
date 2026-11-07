import unittest
import numpy as np
import pandas as pd
from analyze import corr,ranks,order,scores,rank_boot

class TestResearchLogic(unittest.TestCase):
    def test_constant_and_small_pool(self):
        self.assertTrue(np.isnan(corr([1,1],[2,3])[0]))
        self.assertAlmostEqual(corr([1,2],[2,1])[0],-1)
    def test_ties(self):
        np.testing.assert_equal(ranks([.8,.8,.5]),[1.5,1.5,3])
        np.testing.assert_equal(ranks([.8,.8+1e-13,.5]),[1.5,1.5,3])
    def test_newer_iteration_tie(self):
        d=pd.DataFrame({'last_iteration':[2,9,8],'prompt_sha256':['a','b','c']})
        self.assertEqual(order(d,np.array([.8,.8,.5])),[1,0,2])
    def test_metrics(self):
        s=scores([0,0,0,1,1],[0,0,1,1,0])
        self.assertAlmostEqual(s['macro_f1'][0],(2/3+.5)/2)
        self.assertAlmostEqual(s['weighted_f1'][0],.6)
    def test_multiplicity_not_deduplicated(self):
        s=scores([0,1],[0,0],weights=[1,3])
        self.assertAlmostEqual(s['accuracy'][0],.25)
    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):scores([0,1],[2,0])
    def test_vectorized_rank_boot(self):
        x=np.array([1.,1.,3.,4.]);ys=np.array([[1.,2.,3.,3.],[4.,3.,1.,1.],[2.,2.,2.,2.]])
        a,b,_=rank_boot(x,ys)
        for i,y in enumerate(ys):np.testing.assert_allclose([a[i],b[i]],corr(x,y),atol=1e-12,equal_nan=True)

if __name__=='__main__':unittest.main()
