import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from q2_solver import optimize, execute, features
import numpy as np

class Q2Tests(unittest.TestCase):
    def test_arbitrage(self):
        r=optimize(np.array([0.,810.]),np.array([1.,3.]),1200.,target=1200.,cap=1000.)
        self.assertAlmostEqual(r['g'][0],1000.,places=5)
        self.assertAlmostEqual(r['g'][1],0.,places=5)
    def test_newsvendor(self):
        # Five equal-probability demands: any point in [40,50] minimizes cost.
        r=optimize(np.array([[10.],[20.],[30.],[40.],[50.]]),np.array([1.]),1200.,target=1200.,cap=0)
        self.assertTrue(39.9999 <= r['g'][0] <= 50.0001)
    def test_causal_and_paid_excess(self):
        r=execute([100.,0.],[0.,200.],[0.,0.],1200.)
        self.assertAlmostEqual(r['c'][0],100.)
        self.assertAlmostEqual(r['d'][1],81.)
        self.assertAlmostEqual(r['e'][1],119.)
        a=execute([100.],[0.],[0.],10800.)
        self.assertAlmostEqual(a['unused'][0],100.)
    def test_terminal_reachable(self):
        r=execute(np.zeros(12),np.zeros(12),np.zeros(12),1200.,terminal=6000.)
        self.assertAlmostEqual(r['E'][-1],6000.)
        self.assertLessEqual(max(r['c']),5000/6+1e-7)
    def test_features_no_future(self):
        a=np.arange(30*144,dtype=float).reshape(30,144)
        x=features(a,20,'load')
        a[20:]=99999
        np.testing.assert_array_equal(x,features(a,20,'load'))

if __name__=='__main__': unittest.main()
