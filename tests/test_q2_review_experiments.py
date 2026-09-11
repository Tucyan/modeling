import sys, unittest, importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from q2_solver import np, execute

class ReviewTests(unittest.TestCase):
    def engine(self):
        self.assertIsNotNone(importlib.util.find_spec('q2_review_experiments'), 'new experiment engine missing')
        import q2_review_experiments as m
        return m
    def test_terminal_rules(self):
        m=self.engine()
        self.assertEqual(m.terminal_target('T1',1500),1500)
        self.assertEqual(m.terminal_target('T0',1500),6000)
        self.assertIsNone(m.terminal_target('T4',1500))
    def test_causal_day_input(self):
        m=self.engine(); a=np.arange(30*4,dtype=float).reshape(30,4); p=a*.8
        x=m.planning_input(a,p,20,'M1',.8,14)
        a[20:]=9999
        np.testing.assert_array_equal(x,m.planning_input(a,p,20,'M1',.8,14))
    def test_execution_prefix(self):
        m=self.engine(); g=[0.,0.,0.]; a=execute(g,[1.,2.,3.],[0.,0.,0.],6000.)
        b=execute(g,[1.,2.,9999.],[0.,0.,0.],6000.)
        np.testing.assert_array_equal(a['E'][:3],b['E'][:3])
    def test_date_window(self):
        m=self.engine(); self.assertEqual(m.history_window(59),(14,59))
        self.assertEqual(m.history_window(200),(144,200))
    def test_terminal_cannot_dump_battery(self):
        m=self.engine()
        self.assertTrue(hasattr(m,'execute_closed'),'physical terminal controller missing')
        r=m.execute_closed([1000.,1000.],[400.,400.],[0.,0.],6500.,6000.)
        self.assertLessEqual(float(r['spill'].max()),1e-7)
        self.assertAlmostEqual(r['E'][-1],6000.)
    def test_unreachable_terminal_rejected(self):
        m=self.engine()
        self.assertTrue(hasattr(m,'execute_closed'),'physical terminal controller missing')
        with self.assertRaises(ValueError): m.execute_closed([0.],[0.],[0.],6500.,6000.)

if __name__=='__main__': unittest.main()
