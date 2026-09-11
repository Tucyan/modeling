import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from q2_incremental import select_month, calibration_metrics
import pandas as pd
import numpy as np

class IncrementalTests(unittest.TestCase):
    def test_future_cost_does_not_change_selection(self):
        frames = {m: pd.DataFrame({'day':range(14,80), 'cost':v}) for m,v in [('M1',2.),('M3',1.),('M4',3.)]}
        first = select_month(frames,59)
        frames['M1'].loc[frames['M1'].day>=59,'cost']=-1e12
        self.assertEqual(first,select_month(frames,59))
        self.assertEqual(first['winner'],'M3')
        self.assertEqual(first['end_exclusive'],59)

    def test_missing_history_rejected(self):
        frames={'M1':pd.DataFrame({'day':[14,16],'cost':[1,2]})}
        with self.assertRaises(ValueError): select_month(frames,17)

    def test_quantile_score_units_and_ties(self):
        coverage,loss=calibration_metrics(np.array([0.,2.]),np.array([1.,1.]),.8)
        self.assertEqual(coverage,.5)
        self.assertAlmostEqual(loss,.5)

if __name__=='__main__': unittest.main()
