import sys,unittest,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
class RollingTests(unittest.TestCase):
    def test_selection_strictly_past(self):
        self.assertIsNotNone(importlib.util.find_spec('q2_rolling_validation'),'rolling selector missing')
        from q2_rolling_validation import choose_from_scores
        self.assertEqual(choose_from_scores({'M1':2.,'M3':1.,'M4':3.},58,59),'M3')
        with self.assertRaises(ValueError): choose_from_scores({'M1':2.},59,59)
    def test_deterministic_tie(self):
        self.assertIsNotNone(importlib.util.find_spec('q2_rolling_validation'),'rolling selector missing')
        from q2_rolling_validation import choose_from_scores
        self.assertEqual(choose_from_scores({'M1':1.,'M3':1.},58,59),'M1')
if __name__=='__main__':unittest.main()
