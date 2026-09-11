import unittest,sys,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
class VerifyTests(unittest.TestCase):
    def test_wrong_date_value_is_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec('q2_review_verify'),'verifier missing')
        from q2_review_verify import check_paper_dates
        with self.assertRaises(AssertionError):
            check_paper_dates('### 2025-09-23\n0.00\n### 2025-12-21\n6.38',{'2025-09-23':{'emergency_kwh':6.375}})
if __name__=='__main__':unittest.main()
