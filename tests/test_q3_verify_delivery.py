"""Independent Q3 delivery-verifier regression tests."""
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p)
                for p in ('src',)]
import pandas as pd

from q3_verify_delivery import verify_dispatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/q3_final'


class Q3VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT / 'dispatch_10min.csv').exists():
            raise unittest.SkipTest('formal Q3 output has not run')
        cls.daily = pd.read_csv(OUT / 'daily_summary.csv')
        cls.dispatch = pd.read_csv(OUT / 'dispatch_10min.csv')
        cls.versions = pd.read_csv(OUT / 'contract_versions.csv')

    def test_verifier_rejects_changed_final_contract(self):
        """A changed final contract must break the bus/cost or version reconstruction."""
        corrupted = self.dispatch.copy()
        corrupted.loc[0, 'q'] += 1
        with self.assertRaises(AssertionError):
            verify_dispatch(self.daily, corrupted, self.versions, 'N0')

    def test_verifier_accepts_unchanged_formal_ledgers(self):
        """The independent equations accept the untouched canonical CSV ledgers."""
        checks = verify_dispatch(self.daily, self.dispatch, self.versions, 'N0')
        self.assertLess(checks['max_residuals_kwh']['bus'], 1e-6)
        self.assertLess(checks['cost_error_yuan'], 1e-6)


if __name__ == '__main__':
    unittest.main()
