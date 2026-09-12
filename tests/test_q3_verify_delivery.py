"""Independent Q3 delivery-verifier regression tests."""
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p)
                for p in ('src',)]
import pandas as pd
import numpy as np

from q3_verify_delivery import _assert_small, verify_dispatch


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

    def test_verifier_rejects_nonfinite_ledger_values(self):
        with self.assertRaises(AssertionError):
            _assert_small({'nan_residual': np.nan})
        for column, value in (('g', np.nan), ('e', np.inf), ('price', -np.inf)):
            corrupted = self.dispatch.copy()
            corrupted.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(AssertionError):
                verify_dispatch(self.daily, corrupted, self.versions, 'N0')

        bad_daily = self.daily.copy()
        bad_daily.loc[0, 'cost'] = np.nan
        with self.assertRaises(AssertionError):
            verify_dispatch(bad_daily, self.dispatch, self.versions, 'N0')
        bad_versions = self.versions.copy()
        bad_versions.loc[0, 'q'] = np.inf
        with self.assertRaises(AssertionError):
            verify_dispatch(self.daily, self.dispatch, bad_versions, 'N0')
        unknown_version = self.versions.copy()
        unknown_version.loc[0, 'date'] = '2026-01-01'
        with self.assertRaises(AssertionError):
            verify_dispatch(self.daily, self.dispatch, unknown_version, 'N0')
        unsupported_issue = self.versions.copy()
        unsupported_issue.loc[0, 'issue_hour'] = 5
        with self.assertRaises(AssertionError):
            verify_dispatch(self.daily, self.dispatch, unsupported_issue, 'N0')

    def test_verifier_rejects_missing_or_unordered_dates_and_slots(self):
        with self.assertRaises(AssertionError):
            verify_dispatch(self.daily.iloc[:-1], self.dispatch.iloc[:-144],
                            self.versions, 'N0')
        unordered = self.daily.copy()
        unordered.iloc[[0, 1]] = unordered.iloc[[1, 0]].to_numpy()
        with self.assertRaises(AssertionError):
            verify_dispatch(unordered, self.dispatch, self.versions, 'N0')
        missing_slot = self.dispatch.drop(index=0)
        with self.assertRaises(AssertionError):
            verify_dispatch(self.daily, missing_slot, self.versions, 'N0')

    def test_verifier_accepts_independently_selected_n1_ledger(self):
        folder = ROOT / 'outputs/q3_unified_v1/N1'
        if not (folder / 'dispatch_10min.csv').exists():
            self.skipTest('unified N1 output has not run')
        checks = verify_dispatch(
            pd.read_csv(folder / 'daily_summary.csv'),
            pd.read_csv(folder / 'dispatch_10min.csv'),
            pd.read_csv(folder / 'contract_versions.csv'), 'N1')
        self.assertLess(checks['cost_error_yuan'], 1e-6)


if __name__ == '__main__':
    unittest.main()
