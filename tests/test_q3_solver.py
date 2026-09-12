"""Analytical settlement, storage and information-boundary regression cases."""
import sys
import unittest
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p) for p in ('src',)]
import numpy as np
import q3_solver as q3


class Q3Tests(unittest.TestCase):
    def test_settlement_not_double_charged(self):
        self.assertAlmostEqual(q3.settle([100], [80], [1], 'N0').sum(), 110)
        self.assertAlmostEqual(q3.settle([100], [80], [1], 'N1').sum(), 90)
        self.assertAlmostEqual(q3.settle([100], [120], [1], 'N0').sum(), 130)
        self.assertAlmostEqual(q3.settle([100], [110], [1], 'N0').sum(), 115)

    def test_no_refund_cannot_profit_from_reducing_original(self):
        a = q3.optimize([0], [0], [1], 6000, original=[100], cap=0)
        b = q3.optimize([0], [0], [1], 6000, original=[100], cap=0, settlement='N1')
        self.assertAlmostEqual(a['q'][0], 100)
        self.assertAlmostEqual(a['cost'], 100)
        self.assertAlmostEqual(b['q'][0], 0)
        self.assertAlmostEqual(b['cost'], 50)

    def test_new_contract_is_full_one_point_five_price(self):
        a = q3.optimize([120], [0], [1], 6000, original=[100], cap=0)
        self.assertAlmostEqual(a['q'][0], 120)
        self.assertAlmostEqual(a['cost'], 130)

    def test_ac_efficiency_and_terminal(self):
        a = q3.optimize([0, 81], [0, 0], [1, 3], 6000, cap=100)
        np.testing.assert_allclose(a['q'], [100, 0], atol=1e-6)
        self.assertAlmostEqual(a['E'][-1], 6000)
        self.assertAlmostEqual(a['cost'], 100)

    def test_hourly_energy_and_release_boundary(self):
        forecasts = np.ones((365, 4, 24)) * 600
        x = q3.pv_at_issue(forecasts, 31, 6)
        self.assertEqual(len(x), 108)
        self.assertAlmostEqual(x[:6].sum(), 600)
        forecasts[31, 2:] = 99999
        np.testing.assert_array_equal(q3.pv_at_issue(forecasts, 31, 6), x)

    def test_past_contract_frozen_and_continuous_state(self):
        hourly = np.zeros((365, 4, 24))
        hourly[31, 0] = 600
        load = np.full(144, 150.)
        a = q3.simulate_day(31, load, np.zeros(144), load, hourly,
                            np.ones(144), 6000, updates=(), terminal=6000)
        b = q3.simulate_day(31, load, np.zeros(144), load, hourly,
                            np.ones(144), 6000, updates=(6, 12, 18), terminal=6000)
        np.testing.assert_allclose(a['g'], b['g'], atol=1e-6)
        np.testing.assert_allclose(b['q'][:36], b['g'][:36], atol=1e-6)
        self.assertAlmostEqual(b['E'][-1], 6000)
        q3.audit(b, load, np.zeros(144), np.ones(144), 'N0')
        for version in b['versions']:
            self.assertGreaterEqual(version['interval_id'], version['issue_hour'] * 6)

    def test_positive_release_reserve_changes_only_unexecuted_contract(self):
        hourly = np.zeros((365, 4, 24))
        load = np.full(144, 10.)
        reserves = {0: np.zeros(144), 6: np.full(108, 20.)}
        base = q3.simulate_day(31, load, np.zeros(144), load, hourly,
                               np.ones(144), 6000, updates=(6,), terminal=6000)
        safe = q3.simulate_day(31, load, np.zeros(144), load, hourly,
                               np.ones(144), 6000, updates=(6,), terminal=6000,
                               reserve_by_issue=reserves)
        np.testing.assert_allclose(base['q'][:36], safe['q'][:36], atol=1e-7)
        self.assertGreater(safe['q'][36:].sum(), base['q'][36:].sum())

    def test_zero_reserve_exactly_reproduces_uncalibrated_policy(self):
        hourly = np.zeros((365, 4, 24))
        load = np.linspace(20., 80., 144)
        zeros = {0: np.zeros(144), 6: np.zeros(108),
                 12: np.zeros(72), 18: np.zeros(36)}
        base = q3.simulate_day(31, load, np.zeros(144), load, hourly,
                               np.ones(144), 6000, terminal=6000)
        same = q3.simulate_day(31, load, np.zeros(144), load, hourly,
                               np.ones(144), 6000, terminal=6000,
                               reserve_by_issue=zeros)
        for key in ('g', 'q', 'c', 'd', 'e', 'E'):
            np.testing.assert_allclose(base[key], same[key], atol=1e-7)
        self.assertAlmostEqual(base['cost'], same['cost'])

    def test_negative_net_reserve_is_represented_without_negative_load(self):
        hourly = np.zeros((365, 4, 24))
        forecast_load = np.full(144, 10.)
        reserves = {0: np.full(144, -30.)}
        result = q3.simulate_day(
            31, np.zeros(144), np.zeros(144), forecast_load, hourly,
            np.ones(144), 6000, updates=(), terminal=6000,
            reserve_by_issue=reserves)
        self.assertTrue(np.isfinite(result['q']).all())
        self.assertTrue((result['q'] >= -1e-9).all())


if __name__ == '__main__':
    unittest.main()
