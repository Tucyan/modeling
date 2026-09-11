import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class Q1Tests(unittest.TestCase):
    def solver(self):
        self.assertIsNotNone(importlib.util.find_spec('q1_solver'), 'Q1 solver is not implemented')
        from q1_solver import solve_dispatch
        return solve_dispatch

    def test_two_period_efficiency_arbitrage(self):
        r = self.solver()([0, 8.1], [0, 0], [1, 2], dt=1, initial=0, emin=0, emax=10, pmax=10)
        self.assertAlmostEqual(r['cost'], 10, places=7)
        self.assertAlmostEqual(r['charge'][0], 10, places=7)
        self.assertAlmostEqual(r['discharge'][1], 8.1, places=7)
        self.assertAlmostEqual(r['energy'][-1], 0, places=7)

    def test_price_spread_does_not_cover_losses(self):
        r = self.solver()([0, 8.1], [0, 0], [1, 1.1], dt=1, initial=0, emin=0, emax=10, pmax=10)
        self.assertAlmostEqual(r['cost'], 8.91, places=7)
        self.assertAlmostEqual(sum(r['charge']), 0, places=7)

    def test_storage_capacity_and_curtailment(self):
        r = self.solver()([0, 10], [20, 0], [1, 2], dt=1, initial=0, emin=0, emax=9, pmax=10)
        self.assertAlmostEqual(r['cost'], 3.8, places=7)
        self.assertAlmostEqual(sum(r['spill']), 10, places=7)
        self.assertLessEqual(max(r['energy']), 9+1e-7)


if __name__ == '__main__':
    unittest.main()
