import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class RealtimeMpcTests(unittest.TestCase):
    def test_closure_can_buy_energy_at_penalty_price(self):
        result=self.solver()(np.zeros(12),np.zeros(12),np.zeros(12),1200.,np.zeros(12),np.ones(12),terminal=6000.)
        self.assertAlmostEqual(result['E'][-1],6000.,places=6)
        self.assertAlmostEqual(result['e'].sum(),4800/.9,places=6)
    def solver(self):
        self.assertIsNotNone(
            importlib.util.find_spec("q2_realtime_control"),
            "q2 realtime controller is not implemented",
        )
        from q2_realtime_control import execute_mpc

        return execute_mpc

    def test_high_price_gap_saves_limited_storage_for_expensive_slot(self):
        execute_mpc = self.solver()
        result = execute_mpc(
            g=np.zeros(2),
            load=np.array([1000.0, 1000.0]),
            pv=np.zeros(2),
            initial=2033.3333333333333,
            forecast_net=np.array([1000.0, 1000.0]),
            price=np.array([1.0, 10.0]),
        )

        self.assertAlmostEqual(result["d"][0], 0.0, places=6)
        self.assertAlmostEqual(result["d"][1], 5000/6, places=6)
        self.assertAlmostEqual(result["e"][0], 1000.+(5000/6-750)/.81, places=6)
        self.assertAlmostEqual(result["e"][1], 1000.-5000/6, places=6)

    def test_high_price_current_gap_is_served_before_cheaper_future_gap(self):
        execute_mpc = self.solver()
        result = execute_mpc(
            g=np.zeros(2),
            load=np.array([1000.0, 1000.0]),
            pv=np.zeros(2),
            initial=2033.3333333333333,
            forecast_net=np.array([1000.0, 1000.0]),
            price=np.array([10.0, 1.0]),
        )

        self.assertAlmostEqual(result["d"][0], 750.0, places=6)
        self.assertAlmostEqual(result["d"][1], 0.0, places=6)
        self.assertAlmostEqual(result["e"][0], 250.0, places=6)
        self.assertAlmostEqual(result["e"][1], 1000.0, places=6)

    def test_prefix_is_causal_to_future_actual_changes(self):
        execute_mpc = self.solver()
        common = dict(
            g=np.array([0.0, 0.0, 0.0, 0.0]),
            pv=np.zeros(4),
            initial=1200.0,
            forecast_net=np.array([100.0, 100.0, 100.0, 100.0]),
            price=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        first = execute_mpc(load=np.array([100.0, 100.0, 100.0, 100.0]), **common)
        changed = execute_mpc(load=np.array([100.0, 100.0, 9999.0, 9999.0]), **common)

        np.testing.assert_allclose(first["c"][:2], changed["c"][:2], atol=1e-7)
        np.testing.assert_allclose(first["d"][:2], changed["d"][:2], atol=1e-7)
        np.testing.assert_allclose(first["E"][:3], changed["E"][:3], atol=1e-7)

    def test_paid_contract_is_frozen_and_unusable_excess_is_reported(self):
        execute_mpc = self.solver()
        g = np.array([100.0, 100.0])
        result = execute_mpc(
            g=g,
            load=np.zeros(2),
            pv=np.zeros(2),
            initial=10800.0,
            forecast_net=np.zeros(2),
            price=np.array([1.0, 1.0]),
        )

        np.testing.assert_array_equal(result["g"], g)
        self.assertAlmostEqual(result["c"].sum(), 0.0, places=6)
        self.assertAlmostEqual(result["unused"].sum(), 200.0, places=6)
        self.assertAlmostEqual(result["spill"].sum(), 0.0, places=6)

    def test_terminal_target_is_enforced_and_bounds_are_respected(self):
        execute_mpc = self.solver()
        result = execute_mpc(
            g=np.full(12, 1000.0),
            load=np.zeros(12),
            pv=np.zeros(12),
            initial=1200.0,
            forecast_net=np.zeros(12),
            price=np.ones(12),
            terminal=6000.0,
        )

        self.assertAlmostEqual(result["E"][0], 1200.0, places=6)
        self.assertAlmostEqual(result["E"][-1], 6000.0, places=6)
        self.assertGreaterEqual(result["E"].min(), 1200.0 - 1e-7)
        self.assertLessEqual(result["E"].max(), 10800.0 + 1e-7)
        self.assertLessEqual(result["c"].max(), 5000.0 / 6.0 + 1e-7)
        self.assertLessEqual(result["d"].max(), 5000.0 / 6.0 + 1e-7)

    def test_result_contains_solver_statistics_and_physical_balance(self):
        execute_mpc = self.solver()
        result = execute_mpc(
            g=np.array([50.0, 0.0, 100.0]),
            load=np.array([0.0, 200.0, 0.0]),
            pv=np.zeros(3),
            initial=6000.0,
            forecast_net=np.array([0.0, 200.0, 0.0]),
            price=np.array([1.0, 3.0, 1.0]),
        )

        for key in ("seconds", "solver_seconds", "milp", "gap", "steps"):
            self.assertIn(key, result)
        self.assertEqual(result["steps"], 3)
        np.testing.assert_allclose(
            result["E"][1:], result["E"][:-1] + 0.9 * result["c"] - result["d"] / 0.9
        )
        np.testing.assert_allclose(
            result["g"]
            - result["c"]
            + result["d"]
            + result["e"]
            - result["unused"]
            - result["spill"],
            np.array([0.0, 200.0, 0.0]),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
