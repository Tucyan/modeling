import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np

from q3_q2_bridge import load_q2_frozen, replay_q2_frozen
from q3_solver import simulate_day


class Q3Q2BridgeTests(unittest.TestCase):
    def test_injected_contract_is_frozen_without_updates(self):
        hourly = np.zeros((365, 4, 24))
        contract = np.full(144, 100.)
        result = simulate_day(31, np.full(144, 100.), np.zeros(144),
                              np.full(144, 100.), hourly, np.ones(144), 6000.,
                              updates=(), terminal=6000.,
                              original_contract=contract)
        np.testing.assert_allclose(result['g'], contract)
        np.testing.assert_allclose(result['q'], contract)
        self.assertEqual(result['versions'], [])

    def test_injected_contract_is_validated(self):
        hourly = np.zeros((365, 4, 24))
        with self.assertRaises(ValueError):
            simulate_day(31, np.zeros(144), np.zeros(144), np.zeros(144),
                         hourly, np.ones(144), 6000., updates=(),
                         original_contract=np.full(143, 1.))

    def test_formal_q2_frozen_replay_is_physically_identical(self):
        _, source = load_q2_frozen()
        _, mapped = replay_q2_frozen()
        self.assertEqual(len(mapped), 334 * 144)
        np.testing.assert_allclose(mapped.q, mapped.g, rtol=0, atol=1e-9)
        for key in ('c', 'd', 'e', 'unused', 'spill', 'E_start', 'E_end'):
            np.testing.assert_allclose(mapped[key], source[key], rtol=0, atol=1e-6)
        self.assertAlmostEqual(float(mapped.cost_recomputed.sum()),
                               13991435.140429746, places=6)


if __name__ == '__main__':
    unittest.main()
