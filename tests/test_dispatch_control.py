import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np

from dispatch_control import f0_step


class DispatchControlTests(unittest.TestCase):
    def test_f0_step_reports_physical_flows_and_state(self):
        result = f0_step(100., 120., 0., 6000., remaining_slots=143)
        self.assertEqual(set(result), {'c', 'd', 'e', 'unused', 'spill', 'next_state'})
        self.assertAlmostEqual(result['c'], 0.)
        self.assertAlmostEqual(result['d'], 20.)
        self.assertAlmostEqual(result['e'], 0.)
        self.assertAlmostEqual(result['next_state'], 6000. - 20. / .9)

    def test_closing_step_never_discards_battery_energy(self):
        result = f0_step(0., 81., 0., 6090., remaining_slots=0,
                         closing_target=6000.)
        self.assertAlmostEqual(result['d'], 81.)
        self.assertAlmostEqual(result['next_state'], 6000.)
        self.assertAlmostEqual(result['spill'], 0.)

    def test_unreachable_closing_target_is_rejected(self):
        with self.assertRaises(ValueError):
            f0_step(0., 0., 0., 1200., remaining_slots=0,
                    closing_target=6000.)

    def test_soc_bounds_and_exact_power_cap(self):
        low = f0_step(0., 1000., 0., 1200., remaining_slots=10)
        self.assertAlmostEqual(low['d'], 0.)
        self.assertAlmostEqual(low['e'], 1000.)
        self.assertAlmostEqual(low['next_state'], 1200.)
        high = f0_step(1000., 0., 0., 10800., remaining_slots=10)
        self.assertAlmostEqual(high['c'], 0.)
        self.assertAlmostEqual(high['unused'], 1000.)
        capped = f0_step(2000., 0., 0., 6000., remaining_slots=10)
        self.assertAlmostEqual(capped['c'], 5000./6)
        self.assertAlmostEqual(capped['next_state'], 6750.)

    def test_final_closure_approaches_target_from_below(self):
        result = f0_step(100., 0., 0., 5910., remaining_slots=0,
                         closing_target=6000.)
        self.assertAlmostEqual(result['c'], 100.)
        self.assertAlmostEqual(result['next_state'], 6000.)


if __name__ == '__main__':
    unittest.main()
