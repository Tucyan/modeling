"""Selection and release-set behavior tests for formal Q3 experiments."""
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p)
                for p in ('src',)]

import numpy as np

from q3_experiments import RELEASE_SETS, choose_protocol, load_inputs


class Q3ExperimentTests(unittest.TestCase):
    def test_release_sets_cover_every_unique_intraday_subset(self):
        """Dropping or duplicating a release subset makes information-value claims incomplete."""
        self.assertEqual(RELEASE_SETS, {
            'R0': (), 'R6': (6,), 'R12': (12,), 'R18': (18,),
            'R6_12': (6, 12), 'R6_18': (6, 18),
            'R12_18': (12, 18), 'Rall': (6, 12, 18),
        })
        self.assertEqual(len(set(RELEASE_SETS.values())), 8)

    def test_protocol_selection_rejects_current_or_future_scores(self):
        """A policy cannot be selected with observations from its decision day."""
        scores = {'b': {'cost': 9., 'emergency_kwh': 5.},
                  'a': {'cost': 10., 'emergency_kwh': 1.}}
        self.assertEqual(choose_protocol(scores, 30, 31), 'b')
        with self.assertRaises(ValueError):
            choose_protocol(scores, 31, 31)

    def test_protocol_tie_break_is_deterministic(self):
        """Equal historical performance must not depend on dictionary insertion order."""
        scores = {'z': {'cost': 10., 'emergency_kwh': 2.},
                  'a': {'cost': 10., 'emergency_kwh': 2.}}
        self.assertEqual(choose_protocol(scores, 30, 31), 'a')

    def test_input_loader_accepts_only_the_defined_seven_day_warmup_gap(self):
        """Q2 forecasts intentionally start on day 7; later missing values are invalid."""
        inputs = load_inputs()
        self.assertTrue(np.isnan(inputs.load_forecast[:7]).all())
        self.assertTrue(np.isfinite(inputs.load_forecast[7:]).all())


if __name__ == '__main__':
    unittest.main()
