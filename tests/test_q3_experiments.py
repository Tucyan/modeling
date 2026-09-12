"""Selection and release-set behavior tests for formal Q3 experiments."""
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p)
                for p in ('src',)]

import numpy as np
import pandas as pd
from unittest.mock import patch

from q3_experiments import (RELEASE_SETS, choose_protocol, load_inputs,
                            select_january_protocol)


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

    def test_n1_selection_scores_every_candidate_under_n1(self):
        daily = pd.DataFrame([dict(
            cost=1., plan_cost=1., up_cost=0., down_fee=0., refund=0.,
            emergency_cost=0., emergency_kwh=0., terminal=6000.,
            max_lp_gap=0., max_mip_gap=0., milp_solves=0, seconds=0.)])
        dispatch = pd.DataFrame([dict(interval_id=0, e=0., unused=0., spill=0.)])
        seen = []
        def fake(*args, **kwargs):
            seen.append(args[3] if len(args) > 3 else kwargs['settlement'])
            return daily, dispatch, pd.DataFrame()
        with patch('q3_experiments.simulate_policy', side_effect=fake):
            protocol = select_january_protocol(object(), settlement='N1')
        self.assertEqual(protocol['settlement'], 'N1')
        self.assertEqual(seen, ['N1'] * 48)


if __name__ == '__main__':
    unittest.main()
