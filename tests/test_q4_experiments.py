import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from q4_experiments import load_inputs, simulate_policy
from q4_solver import q4_reserve_q2


def test_q4_reserve_skips_warmup_nan_forecasts():
    actual = np.ones((20, 144), dtype=float)
    forecast = np.ones((20, 144), dtype=float)
    forecast[:7] = np.nan
    reserve = q4_reserve_q2(10, actual, actual, forecast, actual, 0.8)
    assert reserve.shape == (144,)
    assert np.isfinite(reserve).all()


def test_short_q4_policy_has_actual_price_and_plan_price_columns():
    inputs = load_inputs()
    daily, dispatch, versions = simulate_policy(
        inputs, 'q4-2', 'P2p', (), 0.8, 0.8, 'N0', start=31, end=32)
    assert len(daily) == 1
    assert {'price', 'plan_price', 'g', 'q'} <= set(dispatch.columns)
    assert np.isfinite(dispatch[['price', 'plan_price']].to_numpy()).all()
    assert len(versions) == 0
