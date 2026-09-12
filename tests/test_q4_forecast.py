import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from q4_forecast import forecast_price


def test_price_forecast_uses_only_completed_days():
    prices = np.arange(40 * 144, dtype=float).reshape(40, 144) / 100.0 + 0.1
    baseline = forecast_price(prices, day=20, issue_hour=0, model='P0p')
    changed = prices.copy()
    changed[21:] += 999.0
    np.testing.assert_allclose(
        baseline, forecast_price(changed, day=20, issue_hour=0, model='P0p'))


def test_price_forecast_returns_remaining_slots_and_positive_values():
    prices = np.full((20, 144), 0.8)
    result = forecast_price(prices, day=14, issue_hour=12, model='P1p')
    assert result.shape == (72,)
    assert np.isfinite(result).all()
    assert (result > 0).all()
