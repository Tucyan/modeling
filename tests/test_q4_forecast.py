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


def test_price_update_uses_last_six_slots_and_exponential_decay():
    prices = np.ones((30, 144), dtype=float)
    observed_prefix = np.r_[np.full(30, 9.0), np.full(6, 3.0)]

    baseline = forecast_price(
        prices, day=20, issue_hour=6, model='P0p',
        observed_prefix=observed_prefix, correction_gamma=0)
    corrected = forecast_price(
        prices, day=20, issue_hour=6, model='P0p',
        observed_prefix=observed_prefix, correction_gamma=1)

    np.testing.assert_allclose(baseline, np.ones(108))
    np.testing.assert_allclose(corrected[0], 3.0)
    np.testing.assert_allclose(corrected[36], 1.0 + 2.0 * np.exp(-1.0))


def test_price_update_rejects_nonbinary_correction_gamma():
    prices = np.ones((20, 144), dtype=float)
    try:
        forecast_price(
            prices, day=14, issue_hour=6, model='P0p',
            observed_prefix=np.ones(36), correction_gamma=0.5)
    except ValueError as error:
        assert 'correction_gamma' in str(error)
    else:
        raise AssertionError('non-binary correction_gamma was accepted')
