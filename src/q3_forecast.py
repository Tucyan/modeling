"""Causal forecast releases and historical risk reserves for Question 3."""
from __future__ import annotations

import numpy as np


RELEASE_HOURS = (0, 6, 12, 18)


def _matrix(value, name, columns=144):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite day-by-{columns} matrix')
    return array


def pv_release_10min(hourly, day, issue_hour):
    """Return the available PV release as remaining 10-minute energy in kWh."""
    forecasts = np.asarray(hourly, dtype=float)
    if issue_hour not in RELEASE_HOURS:
        raise ValueError('Not a forecast release')
    if forecasts.ndim != 3 or forecasts.shape[1:] != (4, 24):
        raise ValueError('hourly must have shape (days, 4, 24)')
    if not 0 <= day < len(forecasts):
        raise ValueError('Forecast day out of range')
    # The last axis is lead time from this release, not the clock hour.
    # Exclude forecasts beyond today's midnight from the remaining-day plan.
    values = forecasts[day, issue_hour // 6, :24 - issue_hour]
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('PV forecast must be finite and nonnegative')
    return np.repeat(values, 6) / 6


def validate_causal_history(history_days, decision_day):
    history = np.asarray(history_days, dtype=int)
    if history.ndim != 1 or len(history) == 0:
        raise ValueError('Risk calibration requires completed history')
    if np.any(history < 0) or np.any(history >= decision_day):
        raise ValueError('Risk history contains unavailable observations')


def residual_samples(day, issue_hour, actual_load, actual_pv,
                     load_forecast, hourly, lookback=56):
    """Build remaining-net-load residual paths from completed prior days only."""
    if issue_hour not in RELEASE_HOURS:
        raise ValueError('Not a forecast release')
    if not isinstance(lookback, int) or lookback <= 0:
        raise ValueError('lookback must be a positive integer')
    load = _matrix(actual_load, 'actual_load')
    pv = _matrix(actual_pv, 'actual_pv')
    predicted_load = np.asarray(load_forecast, dtype=float)
    if predicted_load.ndim != 2 or predicted_load.shape[1] != 144:
        raise ValueError('load_forecast must be a day-by-144 matrix')
    if load.shape != pv.shape or load.shape != predicted_load.shape:
        raise ValueError('Actual and forecast matrices must have equal shapes')
    if not 0 <= day < len(load):
        raise ValueError('Decision day out of range')
    history = np.arange(max(7, day - lookback), day)
    validate_causal_history(history, day)
    offset = issue_hour * 6
    if not np.isfinite(predicted_load[history, offset:]).all():
        raise ValueError('Used load forecasts must be finite')
    rows = []
    for historical_day in history:
        forecast_net = (predicted_load[historical_day, offset:]
                        - pv_release_10min(hourly, historical_day, issue_hour))
        actual_net = (load[historical_day, offset:]
                      - pv[historical_day, offset:])
        rows.append(actual_net - forecast_net)
    return np.asarray(rows, dtype=float)


def risk_reserve(samples, alpha):
    """Return a separate empirical residual quantile for every remaining slot."""
    residuals = np.asarray(samples, dtype=float)
    if residuals.ndim != 2 or len(residuals) == 0 or not np.isfinite(residuals).all():
        raise ValueError('Risk reserve requires a finite nonempty residual matrix')
    if not 0 < alpha < 1:
        raise ValueError('Risk quantile must lie strictly between zero and one')
    return np.quantile(residuals, alpha, axis=0)
