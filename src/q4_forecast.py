"""Causal price forecasts for Question 4.

Prices are stored as actual 10-minute settlement prices.  Every forecast only
uses rows strictly before ``day`` plus the completed prefix of the current day.
The three deliberately simple candidates keep the January selection auditable.
"""
from __future__ import annotations

import numpy as np


PRICE_MODELS = ("P0p", "P1p", "P2p")
RELEASE_HOURS = (0, 6, 12, 18)


def _validate(prices):
    values = np.asarray(prices, dtype=float)
    if values.ndim != 2 or values.shape[1] != 144:
        raise ValueError("prices must be a day-by-144 matrix")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("prices must be finite and positive")
    return values


def _history(values, day, width):
    if not isinstance(day, (int, np.integer)) or not 1 <= day < len(values):
        raise ValueError("day must have at least one completed history day")
    return values[max(0, day - width):day]


def _base_forecast(values, day, model):
    """Return a full-day forecast from completed days only."""
    if model not in PRICE_MODELS:
        raise ValueError(f"unknown price model: {model}")
    recent7 = _history(values, day, 7)
    yesterday = values[day - 1]
    if model == "P0p":
        return recent7.mean(axis=0)
    if model == "P1p":
        # A stable ridge-like shrinkage between yesterday and the recent level.
        return 0.35 * yesterday + 0.65 * recent7.mean(axis=0)
    recent28 = _history(values, day, 28)
    weekday = day % 7
    same_weekday = values[max(0, day - 56):day]
    same_weekday = same_weekday[
        np.arange(max(0, day - 56), day) % 7 == weekday
    ]
    seasonal = (same_weekday.mean(axis=0)
                if len(same_weekday) >= 2 else recent28.mean(axis=0))
    # The tree candidate is represented by a robust local/seasonal ensemble.
    return 0.55 * recent28.mean(axis=0) + 0.45 * seasonal


def forecast_price(prices, day, issue_hour=0, model="P0p", observed_prefix=None,
                   correction_gamma=0, correction_window=6, decay_slots=36):
    """Forecast remaining prices at a release time.

    ``observed_prefix`` may contain the current day's completed prices.  Values
    after the release boundary are ignored; callers can therefore pass a full
    actual day without accidentally leaking future information.
    """
    values = _validate(prices)
    if issue_hour not in RELEASE_HOURS:
        raise ValueError("issue_hour must be one of 0, 6, 12, 18")
    if correction_gamma not in (0, 1):
        raise ValueError("correction_gamma must be 0 or 1")
    if not isinstance(correction_window, (int, np.integer)) or correction_window <= 0:
        raise ValueError("correction_window must be a positive integer")
    if not isinstance(decay_slots, (int, np.integer)) or decay_slots <= 0:
        raise ValueError("decay_slots must be a positive integer")
    if not 0 <= day < len(values):
        raise ValueError("day outside price matrix")
    if day == 0:
        raise ValueError("price forecast needs completed history")
    start = issue_hour * 6
    base = _base_forecast(values, day, model)
    if observed_prefix is not None and start:
        observed = np.asarray(observed_prefix, dtype=float)
        if observed.ndim != 1 or len(observed) < start:
            raise ValueError("observed_prefix must cover completed slots")
        observed = observed[:start]
        if not np.isfinite(observed).all() or np.any(observed <= 0):
            raise ValueError("observed_prefix must be finite and positive")
        if correction_window > start:
            raise ValueError("correction_window exceeds completed slots")
        # Equation (15): use only the most recent completed intervals.
        correction = float(np.mean(
            observed[-correction_window:]
            - base[start-correction_window:start]))
        # Equation (16): decay the correction over the remaining horizon.
        offsets = np.arange(len(base) - start, dtype=float)
        base[start:] = base[start:] + correction_gamma * correction * np.exp(
            -offsets / decay_slots)
    result = np.maximum(base[start:], 1e-6)
    if not np.isfinite(result).all():
        raise ValueError("price forecast became non-finite")
    return result


def price_forecast_matrix(prices, day, model, observed_prefix=None):
    """Return four remaining-horizon forecasts keyed by release hour."""
    return {
        hour: forecast_price(prices, day, hour, model, observed_prefix)
        for hour in RELEASE_HOURS
    }


__all__ = ["PRICE_MODELS", "RELEASE_HOURS", "forecast_price",
           "price_forecast_matrix"]
