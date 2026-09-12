"""Question 4 contract planning and causal execution.

Planning prices and actual settlement prices are deliberately separate inputs.
The LP/MILP and F0 implementation are reused from the verified Q3 baseline;
this module only supplies the Q4 information boundary and accounting.
"""
from __future__ import annotations

import numpy as np

from dispatch_control import f0_step
from q3_forecast import pv_release_10min, residual_samples, risk_reserve
from q3_solver import CAP, _planning_components, optimize, settle


def _vec(value, name, n=None, positive=False):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (n is not None and len(result) != n):
        raise ValueError(f"{name} must be a vector of length {n or 'n'}")
    if not np.isfinite(result).all() or (positive and np.any(result <= 0)):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")
    return result


def q4_reserve_q2(day, actual_load, actual_pv, forecast_load, forecast_pv,
                  alpha, lookback=56):
    """Per-slot net-load residual quantiles for the Q4-2 forecast family."""
    load = np.asarray(actual_load, dtype=float)
    pv = np.asarray(actual_pv, dtype=float)
    fl = np.asarray(forecast_load, dtype=float)
    fp = np.asarray(forecast_pv, dtype=float)
    if load.ndim != 2 or load.shape != pv.shape or load.shape != fl.shape:
        raise ValueError("actual and forecast matrices must have equal day-by-144 shapes")
    if not 0 <= day < len(load) or day < 1:
        raise ValueError("day must have completed history")
    # The shared Q2 forecast cache is intentionally unavailable for days 0-6.
    history = np.arange(max(7, day - lookback), day)
    residual = (load[history] - pv[history]) - (fl[history] - fp[history])
    if not len(residual) or not np.isfinite(residual).all():
        raise ValueError("Q4-2 residual history is empty or non-finite")
    return np.quantile(residual, alpha, axis=0)


def q4_reserves_q3(day, actual_load, actual_pv, forecast_load, hourly_pv,
                   updates, alpha0, alpha_update, lookback=56):
    """Build Q4-3 reserves using the same causal release residual definition as Q3."""
    reserves = {}
    for issue_hour in (0,) + tuple(updates):
        samples = residual_samples(
            day, issue_hour, actual_load, actual_pv, forecast_load, hourly_pv,
            lookback=lookback)
        reserves[issue_hour] = risk_reserve(
            samples, alpha0 if issue_hour == 0 else alpha_update)
    return reserves


def _reserve(reserve_by_issue, issue_hour, length):
    if reserve_by_issue is None or issue_hour not in reserve_by_issue:
        return np.zeros(length, dtype=float)
    reserve = _vec(reserve_by_issue[issue_hour], "risk reserve", length)
    return reserve


def simulate_q4_day(day, load, pv, forecast_load, forecast_pv_q2,
                    hourly_pv, plan_price, actual_price, initial,
                    mode="q4-2", updates=(), settlement="N0", terminal=None,
                    reserve_by_issue=None, price_forecast_by_issue=None,
                    planning_target=6000.0):
    """Plan and execute one Q4 day under the specified information boundary."""
    if mode not in ("q4-2", "q4-3"):
        raise ValueError("mode must be q4-2 or q4-3")
    if any(hour not in (6, 12, 18) for hour in updates):
        raise ValueError("invalid update hour")
    if mode == "q4-2" and updates:
        raise ValueError("Q4-2 does not allow intraday updates")
    load = _vec(load, "load", 144)
    pv = _vec(pv, "pv", 144)
    forecast_load = _vec(forecast_load, "forecast_load", 144)
    forecast_pv_q2 = _vec(forecast_pv_q2, "forecast_pv_q2", 144)
    actual_price = _vec(actual_price, "actual_price", 144, positive=True)
    plan_price = _vec(plan_price, "plan_price", 144, positive=True)
    if not 1200 <= float(initial) <= 10800:
        raise ValueError("initial outside battery bounds")
    hourly = np.asarray(hourly_pv, dtype=float)
    if hourly.ndim != 3 or hourly.shape[1:] != (4, 24):
        raise ValueError("hourly_pv must have shape (days, 4, 24)")

    if mode == "q4-2":
        planning_pv = forecast_pv_q2
    else:
        planning_pv = pv_release_10min(hourly, day, 0)
    adjusted_load, adjusted_pv = _planning_components(
        forecast_load, planning_pv, _reserve(reserve_by_issue, 0, 144))
    first = optimize(adjusted_load, adjusted_pv, plan_price, float(initial),
                     target=planning_target)
    g = first["q"].copy()
    q = g.copy()
    solves = [first]
    flows = {key: np.zeros(144, dtype=float)
             for key in ("c", "d", "e", "unused", "spill")}
    energy = np.zeros(145, dtype=float)
    energy[0] = float(initial)
    versions = []

    for t in range(144):
        hour = t // 6
        if mode == "q4-3" and t % 36 == 0 and hour in updates:
            if price_forecast_by_issue is None or hour not in price_forecast_by_issue:
                raise ValueError(f"missing price forecast for update {hour}")
            update_price = _vec(price_forecast_by_issue[hour],
                                "update price forecast", 144 - t, positive=True)
            update_pv = pv_release_10min(hourly, day, hour)
            adjusted_load, adjusted_pv = _planning_components(
                forecast_load[t:], update_pv,
                _reserve(reserve_by_issue, hour, 144 - t))
            solved = optimize(
                adjusted_load, adjusted_pv, update_price, energy[t],
                original=g[t:], settlement=settlement, target=planning_target)
            previous = q[t:].copy()
            q[t:] = solved["q"]
            solves.append(solved)
            versions.extend(
                dict(issue_hour=hour, interval_id=j, original_g=float(g[j]),
                     previous_q=float(previous[j - t]), q=float(q[j]),
                     delta=float(q[j] - g[j]),
                     change=float(q[j] - previous[j - t]))
                for j in range(t, 144))
        step = f0_step(
            q[t], load[t], pv[t], energy[t], remaining_slots=143 - t,
            closing_target=terminal)
        for key in flows:
            flows[key][t] = step[key]
        energy[t + 1] = step["next_state"]
    if terminal is not None and abs(energy[-1] - float(terminal)) > 1e-6:
        raise RuntimeError("terminal SOC is not reachable")
    result = dict(g=g, q=q, E=energy, versions=versions, **flows)
    result["actual_grid"] = q - result["unused"] + result["e"]
    result["cost"] = float(
        settle(g, q, actual_price, settlement).sum()
        + 5 * actual_price @ result["e"])
    result["plan_price"] = plan_price
    result["milp_solves"] = int(sum(item["milp"] for item in solves))
    result["max_lp_gap"] = float(max(
        (item["gap"] for item in solves if not item["milp"]), default=0.0))
    result["max_mip_gap"] = float(max(
        (item["gap"] for item in solves if item["milp"]), default=0.0))
    audit_q4(result, load, pv, actual_price, settlement)
    return result


def audit_q4(result, load, pv, price, settlement="N0"):
    load = _vec(load, "load", 144)
    pv = _vec(pv, "pv", 144)
    price = _vec(price, "price", 144, positive=True)
    checks = {
        "bus": float(np.max(abs(result["q"] - result["unused"] + result["e"]
                                 + pv - result["spill"] + result["d"] - load
                                 - result["c"]))),
        "state": float(np.max(abs(np.diff(result["E"])
                                   - 0.9 * result["c"] + result["d"] / 0.9))),
        "bounds": float(max(0.0, 1200 - result["E"].min(),
                             result["E"].max() - 10800)),
        "power": float(max(0.0, result["c"].max() - CAP,
                            result["d"].max() - CAP)),
        "disposal": float(max(0.0, np.max(result["unused"] - result["q"]),
                               np.max(result["spill"] - pv))),
        "overlap": int(np.sum((result["c"] > 1e-6)
                               & (result["d"] > 1e-6))),
        "cost": abs(float(settle(result["g"], result["q"], price, settlement).sum()
                            + 5 * price @ result["e"]) - result["cost"]),
    }
    if any(value > 1e-6 for value in checks.values()):
        raise RuntimeError(str(checks))
    return checks


__all__ = ["q4_reserve_q2", "q4_reserves_q3", "simulate_q4_day",
           "audit_q4", "settle"]
