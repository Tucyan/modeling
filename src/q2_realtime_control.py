"""Causal remaining-day MPC for the second-question storage controller.

All power-like inputs and outputs are AC-side energy per ten-minute interval
(kWh), matching :mod:`q2_solver`.  ``g`` is an already paid and frozen
contract schedule; this controller only chooses storage actions.
"""

from pathlib import Path
import os
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".local-deps"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, hstack, lil_matrix


CAP = 5000.0 / 6.0
EMIN = 1200.0
EMAX = 10800.0
ETA_C = 0.9
ETA_D = 0.9


def _as_vector(value, name, n=None):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and len(result) != n:
        raise ValueError(f"{name} must have length {n}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _solve_remaining(
    g, net, price, initial, terminal, discharge_cap=None, charge_cap=None
):
    """Solve one deterministic remaining-horizon problem.

    The first element of ``net`` is the currently observed actual net load;
    all later elements are forecasts available at this control instant.
    """
    n = len(g)
    if discharge_cap is None:
        discharge_cap = np.full(n, CAP, dtype=float)
    else:
        discharge_cap = _as_vector(discharge_cap, "discharge_cap", n)
        if np.any(discharge_cap < 0):
            raise ValueError("discharge_cap must be nonnegative")
    if charge_cap is None:
        charge_cap = np.full(n, CAP, dtype=float)
    else:
        charge_cap = _as_vector(charge_cap, "charge_cap", n)
        if np.any(charge_cap < 0):
            raise ValueError("charge_cap must be nonnegative")
    # c, d, E[0:n+1], emergency, spill
    c0 = 0
    d0 = n
    e0 = 2 * n
    emergency0 = 3 * n + 1
    spill0 = emergency0 + n
    size = spill0 + n

    equality = lil_matrix((2 * n, size), dtype=float)
    rhs = np.zeros(2 * n, dtype=float)
    for t in range(n):
        energy_row = t
        equality[energy_row, c0 + t] = ETA_C
        equality[energy_row, d0 + t] = -1.0 / ETA_D
        equality[energy_row, e0 + t] = 1.0
        equality[energy_row, e0 + t + 1] = -1.0

        balance_row = n + t
        equality[balance_row, c0 + t] = -1.0
        equality[balance_row, d0 + t] = 1.0
        equality[balance_row, emergency0 + t] = 1.0
        equality[balance_row, spill0 + t] = -1.0
        rhs[balance_row] = net[t] - g[t]
    equality = equality.tocsr()

    cost = np.zeros(size, dtype=float)
    # Tiny numerical tie-break; exclude it from reported physical purchase cost.
    cost[c0 : c0 + n] = 1e-5
    cost[d0 : d0 + n] = 1e-5
    cost[emergency0 : emergency0 + n] = 5.0 * price

    lower = np.zeros(size, dtype=float)
    upper = np.full(size, np.inf, dtype=float)
    lower[e0 : e0 + n + 1] = EMIN
    upper[e0 : e0 + n + 1] = EMAX
    lower[e0] = upper[e0] = initial
    if terminal is not None:
        lower[e0 + n] = upper[e0 + n] = terminal
    upper[c0 : c0 + n] = np.minimum(CAP, charge_cap)
    upper[d0 : d0 + n] = np.minimum(CAP, discharge_cap)

    started = time.perf_counter()
    result = linprog(
        cost,
        A_eq=equality,
        b_eq=rhs,
        bounds=list(zip(lower, upper)),
        method="highs",
        options={"time_limit": 30},
    )
    if not result.success:
        raise RuntimeError(f"remaining-horizon LP failed: {result.message}")

    x = result.x
    used_milp = bool(
        np.any((x[c0 : c0 + n] > 1e-7) & (x[d0 : d0 + n] > 1e-7))
    )
    gap = 0.0
    lp_solves = 1
    milp_solves = 0
    if used_milp:
        # c_t <= CAP*y_t and d_t <= CAP*(1-y_t).
        coupling = lil_matrix((2 * n, size + n), dtype=float)
        coupling_rhs = np.r_[np.zeros(n), np.full(n, CAP)]
        for t in range(n):
            coupling[t, c0 + t] = 1.0
            coupling[t, size + t] = -CAP
            coupling[n + t, d0 + t] = 1.0
            coupling[n + t, size + t] = CAP
        extended_equalities = hstack(
            [equality, csr_matrix((equality.shape[0], n))], format="csr"
        )
        result = milp(
            np.r_[cost, np.zeros(n)],
            integrality=np.r_[np.zeros(size), np.ones(n)],
            bounds=Bounds(np.r_[lower, np.zeros(n)], np.r_[upper, np.ones(n)]),
            constraints=[
                LinearConstraint(extended_equalities, rhs, rhs),
                LinearConstraint(coupling.tocsr(), -np.inf, coupling_rhs),
            ],
            options={"time_limit": 30, "mip_rel_gap": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"remaining-horizon MILP failed: {result.message}")
        x = result.x[:size]
        gap = float(result.mip_gap or 0.0)
        milp_solves = 1

    residual = np.max(np.abs(equality @ x - rhs))
    if residual > 1e-5:
        raise RuntimeError(f"remaining-horizon balance residual is {residual}")
    if np.any((x[c0 : c0 + n] > 1e-5) & (x[d0 : d0 + n] > 1e-5)):
        raise RuntimeError("storage charge/discharge exclusivity was not satisfied")

    elapsed = time.perf_counter() - started
    return {
        "c": x[c0 : c0 + n],
        "d": x[d0 : d0 + n],
        "E": x[e0 : e0 + n + 1],
        "solver_seconds": elapsed,
        "milp": used_milp,
        "gap": gap,
        "lp_solves": lp_solves,
        "milp_solves": milp_solves,
    }


def execute_mpc(g, load, pv, initial, forecast_net, price, terminal=None):
    """Execute a causal remaining-day MPC controller.

    At slot ``t``, ``load[t]`` and ``pv[t]`` are treated as the newly observed
    actual values.  The optimization uses that actual net load for its first
    slot and ``forecast_net[t+1:]`` for every later slot.  A fresh remaining-
    horizon optimization is solved after each executed slot.

    ``terminal`` is an optional final battery energy target.  With ``None``
    there is no terminal penalty or terminal target, so the remaining-day
    objective is only five-times emergency purchase cost.
    """
    g = _as_vector(g, "g")
    n = len(g)
    load = _as_vector(load, "load", n)
    pv = _as_vector(pv, "pv", n)
    forecast_net = _as_vector(forecast_net, "forecast_net", n)
    price = _as_vector(price, "price", n)
    if np.any(g < 0):
        raise ValueError("g must be nonnegative")
    if np.any(price < 0):
        raise ValueError("price must be nonnegative")
    if not (EMIN - 1e-8 <= initial <= EMAX + 1e-8):
        raise ValueError(f"initial must be in [{EMIN}, {EMAX}]")
    initial = float(initial)
    if terminal is not None:
        terminal = float(terminal)
        if not (EMIN - 1e-8 <= terminal <= EMAX + 1e-8):
            raise ValueError(f"terminal must be in [{EMIN}, {EMAX}]")

    started = time.perf_counter()
    actual_net = load - pv
    c = np.zeros(n, dtype=float)
    d = np.zeros(n, dtype=float)
    emergency = np.zeros(n, dtype=float)
    unused = np.zeros(n, dtype=float)
    spill = np.zeros(n, dtype=float)
    energy = np.zeros(n + 1, dtype=float)
    energy[0] = initial
    solver_seconds = 0.0
    milp_used = False
    gap = 0.0
    lp_solves = 0
    milp_solves = 0

    for t in range(n):
        remaining_g = g[t:]
        remaining_net = np.r_[actual_net[t], forecast_net[t + 1 :]]
        # The current gross load is observed.  Future gross load is unknown,
        # so the available net-load forecast gives a conservative discharge
        # cap and prevents battery energy from being treated as saleable power.
        remaining_discharge_cap = np.r_[
            max(load[t], 0.0), np.maximum(forecast_net[t + 1 :], 0.0)
        ]
        # Consistent with the project's paid 5x restoration interpretation:
        # extra imports can also supply charging, and all are fully charged.
        remaining_charge_cap = np.full(len(remaining_g),CAP)
        solved = _solve_remaining(
            remaining_g,
            remaining_net,
            price[t:],
            energy[t],
            terminal,
            remaining_discharge_cap,
            remaining_charge_cap,
        )
        c[t] = solved["c"][0]
        d[t] = solved["d"][0]
        energy[t + 1] = energy[t] + ETA_C * c[t] - d[t] / ETA_D
        balance = g[t] + pv[t] - load[t] + d[t] - c[t]
        emergency[t] = max(-balance, 0.0)
        excess = max(balance, 0.0)
        unused[t] = min(g[t], excess)
        spill[t] = max(excess - unused[t], 0.0)

        solver_seconds += solved["solver_seconds"]
        milp_used = milp_used or solved["milp"]
        gap = max(gap, solved["gap"])
        lp_solves += solved["lp_solves"]
        milp_solves += solved["milp_solves"]

    elapsed = time.perf_counter() - started
    return {
        "g": g.copy(),
        "c": c,
        "d": d,
        "e": emergency,
        "unused": unused,
        "spill": spill,
        "E": energy,
        "seconds": elapsed,
        "solver_seconds": solver_seconds,
        "milp": milp_used,
        "gap": gap,
        "steps": n,
        "solver_stats": {
            "lp_solves": lp_solves,
            "milp_solves": milp_solves,
            "solver_seconds": solver_seconds,
            "gap": gap,
        },
    }


__all__ = ["execute_mpc"]
