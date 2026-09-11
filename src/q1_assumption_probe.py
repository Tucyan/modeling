"""Reproduce Q1 assumption comparisons; no source/template files are modified."""
from pathlib import Path
import sysconfig
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / '.local-deps'
abi_suffix = (sysconfig.get_config_var('EXT_SUFFIX') or '').lower()
if abi_suffix and LOCAL_DEPS.exists() and any(str(p).lower().endswith(abi_suffix) for p in LOCAL_DEPS.rglob('*.pyd')):
    sys.path.insert(0, str(LOCAL_DEPS))
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def solve_case(data, eta, fixed_initial):
    n = len(data)
    m = 5 * n + 1
    price = data.price_yuan_per_kwh.to_numpy()
    load = data.load_kwh_10min.to_numpy()
    pv = data.pv_forecast_kwh_10min.to_numpy()
    equations = np.zeros((2 * n + 1, m))
    rhs = np.r_[load - pv, np.zeros(n + 1)]
    for t in range(n):
        equations[t, [t, n+t, 2*n+t, 3*n+t]] = [1, -1, 1, -1]
        equations[n+t, [n+t, 2*n+t, 4*n+t, 4*n+t+1]] = [-eta, 1/eta, -1, 1]
    equations[-1, [4*n, 5*n]] = [-1, 1]
    bounds = ([(0, None)] * n + [(0, 5000/6)] * (2*n)
              + [(0, float(v)) for v in pv] + [(1200, 10800)] * (n+1))
    if fixed_initial:
        bounds[4*n] = (6000, 6000)
    objective = np.r_[price, np.zeros(m-n)]
    result = linprog(objective, A_eq=equations, b_eq=rhs, bounds=bounds,
                     method='highs', options={'time_limit': 30})
    if not result.success:
        raise RuntimeError(result.message)
    g, c, d, spill = np.split(result.x[:4*n], 4)
    energy = result.x[4*n:]
    # Physical audit recomputes equations from named arrays, not the solver matrix.
    balance = float(np.max(np.abs(g+pv-spill+d-load-c)))
    storage = float(np.max(np.abs(np.diff(energy)-eta*c+d/eta)))
    tol = 1e-7
    assert balance < tol and storage < tol
    assert np.min(np.r_[g,c,d,spill]) >= -tol
    assert max(c.max(),d.max()) <= 5000/6+tol
    assert np.max(spill-pv) <= tol
    assert energy.min() >= 1200-tol and energy.max() <= 10800+tol
    assert abs(energy[-1]-energy[0]) < tol
    if fixed_initial:
        assert abs(energy[0]-6000) < tol
    overlap = int(np.sum((c>tol)&(d>tol)))
    cost = float(np.dot(price,g))
    assert abs(cost-result.fun) < tol
    return dict(eta_charge=eta, eta_discharge=eta, fixed_initial=fixed_initial,
                cost_yuan=cost, purchase_kwh=float(g.sum()),
                initial_kwh=float(energy[0]), terminal_kwh=float(energy[-1]),
                charge_kwh=float(c.sum()), discharge_kwh=float(d.sum()),
                spill_kwh=float(spill.sum()), simultaneous_intervals=overlap,
                balance_residual=balance, storage_residual=storage,
                full_model_optimal_verified=(overlap==0))


if __name__ == '__main__':
    data = pd.read_csv(ROOT/'C题_标准化数据与修正模板/C_final_delivery/data_csv/typical_day_10min.csv')
    assert data.interval_id.tolist() == list(range(144))
    cases = [solve_case(data, eta, fixed) for eta in (.9, np.sqrt(.9))
             for fixed in (True, False)]
    base = cases[0]['cost_yuan']
    for case in cases:
        case['difference_from_main_yuan'] = case['cost_yuan'] - base
    target = ROOT/'outputs/q1_model_validation/assumption_sensitivity.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cases, indent=2), encoding='utf-8')
    print(json.dumps(cases, indent=2))
