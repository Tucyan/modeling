"""Adapters for replaying the formal Q2 Rolling_T0 ledger in Q3."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from q3_solver import ROOT, settle, simulate_day


Q2_OUT = ROOT / 'outputs/q2_review_v2/experiments'


def load_q2_frozen(root=ROOT):
    """Load Q2's frozen Rolling_T0 ledger without reconstructing contracts."""
    root = Path(root)
    folder = root / 'outputs/q2_review_v2/experiments'
    daily = pd.read_csv(folder / 'Rolling_T0_daily.csv')
    dispatch = pd.read_csv(folder / 'Rolling_T0_dispatch.csv')
    required = {'date', 'interval_id', 'g', 'c', 'd', 'e', 'unused',
                'spill', 'E_start', 'E_end', 'load', 'pv', 'price'}
    if required - set(dispatch.columns):
        raise ValueError(f'Q2 ledger missing columns: {sorted(required-set(dispatch.columns))}')
    expected = pd.date_range('2025-02-01', '2025-12-31').strftime('%Y-%m-%d').tolist()
    if dispatch.date.astype(str).nunique() != len(expected) or dispatch.date.astype(str).unique().tolist() != expected:
        raise ValueError('Q2 ledger dates are incomplete or unordered')
    if len(dispatch) != len(expected) * 144:
        raise ValueError('Q2 ledger must contain 334 complete days')
    if not np.isfinite(dispatch.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
        raise ValueError('Q2 ledger contains non-finite values')
    return daily, dispatch


def map_q2_dispatch(dispatch):
    """Map Q2 fields to the Q3 ledger schema with zero adjustment."""
    mapped = dispatch.copy()
    mapped['q'] = mapped['g'].to_numpy(dtype=float)
    mapped['actual_grid'] = mapped.q - mapped.unused + mapped.e
    mapped['delta'] = 0.0
    return mapped


def q2_frozen_cost(dispatch, settlement='N0'):
    mapped = map_q2_dispatch(dispatch)
    return float(settle(mapped.g, mapped.q, mapped.price, settlement).sum()
                 + 5 * mapped.price.to_numpy(dtype=float) @ mapped.e.to_numpy(dtype=float))


def replay_q2_frozen(root=ROOT, settlement='N0'):
    """Return a Q3-shaped Q2_frozen ledger after shared-controller replay."""
    daily, source = load_q2_frozen(root)
    mapped = map_q2_dispatch(source)
    replayed = []
    for index, (date, day) in enumerate(mapped.groupby('date', sort=False)):
        initial = float(day.E_start.iloc[0])
        result = replay_day_with_contract(
            day, day.g.to_numpy(dtype=float), initial,
            terminal=6000. if index == mapped.date.astype(str).nunique() - 1 else None)
        frame = day.copy()
        for key in ('c', 'd', 'e', 'unused', 'spill'):
            frame[key] = result[key]
        frame['E_start'], frame['E_end'] = result['E'][:-1], result['E'][1:]
        frame['actual_grid'] = frame.q - frame.unused + frame.e
        frame['settlement'] = settlement
        replayed.append(frame)
    mapped = pd.concat(replayed, ignore_index=True)
    mapped['cost_recomputed'] = settle(
        mapped.g.to_numpy(dtype=float), mapped.q.to_numpy(dtype=float),
        mapped.price.to_numpy(dtype=float), settlement) + 5 * mapped.price * mapped.e
    return daily, mapped


def q2_day_ahead(load, pv, price, forecasts, day, model='M1', alpha=.8,
                 scenarios=14, initial=6000.):
    """Generate a Q2-compatible day-ahead contract for an adapter comparison."""
    from q2_review_experiments import planning_input
    from q2_solver import optimize
    load = np.asarray(load, dtype=float)
    pv = np.asarray(pv, dtype=float)
    price = np.asarray(price, dtype=float)
    forecast = np.asarray(forecasts['P0' if model in ('M0', 'M1', 'R0') else 'P1'])
    net = load - pv
    predicted = forecast[:, 0] - forecast[:, 1]
    planned = planning_input(net, predicted, day, model, alpha, scenarios)
    if model == 'R0':
        return np.maximum(planned, 0.)
    return optimize(planned, price, initial)['g']


def replay_day_with_contract(day, contract, initial, terminal=None):
    """Replay one Q2 contract through Q3's physical controller."""
    day = day.sort_values('interval_id')
    contract = np.asarray(contract, dtype=float)
    if contract.shape != (144,) or not np.isfinite(contract).all() or np.any(contract < 0):
        raise ValueError('contract must be 144 finite nonnegative values')
    actual_load = day.load.to_numpy(dtype=float)
    actual_pv = day.pv.to_numpy(dtype=float)
    hourly = np.zeros((365, 4, 24), dtype=float)
    return simulate_day(31, actual_load, actual_pv, actual_load, hourly,
                        day.price.to_numpy(dtype=float), initial, updates=(),
                        terminal=terminal, original_contract=contract)
