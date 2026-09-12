import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from q4_solver import simulate_q4_day, settle


def _case(actual_price):
    load = np.full(144, 100.0)
    pv = np.zeros(144)
    hourly = np.zeros((2, 4, 24))
    return simulate_q4_day(
        day=1,
        load=load,
        pv=pv,
        forecast_load=load,
        forecast_pv_q2=pv,
        hourly_pv=hourly,
        plan_price=np.ones(144),
        actual_price=np.full(144, actual_price),
        initial=6000.0,
        mode='q4-2',
        updates=(),
        terminal=6000.0,
        reserve_by_issue={0: np.zeros(144)},
    )


def test_actual_price_changes_settlement_only():
    low = _case(2.0)
    high = _case(3.0)
    for key in ('g', 'q', 'c', 'd', 'e', 'E'):
        np.testing.assert_allclose(low[key], high[key], atol=1e-7)
    assert high['cost'] > low['cost']


def test_q4_2_and_q4_3_without_updates_degenerate():
    load = np.full(144, 100.0)
    pv = np.zeros(144)
    hourly = np.zeros((2, 4, 24))
    kwargs = dict(
        day=1, load=load, pv=pv, forecast_load=load,
        forecast_pv_q2=pv, hourly_pv=hourly,
        plan_price=np.ones(144), actual_price=np.ones(144), initial=6000.0,
        updates=(), terminal=6000.0,
        reserve_by_issue={0: np.zeros(144)},
    )
    q42 = simulate_q4_day(mode='q4-2', **kwargs)
    q43 = simulate_q4_day(mode='q4-3', **kwargs)
    for key in ('g', 'q', 'c', 'd', 'e', 'E'):
        np.testing.assert_allclose(q42[key], q43[key], atol=1e-7)


def test_settlement_formula_is_explicitly_price_parameterized():
    np.testing.assert_allclose(settle([100], [80], [2], 'N0'), [220])
    np.testing.assert_allclose(settle([100], [80], [2], 'N1'), [180])
