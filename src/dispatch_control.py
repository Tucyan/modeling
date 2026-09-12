"""Shared causal F0 execution controller for Q2 and Q3.

All arguments and returned flows are AC-side kWh for one 10-minute slot.
"""
from __future__ import annotations

import numpy as np


CAP = 5000.0 / 6.0
LOWER_SOC = 1200.0
UPPER_SOC = 10800.0
CHARGE_EFFICIENCY = 0.9
DISCHARGE_EFFICIENCY = 0.9


def _finite_nonnegative(name, value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def f0_step(contract, load, pv, state, *, remaining_slots,
            closing_target=None, cap=CAP):
    """Execute one slot using the shared F0/safe terminal rule.

    ``remaining_slots`` counts slots after the current slot. ``closing_target``
    constrains only physical evaluation closure; it does not alter the
    contract. The returned ``next_state`` is the battery boundary after this
    slot.
    """
    contract = _finite_nonnegative('contract', contract)
    load = _finite_nonnegative('load', load)
    pv = _finite_nonnegative('pv', pv)
    state = float(state)
    if not np.isfinite(state) or not LOWER_SOC <= state <= UPPER_SOC:
        raise ValueError('state outside battery bounds')
    if not isinstance(remaining_slots, (int, np.integer)) or remaining_slots < 0:
        raise ValueError('remaining_slots must be a nonnegative integer')
    cap = float(cap)
    if not np.isfinite(cap) or cap < 0:
        raise ValueError('cap must be finite and nonnegative')
    if closing_target is not None:
        closing_target = float(closing_target)
        if (not np.isfinite(closing_target)
                or not LOWER_SOC <= closing_target <= UPPER_SOC):
            raise ValueError('closing_target outside battery bounds')

    surplus = contract + pv - load
    if closing_target is not None and state > closing_target:
        c = 0.0
        d = min(cap, load, CHARGE_EFFICIENCY * (state - closing_target))
        next_state = state - d / DISCHARGE_EFFICIENCY
    else:
        delta = (CHARGE_EFFICIENCY * max(surplus, 0.0)
                 - max(-surplus, 0.0) / DISCHARGE_EFFICIENCY)
        if closing_target is None:
            floor, ceiling = LOWER_SOC, UPPER_SOC
        else:
            floor = max(LOWER_SOC,
                        closing_target - remaining_slots * CHARGE_EFFICIENCY * cap)
            ceiling = closing_target
        next_state = np.clip(
            state + delta,
            max(floor, state - cap / DISCHARGE_EFFICIENCY),
            min(ceiling, state + CHARGE_EFFICIENCY * cap),
        )
        c = max(next_state - state, 0.0) / CHARGE_EFFICIENCY
        d = max(state - next_state, 0.0) * DISCHARGE_EFFICIENCY

    if closing_target is not None:
        min_next = max(LOWER_SOC,
                       closing_target - remaining_slots * CHARGE_EFFICIENCY * cap)
        if next_state < min_next - 1e-8:
            raise ValueError('closing target unreachable under remaining power')
        if state <= closing_target and next_state > closing_target + 1e-8:
            raise ValueError('closing target exceeded')
        if remaining_slots == 0 and abs(next_state - closing_target) > 1e-8:
            raise ValueError('closing target unreachable in final slot')

    balance = surplus + d - c
    emergency = max(-balance, 0.0)
    excess = max(balance, 0.0)
    unused = min(contract, excess)
    spill = max(excess - unused, 0.0)
    if spill > pv + 1e-7:
        raise ValueError('execution would discard battery energy as spill')
    next_state = state + CHARGE_EFFICIENCY * c - d / DISCHARGE_EFFICIENCY
    return dict(c=float(c), d=float(d), e=float(emergency),
                unused=float(unused), spill=float(spill),
                next_state=float(next_state))
