"""Causal calibration and release-set experiments for Question 3."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd

from q3_forecast import residual_samples, risk_reserve
from q3_solver import ROOT, DATA, simulate_day


Q2_FORECASTS = ROOT / 'outputs/01a08b78-ed6b-7e60-b40d-9f266855d1ef/q2/forecasts.npz'
MODEL_OUT = ROOT / 'outputs/q3_modeling'
FINAL_OUT = ROOT / 'outputs/q3_final'
EPS = 1e-7

RELEASE_SETS = {
    'R0': (), 'R6': (6,), 'R12': (12,), 'R18': (18,),
    'R6_12': (6, 12), 'R6_18': (6, 18),
    'R12_18': (12, 18), 'Rall': (6, 12, 18),
}


@dataclass(frozen=True)
class RiskPair:
    alpha0: float
    alpha_update: float

    @property
    def name(self):
        return f'a0_{self.alpha0:.2f}_au_{self.alpha_update:.2f}'


RISK_PAIRS = (
    RiskPair(.70, .65), RiskPair(.75, .70), RiskPair(.80, .70),
    RiskPair(.80, .75), RiskPair(.85, .75), RiskPair(.90, .80),
)


@dataclass(frozen=True)
class Q3Inputs:
    load: np.ndarray
    pv: np.ndarray
    price: np.ndarray
    load_forecast: np.ndarray
    hourly_pv: np.ndarray


def choose_protocol(scores, last_history_day, decision_day):
    """Choose deterministically using completed history only."""
    if last_history_day >= decision_day:
        raise ValueError('Selection contains unavailable observations')
    if not scores:
        raise ValueError('No protocol scores')
    return min(scores, key=lambda name: (
        float(scores[name]['cost']), float(scores[name]['emergency_kwh']), name))


def load_inputs():
    actual = pd.read_csv(DATA / 'annual_actual_10min.csv')
    if len(actual) != 365 * 144 or actual.duplicated(['date', 'interval_id']).any():
        raise ValueError('Invalid actual-data keys')
    load = actual.load_kwh_10min.to_numpy(dtype=float).reshape(365, 144)
    pv = actual.pv_actual_kwh_10min.to_numpy(dtype=float).reshape(365, 144)
    price = pd.read_csv(DATA / 'typical_day_10min.csv').price_yuan_per_kwh.to_numpy(dtype=float)
    with np.load(Q2_FORECASTS) as forecasts:
        load_forecast = forecasts['P1'][:, 0, :].copy()
    release = pd.read_csv(DATA / 'pv_forecast_hourly.csv')
    release['day'] = (pd.to_datetime(release.issue_date) - pd.Timestamp('2025-01-01')).dt.days
    release['row'] = release.issue_time.str.split(':').str[0].astype(int) // 6
    if len(release) != 35040 or release.duplicated(['day', 'row', 'horizon_hour']).any():
        raise ValueError('Invalid forecast-release keys')
    hourly = np.full((365, 4, 24), np.nan)
    hourly[release.day, release.row, release.horizon_hour - 1] = release.pv_forecast_kw
    if any(not np.isfinite(value).all() for value in (load, pv, price, hourly)):
        raise ValueError('Non-finite Q3 input')
    if not np.isnan(load_forecast[:7]).all() or not np.isfinite(load_forecast[7:]).all():
        raise ValueError('Unexpected load-forecast availability pattern')
    return Q3Inputs(load, pv, price, load_forecast, hourly)


def reserves_for_day(inputs, day, updates, risk_pair):
    if risk_pair is None:
        return None
    releases = (0,) + tuple(updates)
    reserves = {}
    for issue_hour in releases:
        samples = residual_samples(
            day, issue_hour, inputs.load, inputs.pv, inputs.load_forecast,
            inputs.hourly_pv, lookback=56)
        alpha = risk_pair.alpha0 if issue_hour == 0 else risk_pair.alpha_update
        reserves[issue_hour] = risk_reserve(samples, alpha)
    return reserves


def _summary(daily, dispatch):
    positive = dispatch.e.to_numpy() > EPS
    day_start = dispatch.interval_id.to_numpy() == 0
    event_start = positive & (~np.r_[False, positive[:-1]] | day_start)
    return dict(
        days=int(len(daily)), cost=float(daily.cost.sum()),
        plan_cost=float(daily.plan_cost.sum()), up_cost=float(daily.up_cost.sum()),
        down_fee=float(daily.down_fee.sum()), refund=float(daily.refund.sum()),
        emergency_cost=float(daily.emergency_cost.sum()),
        emergency_kwh=float(daily.emergency_kwh.sum()),
        emergency_days=int((daily.emergency_kwh > EPS).sum()),
        emergency_events=int(event_start.sum()), emergency_slots=int(positive.sum()),
        unused_kwh=float(dispatch.unused.sum()), spill_kwh=float(dispatch.spill.sum()),
        final_soc=float(daily.terminal.iloc[-1]),
        max_lp_gap=float(daily.max_lp_gap.max()),
        max_mip_gap=float(daily.max_mip_gap.max()),
        milp_solves=int(daily.milp_solves.sum()),
        max_seconds=float(daily.seconds.max()), mean_seconds=float(daily.seconds.mean()))


def simulate_policy(inputs, updates, risk_pair, settlement='N0', start=31, end=365):
    if not 7 < start < end <= 365:
        raise ValueError('Invalid evaluation range')
    if tuple(updates) not in RELEASE_SETS.values():
        raise ValueError('Unknown release subset')
    state = 6000.
    days, chunks, versions = [], [], []
    for day in range(start, end):
        tick = time.perf_counter()
        reserves = reserves_for_day(inputs, day, updates, risk_pair)
        result = simulate_day(
            day, inputs.load[day], inputs.pv[day], inputs.load_forecast[day],
            inputs.hourly_pv, inputs.price, state, updates=updates,
            settlement=settlement, terminal=6000. if day == end - 1 else None,
            reserve_by_issue=reserves)
        date = (pd.Timestamp('2025-01-01') + pd.Timedelta(days=day)).strftime('%Y-%m-%d')
        plan_cost = float(inputs.price @ result['g'])
        up_cost = float(1.5 * inputs.price @ np.maximum(result['q'] - result['g'], 0))
        down_fee = float(.5 * inputs.price @ np.maximum(result['g'] - result['q'], 0))
        refund = 2 * down_fee if settlement == 'N1' else 0.
        emergency_cost = float(5 * inputs.price @ result['e'])
        days.append(dict(
            day=day, date=date, settlement=settlement, updates='|'.join(map(str, updates)),
            risk_pair='none' if risk_pair is None else risk_pair.name,
            cost=result['cost'], plan_cost=plan_cost, up_cost=up_cost,
            down_fee=down_fee, refund=refund, emergency_cost=emergency_cost,
            emergency_kwh=float(result['e'].sum()), initial=state,
            terminal=float(result['E'][-1]), max_lp_gap=result['max_lp_gap'],
            max_mip_gap=result['max_mip_gap'], milp_solves=result['milp_solves'],
            seconds=time.perf_counter() - tick))
        frame = pd.DataFrame({key: result[key] for key in
                              ('g', 'q', 'c', 'd', 'e', 'unused', 'spill', 'actual_grid')})
        frame.insert(0, 'interval_id', np.arange(144))
        frame.insert(0, 'date', date)
        frame['E_start'], frame['E_end'] = result['E'][:-1], result['E'][1:]
        frame['load'], frame['pv'], frame['price'] = (
            inputs.load[day], inputs.pv[day], inputs.price)
        frame['delta'] = frame.q - frame.g
        chunks.append(frame)
        versions.extend(dict(date=date, **record) for record in result['versions'])
        state = float(result['E'][-1])
    daily = pd.DataFrame(days)
    dispatch = pd.concat(chunks, ignore_index=True)
    version_columns = ('date', 'issue_hour', 'interval_id', 'original_g',
                       'previous_q', 'q', 'delta', 'change')
    version_frame = pd.DataFrame(versions, columns=version_columns)
    if len(dispatch) != (end - start) * 144:
        raise AssertionError('Incomplete dispatch output')
    if not np.allclose(dispatch.E_end.to_numpy()[:-1],
                       dispatch.E_start.to_numpy()[1:], atol=1e-6, rtol=0):
        raise AssertionError('State discontinuity')
    if abs(state - 6000) > 1e-6:
        raise AssertionError('Final state is not closed')
    return daily, dispatch, version_frame


def save_policy(folder, label, inputs, updates, risk_pair,
                settlement='N0', start=31, end=365):
    folder.mkdir(parents=True, exist_ok=True)
    daily, dispatch, versions = simulate_policy(
        inputs, updates, risk_pair, settlement, start, end)
    daily.to_csv(folder / f'{label}_daily.csv', index=False)
    dispatch.to_csv(folder / f'{label}_dispatch.csv', index=False)
    versions.to_csv(folder / f'{label}_versions.csv', index=False)
    summary = dict(label=label, settlement=settlement, updates=list(updates),
                   risk_pair=None if risk_pair is None else asdict(risk_pair),
                   **_summary(daily, dispatch))
    (folder / f'{label}.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary, daily, dispatch, versions


def select_january_protocol(inputs):
    scores, definitions = {}, {}
    for risk_pair in RISK_PAIRS:
        for release_name, updates in RELEASE_SETS.items():
            label = f'{risk_pair.name}__{release_name}'
            daily, dispatch, _ = simulate_policy(
                inputs, updates, risk_pair, 'N0', start=14, end=31)
            scores[label] = _summary(daily, dispatch)
            definitions[label] = dict(risk_pair=asdict(risk_pair),
                                      release_name=release_name, updates=list(updates))
    winner = choose_protocol(scores, last_history_day=30, decision_day=31)
    return dict(
        calibration_start_day=14, calibration_end_exclusive=31,
        last_history_day=30, decision_day=31, settlement='N0',
        risk_pairs=[asdict(pair) for pair in RISK_PAIRS],
        release_sets={name: list(hours) for name, hours in RELEASE_SETS.items()},
        scores=scores, definitions=definitions, winner=winner,
        selected=definitions[winner],
        interpretation=('Frozen by January 15-31 shadow backtest; February-December '
                        'actuals are not used for selection'))


def _copy_canonical(selected_files):
    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    names = ('daily_summary.csv', 'dispatch_10min.csv', 'contract_versions.csv')
    for source, name in zip(selected_files, names):
        shutil.copy2(source, FINAL_OUT / name)


def run_formal():
    inputs = load_inputs()
    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    candidate_folder = MODEL_OUT / 'formal_candidates'
    protocol = select_january_protocol(inputs)
    (FINAL_OUT / 'selection_protocol.json').write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    selected = protocol['selected']
    selected_pair = RiskPair(**selected['risk_pair'])
    summaries = []
    selected_paths = None
    for release_name, updates in RELEASE_SETS.items():
        label = f'calibrated_{release_name}'
        summary, _, _, _ = save_policy(
            candidate_folder, label, inputs, updates, selected_pair)
        summaries.append(summary)
        if release_name == selected['release_name']:
            selected_paths = tuple(candidate_folder / f'{label}_{suffix}.csv'
                                   for suffix in ('daily', 'dispatch', 'versions'))
    for label, updates in (('D0_uncalibrated', ()),
                           ('Dall_uncalibrated', (6, 12, 18))):
        summary, _, _, _ = save_policy(
            candidate_folder, label, inputs, updates, None)
        summaries.append(summary)
    sensitivity_label = f"N1_{selected['release_name']}"
    summary, _, _, _ = save_policy(
        candidate_folder, sensitivity_label, inputs, tuple(selected['updates']),
        selected_pair, settlement='N1')
    summaries.append(summary)
    if selected_paths is None:
        raise AssertionError('Selected release set was not evaluated')
    _copy_canonical(selected_paths)
    pd.DataFrame(summaries).to_csv(FINAL_OUT / 'comparison.csv', index=False)
    print(json.dumps(dict(winner=protocol['winner'], selected=selected,
                          candidates=len(summaries)), ensure_ascii=False), flush=True)


def run_smoke(start, end):
    inputs = load_inputs()
    folder = MODEL_OUT / f'smoke_{start}_{end}'
    pair = RiskPair(.80, .70)
    summaries = []
    for release_name, updates in RELEASE_SETS.items():
        summary, _, _, _ = save_policy(
            folder, release_name, inputs, updates, pair, start=start, end=end)
        summaries.append(summary)
    (folder / 'summary.json').write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'candidates': len(summaries), 'days': end-start},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('smoke', 'formal'), required=True)
    parser.add_argument('--start', type=int, default=31)
    parser.add_argument('--end', type=int, default=365)
    arguments = parser.parse_args()
    if arguments.stage == 'smoke':
        run_smoke(arguments.start, arguments.end)
    else:
        run_formal()
