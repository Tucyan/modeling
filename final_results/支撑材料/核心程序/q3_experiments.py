"""Causal calibration and release-set experiments for Question 3."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd

from q3_forecast import residual_samples, risk_reserve
from q3_solver import ROOT, DATA, settle, simulate_day
from q3_q2_bridge import (load_q2_frozen, map_q2_dispatch, q2_day_ahead,
                          q2_frozen_cost, replay_q2_frozen)


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
    expected_dates = np.repeat(pd.date_range('2025-01-01', '2025-12-31')
                               .strftime('%Y-%m-%d').to_numpy(), 144)
    expected_intervals = np.tile(np.arange(144), 365)
    if (len(actual) != 365 * 144
            or actual.duplicated(['date', 'interval_id']).any()
            or not np.array_equal(actual.date.astype(str).to_numpy(), expected_dates)
            or not np.array_equal(actual.interval_id.to_numpy(), expected_intervals)):
        raise ValueError('Invalid actual-data keys')
    load = actual.load_kwh_10min.to_numpy(dtype=float).reshape(365, 144)
    pv = actual.pv_actual_kwh_10min.to_numpy(dtype=float).reshape(365, 144)
    price = pd.read_csv(DATA / 'typical_day_10min.csv').price_yuan_per_kwh.to_numpy(dtype=float)
    with np.load(Q2_FORECASTS) as forecasts:
        load_forecast = forecasts['P1'][:, 0, :].copy()
    release = pd.read_csv(DATA / 'pv_forecast_hourly.csv')
    release['day'] = (pd.to_datetime(release.issue_date) - pd.Timestamp('2025-01-01')).dt.days
    release['row'] = release.issue_time.str.split(':').str[0].astype(int) // 6
    expected_release_keys = pd.MultiIndex.from_product(
        [range(365), range(4), range(1, 25)],
        names=['day', 'row', 'horizon_hour'])
    actual_release_keys = pd.MultiIndex.from_frame(
        release[['day', 'row', 'horizon_hour']])
    if (len(release) != 35040
            or release.duplicated(['day', 'row', 'horizon_hour']).any()
            or not actual_release_keys.equals(expected_release_keys)):
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


def simulate_policy(inputs, updates, risk_pair, settlement='N0', start=31, end=365,
                    initial=6000., planning_target=6000.):
    if not 7 < start < end <= 365:
        raise ValueError('Invalid evaluation range')
    if tuple(updates) not in RELEASE_SETS.values():
        raise ValueError('Unknown release subset')
    state = float(initial)
    days, chunks, versions = [], [], []
    for day in range(start, end):
        tick = time.perf_counter()
        reserves = reserves_for_day(inputs, day, updates, risk_pair)
        result = simulate_day(
            day, inputs.load[day], inputs.pv[day], inputs.load_forecast[day],
            inputs.hourly_pv, inputs.price, state, updates=updates,
            settlement=settlement, terminal=6000. if day == end - 1 else None,
            reserve_by_issue=reserves, planning_target=planning_target)
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
    if abs(state - planning_target) > 1e-6:
        raise AssertionError('Final state is not closed')
    return daily, dispatch, version_frame


def save_policy(folder, label, inputs, updates, risk_pair,
                settlement='N0', start=31, end=365, initial=6000.,
                planning_target=6000.):
    folder.mkdir(parents=True, exist_ok=True)
    daily, dispatch, versions = simulate_policy(
        inputs, updates, risk_pair, settlement, start, end, initial,
        planning_target)
    daily.to_csv(folder / f'{label}_daily.csv', index=False)
    dispatch.to_csv(folder / f'{label}_dispatch.csv', index=False)
    versions.to_csv(folder / f'{label}_versions.csv', index=False)
    summary = dict(label=label, settlement=settlement, updates=list(updates),
                   risk_pair=None if risk_pair is None else asdict(risk_pair),
                   **_summary(daily, dispatch))
    (folder / f'{label}.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary, daily, dispatch, versions


def select_january_protocol(inputs, settlement='N0'):
    if settlement not in ('N0', 'N1'):
        raise ValueError('Supported settlements are N0 and N1')
    scores, definitions = {}, {}
    for risk_pair in RISK_PAIRS:
        for release_name, updates in RELEASE_SETS.items():
            label = f'{risk_pair.name}__{release_name}'
            daily, dispatch, _ = simulate_policy(
                inputs, updates, risk_pair, settlement, start=14, end=31)
            scores[label] = _summary(daily, dispatch)
            definitions[label] = dict(risk_pair=asdict(risk_pair),
                                      release_name=release_name, updates=list(updates))
    winner = choose_protocol(scores, last_history_day=30, decision_day=31)
    return dict(
        calibration_start_day=14, calibration_end_exclusive=31,
        last_history_day=30, decision_day=31, settlement=settlement,
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


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stage_signature():
    """Fingerprint every protected input and implementation used by the stages."""
    paths = (
        DATA / 'annual_actual_10min.csv', DATA / 'pv_forecast_hourly.csv',
        DATA / 'typical_day_10min.csv', Q2_FORECASTS,
        ROOT / 'outputs/q2_review_v2/rolling_selection_T0.csv',
        ROOT / 'outputs/q2_review_v2/experiments/Rolling_T0_daily.csv',
        ROOT / 'outputs/q2_review_v2/experiments/Rolling_T0_dispatch.csv',
        *(ROOT / 'src' / name for name in (
            'dispatch_control.py', 'q2_solver.py', 'q2_review_experiments.py',
            'q3_solver.py', 'q3_forecast.py', 'q3_q2_bridge.py',
            'q3_experiments.py')),
    )
    files = {str(path.relative_to(ROOT)).replace('\\', '/'): _sha256(path)
             for path in paths}
    aggregate = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'aggregate_sha256': aggregate, 'files': files}


def _complete_stage(output, stage, artifacts):
    output = Path(output)
    artifact_hashes = {
        str(Path(path).relative_to(output)).replace('\\', '/'): _sha256(path)
        for path in sorted({Path(path) for path in artifacts})}
    manifest = {'stage': stage, 'status': 'complete',
                'source_signature': _stage_signature(),
                'artifacts': artifact_hashes}
    path = output / f'stage_{stage}.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def _require_stage(output, stage):
    output = Path(output)
    path = output / f'stage_{stage}.json'
    if not path.exists():
        raise RuntimeError(f'Required stage is missing: {stage}')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete' or manifest.get('source_signature') != _stage_signature():
        raise RuntimeError(f'Required stage is stale or incomplete: {stage}')
    for relative, expected in manifest.get('artifacts', {}).items():
        artifact = output / relative
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise RuntimeError(f'Required stage artifact changed: {relative}')
    return manifest


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


def _write_q2_frozen(folder):
    source_daily, source_dispatch = load_q2_frozen(ROOT)
    daily, dispatch = replay_q2_frozen(ROOT)
    source_summary = json.loads((ROOT / 'outputs/q2_review_v2/experiments/Rolling_T0.json')
                                .read_text(encoding='utf-8'))
    errors = {
        key: float(np.max(abs(dispatch[key].to_numpy()-source_dispatch[key].to_numpy())))
        for key in ('c','d','e','unused','spill','E_start','E_end')}
    cost_n0 = q2_frozen_cost(dispatch, 'N0')
    cost_n1 = q2_frozen_cost(dispatch, 'N1')
    cost_error = abs(cost_n0-float(source_summary['cost']))
    if max(errors.values()) > 1e-6 or cost_error > 1e-6:
        raise AssertionError('Q2 frozen replay is not equivalent to the formal Q2 ledger')
    summary = dict(label='Q2_frozen', settlement='N0', updates=[],
                   days=int(dispatch.date.nunique()), rows=int(len(dispatch)),
                   cost_n0=cost_n0, cost_n1=cost_n1,
                   plan_cost=float(source_summary['plan_cost']),
                   emergency_cost=float(source_summary['emergency_cost']),
                   emergency_kwh=float(source_summary['emergency_kwh']),
                   emergency_days=int(source_summary['emergency_days']),
                   emergency_events=int(source_summary['emergency_events']),
                   emergency_slots=int(source_summary['emergency_slots']),
                   unused_kwh=float(source_summary['unused_kwh']),
                   spill_kwh=float(source_summary['spill_kwh']),
                   max_controller_replay_error_kwh=max(errors.values()),
                   controller_replay_error_by_field=errors,
                   source_cost_error_yuan=cost_error,
                   source='outputs/q2_review_v2/experiments/Rolling_T0_dispatch.csv')
    folder.mkdir(parents=True, exist_ok=True)
    daily.to_csv(folder / 'Q2_frozen_daily.csv', index=False)
    dispatch.to_csv(folder / 'Q2_frozen_dispatch.csv', index=False)
    pd.DataFrame(columns=('date', 'issue_hour', 'interval_id', 'original_g',
                          'previous_q', 'q', 'delta', 'change')).to_csv(
                              folder / 'Q2_frozen_versions.csv', index=False)
    (folder / 'Q2_frozen.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def _q2_comparison_row(summary, settlement):
    return dict(label='Q2_frozen', settlement=settlement, updates=[], risk_pair=None,
                days=summary['days'], cost=summary[f'cost_{settlement.lower()}'],
                plan_cost=summary['plan_cost'], up_cost=0., down_fee=0., refund=0.,
                emergency_cost=summary['emergency_cost'],
                emergency_kwh=summary['emergency_kwh'],
                emergency_days=summary['emergency_days'],
                emergency_events=summary['emergency_events'],
                emergency_slots=summary['emergency_slots'],
                unused_kwh=summary['unused_kwh'], spill_kwh=summary['spill_kwh'],
                final_soc=6000.)


def _q2_choices():
    choices = {day: 'M1' for day in range(31, 365)}
    records = pd.read_csv(ROOT / 'outputs/q2_review_v2/rolling_selection_T0.csv')
    for record in records.itertuples(index=False):
        start = int(record.decision_day)
        stop = 365 if int(record.month) == 12 else int(
            (pd.Timestamp(2025, int(record.month) + 1, 1)
             - pd.Timestamp(2025, 1, 1)).days)
        choices.update({day: record.winner for day in range(start, stop)})
    return choices


def simulate_q2_adapter_policy(inputs, updates, risk_pair, settlement='N0',
                               start=31, end=365):
    """Use Q2's causal day-ahead rule and Q3 updates on the strategy's own SOC."""
    with np.load(Q2_FORECASTS) as stored:
        forecasts = {key: stored[key].copy() for key in ('P0', 'P1')}
    choices = _q2_choices()
    state = 6000.; days, chunks, versions = [], [], []
    for day in range(start, end):
        tick = time.perf_counter()
        contract = q2_day_ahead(inputs.load, inputs.pv, inputs.price, forecasts,
                                day, choices[day], initial=state)
        reserves = reserves_for_day(inputs, day, updates, risk_pair)
        result = simulate_day(
            day, inputs.load[day], inputs.pv[day], inputs.load_forecast[day],
            inputs.hourly_pv, inputs.price, state, updates=updates,
            settlement=settlement, terminal=6000. if day == end - 1 else None,
            reserve_by_issue=reserves, original_contract=contract,
            planning_target=6000.)
        date = (pd.Timestamp('2025-01-01') + pd.Timedelta(days=day)).strftime('%Y-%m-%d')
        plan_cost = float(inputs.price @ result['g'])
        up_cost = float(1.5 * inputs.price @ np.maximum(result['q'] - result['g'], 0))
        down_fee = float(.5 * inputs.price @ np.maximum(result['g'] - result['q'], 0))
        refund = 2 * down_fee if settlement == 'N1' else 0.
        emergency_cost = float(5 * inputs.price @ result['e'])
        days.append(dict(
            day=day, date=date, settlement=settlement, updates='|'.join(map(str, updates)),
            risk_pair='none' if risk_pair is None else risk_pair.name,
            day_ahead_model=choices[day],
            cost=result['cost'], plan_cost=plan_cost, up_cost=up_cost,
            down_fee=down_fee, refund=refund, emergency_cost=emergency_cost,
            emergency_kwh=float(result['e'].sum()), initial=state,
            terminal=float(result['E'][-1]), max_lp_gap=result['max_lp_gap'],
            max_mip_gap=result['max_mip_gap'], milp_solves=result['milp_solves'],
            seconds=time.perf_counter()-tick))
        frame = pd.DataFrame({key: result[key] for key in
                              ('g','q','c','d','e','unused','spill','actual_grid')})
        frame.insert(0, 'interval_id', np.arange(144)); frame.insert(0, 'date', date)
        frame['E_start'], frame['E_end'] = result['E'][:-1], result['E'][1:]
        frame['load'], frame['pv'], frame['price'] = inputs.load[day], inputs.pv[day], inputs.price
        frame['delta'] = frame.q-frame.g; chunks.append(frame)
        versions.extend(dict(date=date, **record) for record in result['versions'])
        state = float(result['E'][-1])
    daily = pd.DataFrame(days); dispatch = pd.concat(chunks, ignore_index=True)
    version_frame = pd.DataFrame(versions, columns=('date','issue_hour','interval_id',
                                 'original_g','previous_q','q','delta','change'))
    if abs(state-6000.) > 1e-6:
        raise AssertionError('Q2 adapter final state is not closed')
    return daily, dispatch, version_frame


def _fixed_ledger_cost(dispatch, settlement):
    return float(settle(dispatch.g, dispatch.q, dispatch.price, settlement).sum()
                 + 5 * dispatch.price.to_numpy(dtype=float) @ dispatch.e.to_numpy(dtype=float))


def run_unified_ablations(output):
    inputs = load_inputs(); output = Path(output); _require_stage(output, 'unified-formal')
    q2_summary = _write_q2_frozen(output)
    _, q2_source = load_q2_frozen(ROOT)
    _, q2_rebuilt, _ = simulate_q2_adapter_policy(inputs, (), None, 'N0')
    equivalence = {key: float(np.max(abs(q2_rebuilt[key].to_numpy()
                                               - q2_source[key].to_numpy())))
                   for key in ('g','c','d','e','unused','spill','E_start','E_end')}
    if max(equivalence.values()) > 1e-6:
        raise AssertionError('Q2 day-ahead adapter does not reproduce the frozen Q2 ledger')
    (output / 'q2_adapter_no_updates_equivalence.json').write_text(
        json.dumps({'max_error_by_field': equivalence,
                    'max_error_kwh': max(equivalence.values())}, indent=2),
        encoding='utf-8')
    results = []; protocols = {}
    for settlement in ('N0', 'N1'):
        folder = output / settlement
        protocol = json.loads((folder / 'selection_protocol.json').read_text(encoding='utf-8'))
        protocols[settlement] = protocol
        selected = protocol['selected']; pair = RiskPair(**selected['risk_pair'])
        ablation_folder = folder / 'release_ablations'
        for release_name, updates in RELEASE_SETS.items():
            label = f'calibrated_{release_name}'
            if release_name == selected['release_name']:
                summary = json.loads((folder / f'{label}.json').read_text(encoding='utf-8'))
            else:
                summary, _, _, _ = save_policy(
                    ablation_folder, label, inputs, updates, pair,
                    settlement=settlement)
            results.append(summary)
        settlement_rows = [_q2_comparison_row(q2_summary, settlement)] + [
            r for r in results if r['settlement'] == settlement]
        pd.DataFrame(settlement_rows).to_csv(
            folder / 'comparison.csv', index=False)
        daily, dispatch, versions = simulate_q2_adapter_policy(
            inputs, tuple(selected['updates']), pair, settlement)
        label = 'Q2_day_ahead_with_updates'
        daily.to_csv(folder / f'{label}_daily.csv', index=False)
        dispatch.to_csv(folder / f'{label}_dispatch.csv', index=False)
        versions.to_csv(folder / f'{label}_versions.csv', index=False)
        adapter_summary = dict(label=label, settlement=settlement,
                               updates=selected['updates'], risk_pair=selected['risk_pair'],
                               **_summary(daily, dispatch))
        (folder / f'{label}.json').write_text(
            json.dumps(adapter_summary, ensure_ascii=False, indent=2), encoding='utf-8')
        selected_dispatch = pd.read_csv(folder / 'dispatch_10min.csv')
        fixed = {other: _fixed_ledger_cost(selected_dispatch, other)
                 for other in ('N0', 'N1')}
        (folder / 'fixed_ledger_repricing.json').write_text(
            json.dumps(fixed, ensure_ascii=False, indent=2), encoding='utf-8')
    cross = {'protocols_identical': protocols['N0']['selected'] == protocols['N1']['selected'],
             'N0_selected': protocols['N0']['selected'],
             'N1_selected': protocols['N1']['selected'], 'evaluations': []}
    if cross['protocols_identical']:
        cross['intentional_reuse'] = True
        for source, target in (('N0', 'N1'), ('N1', 'N0')):
            summary = json.loads((output/target/'summary.json').read_text(encoding='utf-8'))
            cross['evaluations'].append(dict(source_protocol=source,
                                             target_settlement=target,
                                             cost=summary['cost'],
                                             artifact=f'{target}/summary.json'))
    else:
        cross['intentional_reuse'] = False
        cross_folder = output / 'cross_protocols'
        for source, target in (('N0', 'N1'), ('N1', 'N0')):
            selected = protocols[source]['selected']
            summary, _, _, _ = save_policy(
                cross_folder, f'{source}_protocol_under_{target}', inputs,
                tuple(selected['updates']), RiskPair(**selected['risk_pair']),
                settlement=target)
            cross['evaluations'].append(dict(source_protocol=source,
                                             target_settlement=target,
                                             cost=summary['cost'],
                                             artifact=f'cross_protocols/{source}_protocol_under_{target}.json'))
    (output / 'cross_protocols.json').write_text(
        json.dumps(cross, ensure_ascii=False, indent=2), encoding='utf-8')
    unified_rows = [_q2_comparison_row(q2_summary, settlement)
                    for settlement in ('N0', 'N1')] + results
    (output / 'unified_comparison.csv').write_text(
        pd.DataFrame(unified_rows).to_csv(index=False), encoding='utf-8')
    artifacts = [
        *(output.glob('Q2_frozen*')),
        output/'q2_adapter_no_updates_equivalence.json',
        output/'unified_comparison.csv', output/'cross_protocols.json',
        output/'stage_unified-formal.json',
    ]
    for settlement in ('N0', 'N1'):
        folder = output/settlement
        artifacts.extend((folder/'comparison.csv',
                          folder/'Q2_day_ahead_with_updates_daily.csv',
                          folder/'Q2_day_ahead_with_updates_dispatch.csv',
                          folder/'Q2_day_ahead_with_updates_versions.csv',
                          folder/'Q2_day_ahead_with_updates.json',
                          folder/'fixed_ledger_repricing.json'))
        artifacts.extend((folder/'release_ablations').glob('*'))
    artifacts.extend((output/'cross_protocols').glob('*')
                     if (output/'cross_protocols').exists() else [])
    _complete_stage(output, 'unified-ablations', artifacts)
    print(json.dumps({'stage':'unified-ablations','policies':len(results),
                      'q2_adapters':2}, ensure_ascii=False), flush=True)


def run_unified_smoke(output):
    """Run February, year-end and month-boundary gates for both settlements."""
    inputs = load_inputs()
    output = Path(output)
    q2_summary = _write_q2_frozen(output)
    _, q2_dispatch = replay_q2_frozen(ROOT)
    summaries = []
    for window_name, start, end in (('feb01_07',31,38),
                                    ('dec25_31',358,365),
                                    ('feb28_mar02',58,61)):
        for settlement in ('N0', 'N1'):
            folder = output / settlement / 'smoke' / window_name
            start_date = (pd.Timestamp('2025-01-01') + pd.Timedelta(days=start)).strftime('%Y-%m-%d')
            end_date = (pd.Timestamp('2025-01-01') + pd.Timedelta(days=end-1)).strftime('%Y-%m-%d')
            q2_window = q2_dispatch[(q2_dispatch.date >= start_date)
                                    & (q2_dispatch.date <= end_date)]
            if len(q2_window) != (end-start)*144:
                raise AssertionError('Q2 frozen smoke window is incomplete')
            summaries.append(dict(label='Q2_frozen', settlement=settlement,
                                  window=window_name, start=start, end=end,
                                  rows=len(q2_window), physical_gate=True,
                                  cost=_fixed_ledger_cost(q2_window, settlement),
                                  full_replay_error_kwh=q2_summary['max_controller_replay_error_kwh']))
            adapter_daily, adapter_dispatch, adapter_versions = simulate_q2_adapter_policy(
                inputs, (6, 12, 18), RiskPair(.80, .70), settlement, start, end)
            adapter_daily.to_csv(folder/'Q2_adapter_daily.csv', index=False)
            adapter_dispatch.to_csv(folder/'Q2_adapter_dispatch.csv', index=False)
            adapter_versions.to_csv(folder/'Q2_adapter_versions.csv', index=False)
            adapter_summary = dict(label='Q2_adapter', settlement=settlement,
                                   window=window_name, start=start, end=end,
                                   **_summary(adapter_daily, adapter_dispatch))
            (folder/'Q2_adapter.json').write_text(
                json.dumps(adapter_summary, ensure_ascii=False, indent=2), encoding='utf-8')
            summaries.append(adapter_summary)
            for release_name, updates in RELEASE_SETS.items():
                summary, _, _, _ = save_policy(
                    folder, f'smoke_{release_name}', inputs, updates,
                    RiskPair(.80, .70), settlement=settlement, start=start, end=end)
                summary.update(window=window_name, start=start, end=end)
                summaries.append(summary)
    (output / 'unified_smoke_summary.json').write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    smoke_artifacts = [path for path in output.rglob('*') if path.is_file()
                       and ('/smoke/' in str(path).replace('\\', '/')
                            or path.name.startswith('Q2_frozen')
                            or path.name == 'unified_smoke_summary.json')]
    _complete_stage(output, 'unified-smoke', smoke_artifacts)
    print(json.dumps({'stage': 'unified-smoke', 'windows': 3,
                      'policies': len(summaries)}, ensure_ascii=False), flush=True)


def run_unified_formal(output):
    """Select N0/N1 independently and write isolated formal ledgers."""
    inputs = load_inputs()
    output = Path(output)
    _require_stage(output, 'unified-smoke')
    _write_q2_frozen(output)
    all_results = []
    for settlement in ('N0', 'N1'):
        protocol = select_january_protocol(inputs, settlement=settlement)
        protocol['source_signature'] = _stage_signature()
        folder = output / settlement
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'selection_protocol.json').write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
        selected = protocol['selected']
        pair = RiskPair(**selected['risk_pair'])
        label = f"calibrated_{selected['release_name']}"
        summary, daily, dispatch, versions = save_policy(
            folder, label, inputs, tuple(selected['updates']), pair,
            settlement=settlement)
        daily.to_csv(folder / 'daily_summary.csv', index=False)
        dispatch.to_csv(folder / 'dispatch_10min.csv', index=False)
        versions.to_csv(folder / 'contract_versions.csv', index=False)
        (folder / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        all_results.append(dict(settlement=settlement, protocol=protocol,
                                summary=summary))
    (output / 'unified_formal_summary.json').write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8')
    formal_names = {'selection_protocol.json', 'summary.json', 'daily_summary.csv',
                    'dispatch_10min.csv', 'contract_versions.csv',
                    'unified_formal_summary.json'}
    artifacts = [path for path in output.rglob('*') if path.is_file()
                 and path.name in formal_names]
    _complete_stage(output, 'unified-formal', artifacts)
    print(json.dumps({'stage': 'unified-formal', 'settlements': ['N0', 'N1']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('smoke', 'formal', 'unified-smoke',
                                             'unified-formal', 'unified-ablations'), required=True)
    parser.add_argument('--start', type=int, default=31)
    parser.add_argument('--end', type=int, default=365)
    parser.add_argument('--output', default=str(MODEL_OUT / 'q3_unified_v1'))
    arguments = parser.parse_args()
    if arguments.stage == 'unified-ablations':
        run_unified_ablations(arguments.output)
    elif arguments.stage == 'unified-smoke':
        run_unified_smoke(arguments.output)
    elif arguments.stage == 'unified-formal':
        run_unified_formal(arguments.output)
    elif arguments.stage == 'smoke':
        run_smoke(arguments.start, arguments.end)
    else:
        run_formal()
