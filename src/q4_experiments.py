"""Causal selection, replay and delivery artifacts for Question 4."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

from q4_forecast import PRICE_MODELS, forecast_price
from q4_solver import q4_reserve_q2, q4_reserves_q3, simulate_q4_day


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'C题_标准化数据与修正模板/C_final_delivery/data_csv'
Q2_FORECASTS = ROOT / 'outputs/01a08b78-ed6b-7e60-b40d-9f266855d1ef/q2/forecasts.npz'
OUT = ROOT / 'outputs/q4_final'
RELEASE_SETS = {
    'R0': (), 'R6': (6,), 'R12': (12,), 'R18': (18,),
    'R6_12': (6, 12), 'R6_18': (6, 18),
    'R12_18': (12, 18), 'Rall': (6, 12, 18),
}
Q42_ALPHAS = (0.5, 0.7, 0.8, 0.9)
Q43_RISK_PAIRS = ((0.5, 0.5), (0.7, 0.65), (0.8, 0.65),
                  (0.8, 0.75), (0.85, 0.75), (0.9, 0.8))
PRICE_CORRECTION_GAMMAS = (0, 1)
EPS = 1e-7


@dataclass(frozen=True)
class Q4Inputs:
    load: np.ndarray
    pv: np.ndarray
    price: np.ndarray
    load_forecast: np.ndarray
    pv_forecast_q2: np.ndarray
    hourly_pv: np.ndarray


def load_inputs():
    actual = pd.read_csv(DATA / 'annual_actual_10min.csv')
    expected_dates = np.repeat(pd.date_range('2025-01-01', '2025-12-31')
                               .strftime('%Y-%m-%d').to_numpy(), 144)
    expected_intervals = np.tile(np.arange(144), 365)
    if (len(actual) != 365 * 144
            or actual.duplicated(['date', 'interval_id']).any()
            or not np.array_equal(actual.date.astype(str).to_numpy(), expected_dates)
            or not np.array_equal(actual.interval_id.to_numpy(), expected_intervals)):
        raise ValueError('Invalid annual actual keys')
    load = actual.load_kwh_10min.to_numpy(dtype=float).reshape(365, 144)
    pv = actual.pv_actual_kwh_10min.to_numpy(dtype=float).reshape(365, 144)
    price = actual.price_yuan_per_kwh.to_numpy(dtype=float).reshape(365, 144)
    if not np.isfinite(np.r_[load.ravel(), pv.ravel(), price.ravel()]).all():
        raise ValueError('Non-finite annual actual inputs')
    if np.any(load < 0) or np.any(pv < 0) or np.any(price <= 0):
        raise ValueError('Annual actual inputs outside physical ranges')
    with np.load(Q2_FORECASTS) as cached:
        q2 = cached['P1'].copy()
    if q2.shape != (365, 2, 144):
        raise ValueError('Unexpected Q2 forecast cache shape')
    load_forecast = q2[:, 0, :]
    pv_forecast_q2 = q2[:, 1, :]
    if not np.isfinite(load_forecast[7:]).all() or not np.isfinite(pv_forecast_q2[7:]).all():
        raise ValueError('Q2 forecasts are not finite after warm-up')
    release = pd.read_csv(DATA / 'pv_forecast_hourly.csv')
    release['day'] = (pd.to_datetime(release.issue_date)
                      - pd.Timestamp('2025-01-01')).dt.days
    release['row'] = release.issue_time.str.split(':').str[0].astype(int) // 6
    expected = pd.MultiIndex.from_product(
        [range(365), range(4), range(1, 25)],
        names=['day', 'row', 'horizon_hour'])
    actual_keys = pd.MultiIndex.from_frame(release[['day', 'row', 'horizon_hour']])
    if (len(release) != 365 * 4 * 24
            or release.duplicated(['day', 'row', 'horizon_hour']).any()
            or not actual_keys.equals(expected)):
        raise ValueError('Invalid hourly forecast-release keys')
    hourly = np.full((365, 4, 24), np.nan)
    hourly[release.day, release.row, release.horizon_hour - 1] = release.pv_forecast_kw
    if not np.isfinite(hourly).all() or np.any(hourly < 0):
        raise ValueError('Invalid hourly PV forecast values')
    return Q4Inputs(load, pv, price, load_forecast, pv_forecast_q2, hourly)


def _date(day):
    return (pd.Timestamp('2025-01-01') + pd.Timedelta(days=int(day))).strftime('%Y-%m-%d')


def _price_metrics(inputs):
    rows = []
    for model in PRICE_MODELS:
        errors = []
        for day in range(14, 31):
            errors.extend(forecast_price(inputs.price, day, 0, model)
                          - inputs.price[day])
        errors = np.asarray(errors)
        rows.append(dict(model=model, days=17, slots=len(errors),
                         mae=float(np.mean(np.abs(errors))),
                         rmse=float(np.sqrt(np.mean(errors ** 2))),
                         max_abs=float(np.max(np.abs(errors)))))
    return pd.DataFrame(rows)


def _summary(daily, dispatch):
    positive = dispatch.e.to_numpy(dtype=float) > EPS
    starts = positive & (~np.r_[False, positive[:-1]]
                         | (dispatch.interval_id.to_numpy() == 0))
    return dict(days=int(len(daily)), cost=float(daily.cost.sum()),
                plan_cost=float(daily.plan_cost.sum()),
                up_cost=float(daily.up_cost.sum()),
                down_fee=float(daily.down_fee.sum()),
                refund=float(daily.refund.sum()),
                emergency_cost=float(daily.emergency_cost.sum()),
                emergency_kwh=float(daily.emergency_kwh.sum()),
                emergency_days=int((daily.emergency_kwh > EPS).sum()),
                emergency_events=int(starts.sum()),
                emergency_slots=int(positive.sum()),
                unused_kwh=float(dispatch.unused.sum()),
                spill_kwh=float(dispatch.spill.sum()),
                final_soc=float(daily.terminal.iloc[-1]),
                max_lp_gap=float(daily.max_lp_gap.max()),
                max_mip_gap=float(daily.max_mip_gap.max()),
                milp_solves=int(daily.milp_solves.sum()),
                max_seconds=float(daily.seconds.max()),
                mean_seconds=float(daily.seconds.mean()))


def simulate_policy(inputs, mode, price_model, updates=(), alpha0=0.8,
                    alpha_update=0.7, settlement='N0', start=31, end=365,
                    price_gamma=0):
    if mode not in ('q4-2', 'q4-3'):
        raise ValueError('unknown Q4 mode')
    if mode == 'q4-2' and updates:
        raise ValueError('Q4-2 cannot contain updates')
    if price_gamma not in PRICE_CORRECTION_GAMMAS:
        raise ValueError('price_gamma must be 0 or 1')
    state = 6000.0
    days, chunks, versions = [], [], []
    for day in range(start, end):
        tick = time.perf_counter()
        plan_price = forecast_price(inputs.price, day, 0, price_model)
        if mode == 'q4-2':
            reserve = {0: q4_reserve_q2(
                day, inputs.load, inputs.pv, inputs.load_forecast,
                inputs.pv_forecast_q2, alpha0)}
            issue_prices = None
        else:
            reserve = q4_reserves_q3(
                day, inputs.load, inputs.pv, inputs.load_forecast,
                inputs.hourly_pv, updates, alpha0, alpha_update)
            issue_prices = {
                hour: forecast_price(inputs.price, day, hour, price_model,
                                     observed_prefix=inputs.price[day, :hour * 6],
                                     correction_gamma=price_gamma)
                for hour in updates
            }
        result = simulate_q4_day(
            day=day, load=inputs.load[day], pv=inputs.pv[day],
            forecast_load=inputs.load_forecast[day],
            forecast_pv_q2=inputs.pv_forecast_q2[day],
            hourly_pv=inputs.hourly_pv, plan_price=plan_price,
            actual_price=inputs.price[day], initial=state, mode=mode,
            updates=updates, settlement=settlement,
            terminal=6000.0 if day == end - 1 else None,
            reserve_by_issue=reserve, price_forecast_by_issue=issue_prices)
        actual_price = inputs.price[day]
        q, g = result['q'], result['g']
        plan_cost = float(actual_price @ g)
        up = float(1.5 * actual_price @ np.maximum(q - g, 0))
        down = float(0.5 * actual_price @ np.maximum(g - q, 0))
        refund = 2 * down if settlement == 'N1' else 0.0
        emergency_cost = float(5 * actual_price @ result['e'])
        days.append(dict(
            day=day, date=_date(day), mode=mode, settlement=settlement,
            price_model=price_model, updates='|'.join(map(str, updates)),
            price_gamma=price_gamma, alpha0=alpha0,
            alpha_update=alpha_update, cost=result['cost'],
            plan_cost=plan_cost, up_cost=up, down_fee=down, refund=refund,
            emergency_cost=emergency_cost,
            emergency_kwh=float(result['e'].sum()),
            charge_kwh=float(result['c'].sum()),
            discharge_kwh=float(result['d'].sum()),
            unused_kwh=float(result['unused'].sum()),
            spill_kwh=float(result['spill'].sum()), initial=state,
            terminal=float(result['E'][-1]), max_lp_gap=result['max_lp_gap'],
            max_mip_gap=result['max_mip_gap'],
            milp_solves=result['milp_solves'], seconds=time.perf_counter() - tick))
        frame = pd.DataFrame({key: result[key] for key in
                              ('g', 'q', 'c', 'd', 'e', 'unused', 'spill', 'actual_grid')})
        frame.insert(0, 'interval_id', np.arange(144))
        frame.insert(0, 'date', _date(day))
        frame['E_start'], frame['E_end'] = result['E'][:-1], result['E'][1:]
        frame['load'], frame['pv'], frame['price'] = inputs.load[day], inputs.pv[day], actual_price
        frame['plan_price'] = plan_price
        frame['delta'] = frame.q - frame.g
        chunks.append(frame)
        versions.extend(dict(date=_date(day), **version) for version in result['versions'])
        state = float(result['E'][-1])
    daily = pd.DataFrame(days)
    dispatch = pd.concat(chunks, ignore_index=True)
    version_columns = ('date', 'issue_hour', 'interval_id', 'original_g',
                       'previous_q', 'q', 'delta', 'change')
    version_frame = pd.DataFrame(versions, columns=version_columns)
    if len(dispatch) != (end - start) * 144:
        raise AssertionError('incomplete Q4 dispatch')
    if not np.allclose(dispatch.E_end.to_numpy()[:-1],
                       dispatch.E_start.to_numpy()[1:], atol=1e-6, rtol=0):
        raise AssertionError('Q4 state discontinuity')
    if abs(state - 6000.0) > 1e-6:
        raise AssertionError('Q4 final state is not closed')
    if not np.isfinite(dispatch.select_dtypes(include=[np.number]).to_numpy()).all():
        raise AssertionError('Q4 dispatch contains non-finite values')
    return daily, dispatch, version_frame


def save_policy(folder, label, inputs, mode, price_model, updates=(),
                alpha0=0.8, alpha_update=0.7, settlement='N0',
                start=31, end=365, price_gamma=0):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    daily, dispatch, versions = simulate_policy(
        inputs, mode, price_model, updates, alpha0, alpha_update,
        settlement, start, end, price_gamma)
    daily.to_csv(folder / f'{label}_daily.csv', index=False)
    dispatch.to_csv(folder / f'{label}_dispatch.csv', index=False)
    versions.to_csv(folder / f'{label}_versions.csv', index=False)
    summary = dict(label=label, mode=mode, settlement=settlement,
                   price_model=price_model, updates=list(updates),
                   price_gamma=price_gamma,
                   alpha0=alpha0, alpha_update=alpha_update,
                   **_summary(daily, dispatch))
    (folder / f'{label}.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary, daily, dispatch, versions


def write_forecast_records(inputs, mode, price_model, updates, output_path,
                           start=31, end=365, price_gamma=0):
    """Persist every causal price forecast used by the selected replay."""
    issues = (0,) if mode == 'q4-2' else (0,) + tuple(updates)
    rows = []
    for day in range(start, end):
        for hour in issues:
            values = forecast_price(
                inputs.price, day, hour, price_model,
                observed_prefix=inputs.price[day, :hour * 6],
                correction_gamma=price_gamma)
            for offset, forecast in enumerate(values):
                interval = hour * 6 + offset
                rows.append(dict(
                    date=_date(day), day=day, issue_hour=hour,
                    interval_id=interval, forecast_price=float(forecast),
                    actual_price=float(inputs.price[day, interval]),
                    price_gamma=price_gamma,
                    correction_window=6, decay_slots=36,
                    history_start_day=max(7, day - 56),
                    history_end_exclusive=day,
                    observed_prefix_slots=hour * 6))
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def _choose(rows):
    return min(rows, key=lambda row: (float(row['cost']),
                                      float(row['emergency_kwh']),
                                      str(row['label'])))


def select_protocol(inputs, settlement):
    price_metrics = _price_metrics(inputs)
    price_model = str(price_metrics.sort_values(['mae', 'model']).iloc[0].model)
    q42_scores = []
    q43_scores = []
    for alpha in Q42_ALPHAS:
        daily, dispatch, _ = simulate_policy(
            inputs, 'q4-2', price_model, (), alpha, alpha,
            settlement, 14, 31)
        q42_scores.append(dict(label=f'a{alpha:.2f}', mode='q4-2',
                               price_model=price_model, updates=[],
                               price_gamma=0,
                               alpha0=alpha, alpha_update=alpha,
                               **_summary(daily, dispatch)))
    for alpha0, alpha_update in Q43_RISK_PAIRS:
        for release_name, updates in RELEASE_SETS.items():
            for price_gamma in ((0,) if not updates else PRICE_CORRECTION_GAMMAS):
                daily, dispatch, _ = simulate_policy(
                    inputs, 'q4-3', price_model, updates, alpha0, alpha_update,
                    settlement, 14, 31, price_gamma)
                q43_scores.append(dict(
                    label=(f'a0_{alpha0:.2f}_au_{alpha_update:.2f}'
                           f'__{release_name}__gamma_{price_gamma}'),
                    mode='q4-3', price_model=price_model,
                    updates=list(updates), price_gamma=price_gamma,
                    alpha0=alpha0, alpha_update=alpha_update,
                    release_name=release_name,
                    **_summary(daily, dispatch)))
    q42 = _choose(q42_scores)
    q43 = _choose(q43_scores)
    return dict(
        calibration_start_day=14, calibration_end_exclusive=31,
        last_history_day=30, decision_day=31, settlement=settlement,
        price_models=list(PRICE_MODELS), price_metrics=price_metrics.to_dict('records'),
        selected_price_model=price_model, q42_alphas=list(Q42_ALPHAS),
        price_correction_gammas=list(PRICE_CORRECTION_GAMMAS),
        q43_risk_pairs=[dict(alpha0=a, alpha_update=b) for a, b in Q43_RISK_PAIRS],
        release_sets={name: list(value) for name, value in RELEASE_SETS.items()},
        q42_scores=q42_scores, q43_scores=q43_scores,
        selected_q42=q42, selected_q43=q43,
        interpretation='Frozen by January 15-31 causal shadow backtest; output days were not used for selection')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_smoke(output):
    inputs = load_inputs()
    output = Path(output)
    rows = []
    # Smoke gates exercise the interface with a fixed, documented candidate;
    # January selection is intentionally deferred to the formal stage.
    metrics = _price_metrics(inputs)
    price_model = str(metrics.sort_values(['mae', 'model']).iloc[0].model)
    for window, start, end in (('feb01_07', 31, 38),
                               ('feb28_mar02', 58, 61),
                               ('dec25_31', 358, 365)):
        folder = output / 'smoke' / window
        for settlement in ('N0', 'N1'):
            candidates = (('q4-2', (), .8, .8),
                          ('q4-3', (6, 12, 18), .8, .7, 1))
            for candidate in candidates:
                if len(candidate) == 4:
                    mode, updates, alpha0, alpha_update = candidate
                    price_gamma = 0
                else:
                    mode, updates, alpha0, alpha_update, price_gamma = candidate
                summary, _, _, _ = save_policy(
                    folder / settlement, mode, inputs, mode,
                    price_model, updates, alpha0, alpha_update, settlement,
                    start, end, price_gamma)
                summary.update(window=window, start=start, end=end)
                rows.append(summary)
    (output / 'smoke_summary.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'stage': 'smoke', 'rows': len(rows)}, ensure_ascii=False))


def run_formal(output):
    inputs = load_inputs()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for settlement in ('N0', 'N1'):
        folder = output / settlement
        protocol = select_protocol(inputs, settlement)
        (folder / 'selection_protocol.json').parent.mkdir(parents=True, exist_ok=True)
        (folder / 'selection_protocol.json').write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
        selected2, selected3 = protocol['selected_q42'], protocol['selected_q43']
        summaries = []
        for mode, selected in (('q4-2', selected2), ('q4-3', selected3)):
            summary, daily, dispatch, versions = save_policy(
                folder, mode, inputs, mode, protocol['selected_price_model'],
                tuple(selected['updates']), selected['alpha0'], selected['alpha_update'],
                settlement, 31, 365, selected.get('price_gamma', 0))
            daily.to_csv(folder / f'{mode}_daily_summary.csv', index=False)
            dispatch.to_csv(folder / f'{mode}_dispatch_10min.csv', index=False)
            versions.to_csv(folder / f'{mode}_contract_versions.csv', index=False)
            write_forecast_records(
                inputs, mode, protocol['selected_price_model'],
                tuple(selected['updates']), folder / f'{mode}_forecast_records.csv',
                price_gamma=selected.get('price_gamma', 0))
            summaries.append(summary)
        (folder / 'summary.json').write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
        all_results.append(dict(settlement=settlement, protocol=protocol,
                                summaries=summaries))
    (output / 'formal_summary.json').write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'stage': 'formal', 'settlements': ['N0', 'N1']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('smoke', 'formal'), required=True)
    parser.add_argument('--output', default=str(OUT))
    args = parser.parse_args()
    if args.stage == 'smoke':
        run_smoke(args.output)
    else:
        run_formal(args.output)
