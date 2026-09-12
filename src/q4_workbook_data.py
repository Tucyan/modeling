"""Create result4 workbook payloads from a Q4 dispatch ledger."""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-7


def _time(boundary):
    if not 0 <= boundary <= 144:
        raise ValueError('interval boundary out of range')
    minutes = 10 * boundary
    return f'{minutes // 60:02d}:{minutes % 60:02d}'


def _events(day):
    values = day[['interval_id', 'e']].copy()
    values = values[values.e > EPS].sort_values('interval_id')
    if values.empty:
        return []
    result = []
    start = previous = int(values.iloc[0].interval_id)
    total = float(values.iloc[0].e)
    for row in values.iloc[1:].itertuples(index=False):
        interval = int(row.interval_id)
        if interval == previous + 1:
            total += float(row.e)
        else:
            result.append((start, previous + 1, total))
            start, total = interval, float(row.e)
        previous = interval
    result.append((start, previous + 1, total))
    return result


def _matrix(dispatch, value, settlement):
    pivot = dispatch.pivot(index='date', columns='interval_id', values=value)
    dates = pd.date_range('2025-02-01', '2025-12-31').strftime('%Y-%m-%d').tolist()
    if pivot.index.astype(str).tolist() != dates or pivot.columns.tolist() != list(range(144)):
        raise AssertionError(f'incomplete {value} matrix')
    rows = []
    for date in dates:
        day = dispatch[dispatch.date.astype(str) == date].sort_values('interval_id')
        values = pivot.loc[date].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f'non-finite {value}')
        if value == 'g':
            cost = float((day.price * day.g).sum())
        else:
            down = 0.5 if settlement == 'N0' else -0.5
            cost = float((day.price * (day.g + 1.5*np.maximum(day.q-day.g, 0)
                                      + down*np.maximum(day.g-day.q, 0))).sum())
        rows.append(values.tolist() + [float(values.sum()), cost])
    return rows


def build_payload(dispatch_path, settlement='N0'):
    if settlement not in ('N0', 'N1'):
        raise ValueError('settlement must be N0 or N1')
    dispatch = pd.read_csv(dispatch_path)
    required = {'date', 'interval_id', 'g', 'q', 'c', 'd', 'e', 'E_start', 'E_end',
                'price'}
    if required - set(dispatch.columns):
        raise AssertionError(f'missing columns: {sorted(required-set(dispatch.columns))}')
    expected = pd.date_range('2025-02-01', '2025-12-31').strftime('%Y-%m-%d').tolist()
    if len(dispatch) != 334 * 144 or dispatch.duplicated(['date', 'interval_id']).any():
        raise AssertionError('dispatch must contain 334 complete days')
    battery = []
    emergency = []
    for date in expected:
        day = dispatch[dispatch.date.astype(str) == date].sort_values('interval_id')
        if len(day) != 144:
            raise AssertionError(f'missing date {date}')
        for block in range(6):
            part = day.iloc[block*24:(block+1)*24]
            battery.append([
                date if block == 0 else None,
                f'{block*4}:00-{(block+1)*4}:00',
                float(part.c.sum()), float(part.d.sum()),
                '0:00' if block == 0 else '24:00' if block == 1 else None,
                float(day.iloc[0].E_start) if block == 0 else
                float(day.iloc[-1].E_end) if block == 1 else None,
            ])
        events = _events(day)
        if events:
            emergency.extend([
                [date if index == 0 else None,
                 f'{_time(start)}-{_time(end)}', total]
                for index, (start, end, total) in enumerate(events)
            ])
        else:
            emergency.append([date, '无', 0.])
    return {'settlement': settlement,
            'plan': _matrix(dispatch, 'g', settlement),
            'adjustment': _matrix(dispatch, 'q', settlement),
            'battery': battery, 'emergency': emergency}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--settlement', default='N0')
    args = parser.parse_args()
    payload = build_payload(args.input, args.settlement)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'workbook_payload.json').write_text(
        json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'plan_rows': len(payload['plan']),
                      'adjustment_rows': len(payload['adjustment']),
                      'battery_rows': len(payload['battery']),
                      'emergency_rows': len(payload['emergency'])}, ensure_ascii=False))
