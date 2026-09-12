"""Build the verified JSON payload consumed by the result3.xlsx writer."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/q3_final'
EPS = 1e-7


def contiguous_emergency_ranges(day_frame):
    """Merge adjacent positive emergency-purchase slots into half-open ranges."""
    frame = day_frame[['interval_id', 'e']].copy()
    frame = frame[frame.e > EPS].sort_values('interval_id')
    if frame.empty:
        return []
    events = []
    start = previous = int(frame.iloc[0].interval_id)
    total = float(frame.iloc[0].e)
    for row in frame.iloc[1:].itertuples(index=False):
        interval = int(row.interval_id)
        if interval == previous + 1:
            total += float(row.e)
        else:
            events.append({'start': start, 'end': previous + 1, 'kwh': total})
            start, total = interval, float(row.e)
        previous = interval
    events.append({'start': start, 'end': previous + 1, 'kwh': total})
    return events


def interval_time(boundary):
    if not 0 <= boundary <= 144:
        raise ValueError('Interval boundary out of range')
    minutes = boundary * 10
    return f'{minutes // 60:02d}:{minutes % 60:02d}'


def _matrix(dispatch, value, cost_kind):
    matrix = dispatch.pivot(index='date', columns='interval_id', values=value)
    expected_dates = pd.date_range('2025-02-01', '2025-12-31').strftime('%Y-%m-%d')
    if list(matrix.index) != list(expected_dates) or list(matrix.columns) != list(range(144)):
        raise AssertionError(f'Incomplete {value} matrix')
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError(f'Non-finite {value} values')
    rows = []
    for index, date in enumerate(matrix.index):
        day = dispatch[dispatch.date == date]
        if cost_kind == 'plan':
            cost = float((day.price * day.g).sum())
        elif cost_kind == 'adjustment_n0':
            cost = float((day.price * (day.g
                         + 1.5*np.maximum(day.q-day.g, 0)
                         + .5*np.maximum(day.g-day.q, 0))).sum())
        else:
            raise ValueError('Unknown workbook cost kind')
        rows.append(values[index].tolist() + [float(values[index].sum()), cost])
    return rows


def build_payload(dispatch_path=None):
    path = OUT / 'dispatch_10min.csv' if dispatch_path is None else Path(dispatch_path)
    dispatch = pd.read_csv(path)
    required = {'date', 'interval_id', 'g', 'q', 'c', 'd', 'e', 'E_start', 'E_end'}
    if required - set(dispatch.columns):
        raise AssertionError(f'Missing dispatch columns: {sorted(required-set(dispatch.columns))}')
    if len(dispatch) != 334 * 144 or dispatch.duplicated(['date', 'interval_id']).any():
        raise AssertionError('Dispatch ledger must contain 334 complete days')
    expected_dates = pd.date_range('2025-02-01', '2025-12-31').strftime('%Y-%m-%d')
    battery = []
    emergency = []
    for date in expected_dates:
        day = dispatch[dispatch.date == date].sort_values('interval_id')
        if len(day) != 144:
            raise AssertionError(f'Missing output date {date}')
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
        events = contiguous_emergency_ranges(day)
        if events:
            emergency.extend([
                [date if index == 0 else None,
                 f"{interval_time(event['start'])}-{interval_time(event['end'])}",
                 event['kwh']]
                for index, event in enumerate(events)
            ])
        else:
            emergency.append([date, '无', 0.])
    return {'plan': _matrix(dispatch, 'g', 'plan'),
            'adjustment': _matrix(dispatch, 'q', 'adjustment_n0'),
            'battery': battery, 'emergency': emergency}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    (OUT / 'workbook_payload.json').write_text(
        json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'plan_rows': len(payload['plan']),
                      'battery_rows': len(payload['battery']),
                      'emergency_rows': len(payload['emergency'])}))


if __name__ == '__main__':
    main()
