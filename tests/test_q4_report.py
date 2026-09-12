import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from q4_report import TARGET_DATES, date_summary, render_target_tables


ROOT = Path(__file__).resolve().parents[1]
N0 = ROOT / 'outputs/q4_p0_final/N0'


def test_target_date_summary_has_required_q4_fields():
    dispatch = pd.read_csv(N0 / 'q4-3_dispatch_10min.csv')
    summary = date_summary(dispatch, '2025-03-20', mode='q4-3', settlement='N0')

    assert summary['date'] == '2025-03-20'
    assert len(summary['slot_values']) == 6
    assert all('actual' in row for row in summary['slot_values'])
    assert len(summary['blocks']) == 6
    assert summary['initial_soc'] == dispatch.loc[dispatch.date.eq('2025-03-20'), 'E_start'].iloc[0]
    assert summary['final_soc'] == dispatch.loc[dispatch.date.eq('2025-03-20'), 'E_end'].iloc[-1]


def test_target_tables_cover_all_four_required_dates():
    text = render_target_tables(
        N0 / 'q4-2_dispatch_10min.csv',
        N0 / 'q4-3_dispatch_10min.csv',
        settlement='N0')

    for date in TARGET_DATES:
        assert f'### {date}' in text
    assert text.count('紧急购电') >= 8
