"""Read-only policy replay and settlement checks for the Q2/Q3 review."""
from pathlib import Path
import sys, json
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import q3_solver as q3
from q3_verify_delivery import verify_dispatch
from q2_review_experiments import execute_closed

folder = ROOT / 'outputs/q2_review_v2/experiments'
x = pd.read_csv(folder / 'Rolling_T0_dispatch.csv')
daily = pd.read_csv(folder / 'Rolling_T0_daily.csv')
assert np.isfinite(x.select_dtypes(include='number')).all().all()
x['q'] = x.g
versions = pd.DataFrame(columns=['date','issue_hour','interval_id','original_g','previous_q','q','delta','change'])
result = {'q2_as_q3_no_adjustment': verify_dispatch(daily, x, versions)}
result['q2_N1_cost'] = float(q3.settle(x.g, x.q, x.price, 'N1').sum() + 5*x.price@x.e)
# Inject the observed original contract only to isolate controller equivalence.
# This diagnostic is not a new planning policy or a future-informed backtest.
errors = []
for _, frame in x.groupby('date', sort=False):
    terminal = 6000. if frame.date.iloc[0] == '2025-12-31' else None
    initial = float(frame.E_start.iloc[0])
    contract = frame.g.to_numpy()
    with patch.object(q3, 'optimize', return_value={'q':contract,'milp':False,'gap':0.}):
        replay = q3.simulate_day(0, frame.load.to_numpy(), frame.pv.to_numpy(),
            frame.load.to_numpy(), np.zeros((1,4,24)), frame.price.to_numpy(),
            initial, updates=(), terminal=terminal)
    for key in ('c','d','e','unused','spill'):
        errors.append(float(np.max(abs(replay[key]-frame[key].to_numpy()))))
    errors.append(float(np.max(abs(replay['E'][1:]-frame.E_end.to_numpy()))))
result['all_334_days_controller_replay_max_error_kwh'] = max(errors)
comparison = pd.read_csv(ROOT / 'outputs/q3_final/comparison.csv').set_index('label')
result['q3_costs'] = comparison.loc[['calibrated_R0','calibrated_Rall','N1_Rall'], ['cost','emergency_kwh']].to_dict('index')
result['q3_R0_minus_q2_yuan'] = float(comparison.loc['calibrated_R0','cost'] - daily.cost.sum())
result['q3_Rall_saving_vs_own_R0_yuan'] = float(comparison.loc['calibrated_R0','cost'] - comparison.loc['calibrated_Rall','cost'])
result['N0_to_N1_saving_yuan'] = float(comparison.loc['calibrated_Rall','cost'] - comparison.loc['N1_Rall','cost'])
assert max(errors) < 1e-6
assert abs(result['q2_N1_cost'] - daily.cost.sum()) < 1e-6
output = Path(__file__).with_name('evidence.json')
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(output.read_text(encoding='utf-8'))
