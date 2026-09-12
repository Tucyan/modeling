"""Q1 deterministic dispatch, with an LP certificate and physical checks."""
from pathlib import Path
import sysconfig
import sys
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / '.local-deps'
abi_suffix = (sysconfig.get_config_var('EXT_SUFFIX') or '').lower()
if abi_suffix and LOCAL_DEPS.exists() and any(str(p).lower().endswith(abi_suffix) for p in LOCAL_DEPS.rglob('*.pyd')):
    sys.path.insert(0, str(LOCAL_DEPS))
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog


def solve_dispatch(load, pv, price, *, dt=1/6, initial=6000, emin=1200,
                   emax=10800, pmax=5000, eta_c=.9, eta_d=.9):
    load, pv, price = (np.asarray(v, dtype=float) for v in (load, pv, price))
    n = len(load)
    assert n and load.shape == pv.shape == price.shape
    assert np.isfinite(np.r_[load,pv,price]).all()
    assert min(load.min(),pv.min()) >= 0 and price.min() > 0
    assert emin <= initial <= emax and dt > 0 and pmax > 0
    assert 0 < eta_c <= 1 and 0 < eta_d <= 1
    size = 5*n+1
    a = np.zeros((2*n,size))
    b = np.r_[load-pv,np.zeros(n)]
    for t in range(n):
        a[t,[t,n+t,2*n+t,3*n+t]] = [1,-1,1,-1]
        a[n+t,[n+t,2*n+t,4*n+t,4*n+t+1]] = [-eta_c,1/eta_d,-1,1]
    cost = np.r_[price,np.zeros(size-n)]
    lower = np.r_[np.zeros(4*n),np.full(n+1,emin)]
    upper = np.r_[np.full(n,np.inf),np.full(2*n,pmax*dt),pv,np.full(n+1,emax)]
    lower[4*n] = upper[4*n] = initial
    lower[-1] = upper[-1] = initial
    r = linprog(cost,A_eq=a,b_eq=b,bounds=list(zip(lower,upper)),
                method='highs',options={'time_limit':30})
    if not r.success:
        raise RuntimeError(r.message)
    g,c,d,s = np.split(r.x[:4*n],4)
    energy = r.x[4*n:]
    tol = 1e-7
    checks = {
        'bus_residual_kwh':float(np.max(np.abs(g+pv-s+d-load-c))),
        'storage_residual_kwh':float(np.max(np.abs(np.diff(energy)-eta_c*c+d/eta_d))),
        'lower_bound_violation_kwh':float(np.maximum(lower-r.x,0).max()),
        'upper_bound_violation_kwh':float(np.maximum(r.x-upper,0).max()),
        'overlap_intervals':int(np.sum((c>tol)&(d>tol))),
        'initial_error_kwh':float(abs(energy[0]-initial)),
        'terminal_error_kwh':float(abs(energy[-1]-initial)),
    }
    assert all(v<=tol for v in checks.values()), checks
    # This solver deliberately stops rather than presenting a nonexclusive LP as feasible.
    y,lo,hi = r.eqlin.marginals,r.lower.marginals,r.upper.marginals
    finite = np.isfinite(upper)
    dual = float(b@y+lower@lo+upper[finite]@hi[finite])
    certificate = {
        'lp_status':r.message,
        'lp_dual_objective_yuan':dual,
        'primal_dual_gap_yuan':float(abs(r.fun-dual)),
        'stationarity_residual':float(np.max(np.abs(cost-a.T@y-lo-hi))),
        'dual_sign_violation':float(max(0,-lo.min(),hi.max())),
        'full_milp_feasible':checks['overlap_intervals']==0,
    }
    assert certificate['primal_dual_gap_yuan'] < tol
    assert certificate['stationarity_residual'] < tol
    assert certificate['dual_sign_violation'] < tol
    return dict(cost=float(price@g),purchase=g.tolist(),charge=c.tolist(),
                discharge=d.tolist(),spill=s.tolist(),energy=energy.tolist(),
                checks=checks,certificate=certificate)


if __name__ == '__main__':
    path=ROOT/'C题_标准化数据与修正模板/C_final_delivery/data_csv/typical_day_10min.csv'
    data=pd.read_csv(path)
    assert data.interval_id.tolist()==list(range(144))
    assert np.allclose(data.load_kw/6,data.load_kwh_10min,atol=1e-9,rtol=0)
    assert np.allclose(data.pv_forecast_kw/6,data.pv_forecast_kwh_10min,atol=1e-9,rtol=0)
    r=solve_dispatch(data.load_kwh_10min,data.pv_forecast_kwh_10min,data.price_yuan_per_kwh)
    out=ROOT/'outputs/01a08b78-ed6b-7e60-b40d-9f266855d1ef/q1'
    out.mkdir(parents=True,exist_ok=True)
    detail=data.copy()
    for column,key in [('purchase_kwh','purchase'),('charge_kwh','charge'),
                       ('discharge_kwh','discharge'),('spill_kwh','spill')]:
        detail[column]=r[key]
    detail['energy_start_kwh']=r['energy'][:-1]
    detail['energy_end_kwh']=r['energy'][1:]
    detail['cost_yuan']=detail.price_yuan_per_kwh*detail.purchase_kwh
    detail.to_csv(out/'dispatch_10min.csv',index=False,encoding='utf-8-sig')
    baseline=np.maximum(data.load_kwh_10min-data.pv_forecast_kwh_10min,0)
    basecost=float(baseline@data.price_yuan_per_kwh)
    blocks=[]
    for k in range(6):
        rows=detail.iloc[k*24:(k+1)*24]
        blocks.append(dict(interval=f'{k*4}:00-{(k+1)*4}:00',
                           charge=float(rows.charge_kwh.sum()),discharge=float(rows.discharge_kwh.sum()),
                           energy_start=float(rows.energy_start_kwh.iloc[0]),energy_end=float(rows.energy_end_kwh.iloc[-1])))
    totals=dict(purchase_kwh=sum(r['purchase']),charge_kwh=sum(r['charge']),
                discharge_kwh=sum(r['discharge']),spill_kwh=sum(r['spill']),
                load_kwh=float(data.load_kwh_10min.sum()),pv_kwh=float(data.pv_forecast_kwh_10min.sum()),
                loss_kwh=sum(r['charge'])-sum(r['discharge']),cost_yuan=r['cost'],
                baseline_purchase_kwh=float(baseline.sum()),baseline_cost_yuan=basecost,
                saving_yuan=basecost-r['cost'],saving_pct=100*(basecost-r['cost'])/basecost)
    payload=dict(**r,totals=totals,blocks=blocks,
                 specified=[dict(interval=data.interval_label.iloc[t],purchase=r['purchase'][t]) for t in [60,72,84,96,108,120]],
                 source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 scipy_version=scipy.__version__,assumptions=dict(dt=1/6,eta_c=.9,eta_d=.9,initial=6000,emin=1200,emax=10800,pmax=5000))
    (out/'solution.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(totals=totals,blocks=blocks,specified=payload['specified'],certificate=r['certificate']),ensure_ascii=True,indent=2))
