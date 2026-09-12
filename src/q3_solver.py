"""Q3-D: deterministic four-release planning, N0/N1 net settlement, causal F0.

AC-side kWh throughout. This baseline does not anticipate future forecast releases
or claim stochastic optimality. Public Q2 solvers are neither patched nor called.
"""
from pathlib import Path
import sys, json, argparse, hashlib
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, hstack, csr_matrix
from q3_forecast import pv_release_10min

CAP = 5000 / 6
DATA = ROOT / 'C题_标准化数据与修正模板/C_final_delivery/data_csv'


def settle(g, q, price, settlement='N0'):
    if settlement not in ('N0', 'N1'):
        raise ValueError('Supported final-net settlements: N0, N1')
    g, q, p = map(lambda x: np.asarray(x, float), (g, q, price))
    return p * (g + 1.5 * np.maximum(q-g, 0)
                + (.5 if settlement == 'N0' else -.5) * np.maximum(g-q, 0))


def optimize(load, pv, price, initial, original=None, settlement='N0',
             target=6000., cap=CAP):
    load, pv, price = map(lambda x: np.asarray(x, float), (load, pv, price))
    n = len(load)
    if n == 0 or any(x.shape != (n,) or not np.isfinite(x).all() for x in (load, pv, price)):
        raise ValueError('Expected finite, equal-length vectors')
    if np.min(load) < 0 or np.min(pv) < 0 or np.min(price) <= 0 or cap < 0:
        raise ValueError('Invalid load, PV, prices or cap')
    if not 1200 <= initial <= 10800 or not 1200 <= target <= 10800:
        raise ValueError('Invalid battery boundary')
    if settlement not in ('N0', 'N1'):
        raise ValueError('Unsupported settlement')
    g = None if original is None else np.asarray(original, float)
    if g is not None and (g.shape != (n,) or not np.isfinite(g).all() or np.min(g) < 0):
        raise ValueError('Invalid original contract')
    # q,c,d,E[n+1],unused,spill,up,down. No fictitious energy sink.
    q0,c0,d0,E0,w0,s0,u0,v0 = 0,n,2*n,3*n,4*n+1,5*n+1,6*n+1,7*n+1
    size = 8*n+1
    A = lil_matrix((2*n+(n if g is not None else 0), size))
    b = np.r_[np.zeros(n), load-pv, g if g is not None else []]
    for t in range(n):
        A[t,[c0+t,d0+t,E0+t,E0+t+1]] = [-.9,1/.9,-1,1]
        A[n+t,[q0+t,c0+t,d0+t,w0+t,s0+t]] = [1,-1,1,-1,-1]
        if g is not None:
            A[2*n+t,[q0+t,u0+t,v0+t]] = [1,-1,1]
    U = lil_matrix((n,size))
    for t in range(n):
        U[t,[w0+t,q0+t]] = [1,-1]
    A,U = A.tocsr(),U.tocsr()
    lo,hi = np.zeros(size),np.full(size,np.inf)
    hi[c0:d0+n] = cap
    lo[E0:E0+n+1],hi[E0:E0+n+1] = 1200,10800
    lo[E0] = hi[E0] = initial
    lo[E0+n] = hi[E0+n] = target
    hi[s0:s0+n] = pv
    cost = np.zeros(size)
    cost[c0:d0+n] = 1e-7
    if g is None:
        cost[:n] = price
        hi[u0:] = 0
    else:
        cost[u0:u0+n] = 1.5*price
        cost[v0:v0+n] = (.5 if settlement == 'N0' else -.5)*price
    result = linprog(cost,A_ub=U,b_ub=np.zeros(n),A_eq=A,b_eq=b,
                     bounds=list(zip(lo,hi)),method='highs',options={'time_limit':30})
    if not result.success:
        raise RuntimeError(result.message)
    x = result.x
    finite = np.isfinite(hi)
    dual = b @ result.eqlin.marginals + lo @ result.lower.marginals + hi[finite] @ result.upper.marginals[finite]
    gap = abs(float(result.fun-dual))
    binary = bool(np.any((x[c0:c0+n]>1e-7)&(x[d0:d0+n]>1e-7)))
    if binary:
        B = lil_matrix((2*n,size+n))
        for t in range(n):
            B[t,[c0+t,size+t]] = [1,-cap]
            B[n+t,[d0+t,size+t]] = [1,cap]
        result = milp(np.r_[cost,np.zeros(n)],integrality=np.r_[np.zeros(size),np.ones(n)],
                      bounds=Bounds(np.r_[lo,np.zeros(n)],np.r_[hi,np.ones(n)]),
                      constraints=[LinearConstraint(hstack([A,csr_matrix((len(b),n))]),b,b),
                                   LinearConstraint(hstack([U,csr_matrix((n,n))]),-np.inf,np.zeros(n)),
                                   LinearConstraint(B.tocsr(),-np.inf,np.r_[np.zeros(n),np.full(n,cap)])],
                      options={'time_limit':30,'mip_rel_gap':1e-8})
        if not result.success:
            raise RuntimeError(result.message)
        x,gap = result.x[:size],float(result.mip_gap)
    if max(np.max(abs(A@x-b)),np.max(U@x),np.max(lo-x),np.max(x-hi))>1e-6:
        raise RuntimeError('Optimization constraint violation')
    if np.any((x[c0:c0+n]>1e-6)&(x[d0:d0+n]>1e-6)):
        raise RuntimeError('Simultaneous charging/discharging')
    q = x[:n].copy()
    return dict(q=q,c=x[c0:c0+n],d=x[d0:d0+n],E=x[E0:E0+n+1],
                cost=float(price@q if g is None else settle(g,q,price,settlement).sum()),
                milp=binary,gap=gap)


def pv_at_issue(hourly, day, issue_hour):
    """Use the same release-to-slot mapping as historical risk calibration."""
    return pv_release_10min(hourly, day, issue_hour)


def _reserve_at_issue(reserve_by_issue, issue_hour, length):
    if reserve_by_issue is None or issue_hour not in reserve_by_issue:
        return np.zeros(length)
    reserve = np.asarray(reserve_by_issue[issue_hour], dtype=float)
    if reserve.shape != (length,) or not np.isfinite(reserve).all():
        raise ValueError('Risk reserve must be finite and match the remaining horizon')
    return reserve


def _planning_components(forecast_load, forecast_pv, net_reserve):
    """Represent a risk-adjusted net forecast with nonnegative planning inputs."""
    load = np.asarray(forecast_load, dtype=float)
    pv = np.asarray(forecast_pv, dtype=float)
    reserve = np.asarray(net_reserve, dtype=float)
    if load.shape != pv.shape or load.shape != reserve.shape:
        raise ValueError('Planning forecast components must have equal shapes')
    if (not np.isfinite(load).all() or not np.isfinite(pv).all()
            or np.any(load < 0) or np.any(pv < 0)):
        raise ValueError('Planning load and PV must be finite and nonnegative')
    adjusted_net = load - pv + reserve
    return np.maximum(adjusted_net, 0), np.maximum(-adjusted_net, 0)


def simulate_day(day, load, pv, forecast_load, hourly, price, initial,
                 updates=(6,12,18), settlement='N0', terminal=None,
                 reserve_by_issue=None):
    if any(h not in (6,12,18) for h in updates):
        raise ValueError('Invalid update time')
    first_pv = pv_at_issue(hourly,day,0)
    first_load, first_pv = _planning_components(
        forecast_load, first_pv, _reserve_at_issue(reserve_by_issue, 0, 144))
    first = optimize(first_load,first_pv,price,initial)
    g,q = first['q'].copy(),first['q'].copy()
    flows = {k:np.zeros(144) for k in ('c','d','e','unused','spill')}
    E = np.zeros(145); E[0] = initial
    versions,solves = [],[first]
    for t in range(144):
        hour = t//6
        if t%36 == 0 and hour in updates:
            update_pv = pv_at_issue(hourly,day,hour)
            adjusted_load, update_pv = _planning_components(
                forecast_load[t:], update_pv,
                _reserve_at_issue(reserve_by_issue, hour, 144-t))
            solved = optimize(adjusted_load,update_pv,price[t:],
                              E[t],original=g[t:],settlement=settlement)
            prev = q[t:].copy(); q[t:] = solved['q']; solves.append(solved)
            versions.extend(dict(issue_hour=hour,interval_id=j,original_g=float(g[j]),
                                 previous_q=float(prev[j-t]),q=float(q[j]),
                                 delta=float(q[j]-g[j]),change=float(q[j]-prev[j-t])) for j in range(t,144))
        state = E[t]; surplus = q[t]+pv[t]-load[t]
        if terminal is not None and state>terminal:
            c,d = 0.,min(CAP,load[t],.9*(state-terminal))
        else:
            floor = 1200 if terminal is None else max(1200,terminal-(143-t)*.9*CAP)
            ceiling = 10800 if terminal is None else terminal
            nxt = np.clip(state+.9*max(surplus,0)-max(-surplus,0)/.9,
                          max(floor,state-CAP/.9),min(ceiling,state+.9*CAP))
            c,d = max(nxt-state,0)/.9,max(state-nxt,0)*.9
        balance = surplus+d-c
        e,excess = max(-balance,0),max(balance,0)
        w = min(q[t],excess); s = max(excess-w,0)
        for k,v in zip(flows,(c,d,e,w,s)):
            flows[k][t] = v
        E[t+1] = state+.9*c-d/.9
    if terminal is not None and abs(E[-1]-terminal)>1e-6:
        raise RuntimeError('Closing SOC unreachable without illegal disposal')
    r = dict(g=g,q=q,E=E,versions=versions,**flows)
    r['actual_grid'] = q-r['unused']+r['e']
    r['cost'] = float(settle(g,q,price,settlement).sum()+5*price@r['e'])
    r['milp_solves'] = sum(s['milp'] for s in solves)
    r['max_lp_gap'] = max((s['gap'] for s in solves if not s['milp']),default=0.)
    r['max_mip_gap'] = max((s['gap'] for s in solves if s['milp']),default=0.)
    audit(r,load,pv,price,settlement)
    return r


def audit(r, load, pv, price, settlement):
    checks = dict(bus=float(np.max(abs(r['q']-r['unused']+r['e']+pv-r['spill']+r['d']-load-r['c']))),
                  state=float(np.max(abs(np.diff(r['E'])-.9*r['c']+r['d']/.9))),
                  bounds=float(max(0,1200-r['E'].min(),r['E'].max()-10800)),
                  power=float(max(0,r['c'].max()-CAP,r['d'].max()-CAP)),
                  disposal=float(max(0,np.max(r['unused']-r['q']),np.max(r['spill']-pv))),
                  overlap=int(np.sum((r['c']>1e-6)&(r['d']>1e-6))),
                  negative=float(max(0,-min(r[k].min() for k in ('g','q','c','d','e','unused','spill')))),
                  cost=abs(float(settle(r['g'],r['q'],price,settlement).sum()+5*price@r['e'])-r['cost']))
    if any(v>1e-6 for v in checks.values()):
        raise RuntimeError(str(checks))
    return checks


def run(start=31,end=38):
    if not 7<=start<end<=365:
        raise ValueError('Invalid evaluation range')
    actual = pd.read_csv(DATA/'annual_actual_10min.csv')
    price = pd.read_csv(DATA/'typical_day_10min.csv').price_yuan_per_kwh.to_numpy()
    l = actual.load_kwh_10min.to_numpy().reshape(365,144)
    v = actual.pv_actual_kwh_10min.to_numpy().reshape(365,144)
    f = pd.read_csv(DATA/'pv_forecast_hourly.csv')
    f['day'] = (pd.to_datetime(f.issue_date)-pd.Timestamp('2025-01-01')).dt.days
    f['r'] = f.issue_time.str.split(':').str[0].astype(int)//6
    if f.duplicated(['day','r','horizon_hour']).any() or len(f)!=35040:
        raise ValueError('Invalid forecast keys')
    hourly = np.full((365,4,24),np.nan)
    hourly[f.day,f.r,f.horizon_hour-1] = f.pv_forecast_kw
    if not np.isfinite(hourly).all():
        raise ValueError('Missing forecast')
    # Causal weekly load baseline; no retraining or future actual load in plans.
    pred = np.full_like(l,np.nan); pred[7:] = l[:-7]
    folder = ROOT/f'outputs/q3_modeling/pilot_{start}_{end}'
    folder.mkdir(parents=True,exist_ok=True)
    summary = []
    for label,updates,settlement in [('D0',(),'N0'),('Dall',(6,12,18),'N0'),('Dall_refund',(6,12,18),'N1')]:
        state,days,chunks,versions = 6000.,[],[],[]
        for i in range(start,end):
            r = simulate_day(i,l[i],v[i],pred[i],hourly,price,state,updates,settlement,
                             terminal=6000. if i==end-1 else None)
            date = str(pd.Timestamp('2025-01-01')+pd.Timedelta(days=i))[:10]
            pc = float(price@r['g']); up = float(1.5*price@np.maximum(r['q']-r['g'],0))
            down = float(.5*price@np.maximum(r['g']-r['q'],0))
            refund = 2*down if settlement=='N1' else 0.
            days.append(dict(date=date,cost=r['cost'],plan_cost=pc,up_cost=up,down_fee=down,refund=refund,
                             emergency_cost=float(5*price@r['e']),emergency_kwh=float(r['e'].sum()),
                             initial=state,terminal=float(r['E'][-1]),max_lp_gap=r['max_lp_gap'],
                             max_mip_gap=r['max_mip_gap'],milp_solves=r['milp_solves']))
            frame = pd.DataFrame({k:r[k] for k in ('g','q','c','d','e','unused','spill','actual_grid')})
            frame['date'],frame['interval_id'] = date,np.arange(144)
            frame['E_start'],frame['E_end'] = r['E'][:-1],r['E'][1:]
            frame['load'],frame['pv'],frame['price'] = l[i],v[i],price
            frame['delta'] = frame.q-frame.g
            chunks.append(frame); versions.extend(dict(date=date,**z) for z in r['versions'])
            state = float(r['E'][-1])
        daily,detail = pd.DataFrame(days),pd.concat(chunks,ignore_index=True)
        if not np.allclose(detail.E_end.to_numpy()[:-1],detail.E_start.to_numpy()[1:],atol=1e-6,rtol=0):
            raise RuntimeError('State discontinuity')
        daily.to_csv(folder/f'{label}_daily.csv',index=False)
        detail.to_csv(folder/f'{label}_dispatch.csv',index=False)
        pd.DataFrame(versions).to_csv(folder/f'{label}_versions.csv',index=False)
        summary.append(dict(label=label,settlement=settlement,updates=updates,days=end-start,
                            cost=float(daily.cost.sum()),emergency_kwh=float(daily.emergency_kwh.sum()),
                            final_soc=state,max_lp_gap=float(daily.max_lp_gap.max())))
        print(json.dumps(summary[-1]),flush=True)
    files = [DATA/'annual_actual_10min.csv',DATA/'pv_forecast_hourly.csv',DATA/'typical_day_10min.csv',Path(__file__)]
    report = dict(results=summary,input_and_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  scope='Q3-D deterministic pilot, H0 hourly mapping, weekly load forecast, no risk calibration; not final selected policy')
    (folder/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',type=int,default=31)
    parser.add_argument('--end',type=int,default=38)
    args = parser.parse_args(); run(args.start,args.end)
