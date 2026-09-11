"""Versioned causal Q2 experiments; all flow quantities are AC-side kWh."""
from dataclasses import dataclass, asdict, replace
import argparse, hashlib, json, time
from q2_solver import ROOT, OUT as OLD, DATA, np, pd, optimize, execute, read_data

OUT=ROOT/'outputs/q2_review_v2'
EPS=1e-7

@dataclass(frozen=True)
class Config:
    model: str='M1'
    terminal: str='T0'
    alpha: float=.8
    initial: float=6000.
    close: bool=True
    error_scale: float=1.
    controller: str='F0'
    scenarios: int=14
    closure: str='safe'

def execute_closed(g,load,pv,initial,terminal):
    """Causal closing-day controller; no battery energy may be discarded."""
    n=len(g); state=initial; E=[state]
    flows={key:[] for key in ('c','d','e','unused','spill')}
    for t in range(n):
        surplus=g[t]+pv[t]-load[t]
        floor=max(1200.,terminal-(n-t-1)*.9*5000/6)
        if state>terminal:
            c=0.; d=min(5000/6,load[t],.9*(state-terminal))
        else:
            next_state=np.clip(state+.9*max(surplus,0)-max(-surplus,0)/.9,
                max(floor,state-5000/6/.9),min(terminal,state+.9*5000/6))
            c=max(next_state-state,0)/.9; d=max(state-next_state,0)*.9
        balance=surplus+d-c; e=max(-balance,0); excess=max(balance,0)
        unused=min(g[t],excess); spill=max(0,excess-unused)
        if spill>pv[t]+1e-6: raise ValueError('Terminal restoration would dump battery energy')
        state+=.9*c-d/.9; E.append(state)
        for key,value in zip(flows,(c,d,e,unused,spill)): flows[key].append(value)
    if abs(state-terminal)>1e-6: raise ValueError('Terminal target unreachable under observed demand; no artificial discharge allowed')
    return dict(g=np.asarray(g),E=np.asarray(E),**{k:np.asarray(v) for k,v in flows.items()})

def terminal_target(rule,initial):
    return {'T0':6000.,'T1':initial,'T2':4800.,'T3':7200.,'T4':None}[rule]

def history_window(day):
    return max(14,day-56),day

def planning_input(actual_net,predicted_net,day,model,alpha,scenarios):
    past=np.arange(max(7,day-56),day)
    residual=actual_net[past]-predicted_net[past]
    if model in ('M1','M3','R0'):
        return predicted_net[day]+np.quantile(residual,alpha,axis=0)
    if model=='M4':
        ids=np.unique(np.linspace(0,len(past)-1,min(scenarios,len(past)),dtype=int))
        return predicted_net[day]+residual[ids]
    return predicted_net[day]

def load_inputs():
    _,l,v,p=read_data()
    with np.load(OLD/'forecasts.npz') as f:
        forecasts={key:f[key].copy() for key in ('P0','P1')}
    return l,v,p,forecasts

def summarize(daily,detail):
    positive=detail.e.to_numpy()>EPS
    starts=positive & ~np.r_[False,positive[:-1]]
    day_start=detail.interval_id.to_numpy()==0
    events=int((starts | (positive & day_start)).sum())
    return dict(cost=float(daily.cost.sum()),plan_cost=float(daily.plan_cost.sum()),
        emergency_cost=float(daily.emergency_cost.sum()),emergency_kwh=float(daily.emergency_kwh.sum()),
        emergency_days=int((daily.emergency_kwh>EPS).sum()),emergency_events=events,
        emergency_slots=int(positive.sum()),unused_kwh=float(detail.unused.sum()),spill_kwh=float(detail.spill.sum()),
        low_initial_days=int((daily.initial<2400).sum()),first_E=float(daily.initial.iloc[0]),
        last_E=float(daily.terminal.iloc[-1]),milp_days=int(daily.milp.sum()),
        max_lp_absolute_gap_yuan=float(daily.lp_absolute_gap_yuan.max()),
        max_mip_relative_gap=float(daily.mip_relative_gap.max()),max_seconds=float(daily.seconds.max()),
        mean_seconds=float(daily.seconds.mean()),p95_cost=float(daily.cost.quantile(.95)))

def simulate(config,start=31,end=365,inputs=None,choices=None):
    l,v,p,forecasts=load_inputs() if inputs is None else inputs
    state=config.initial; days=[]; chunks=[]
    for i in range(start,end):
        tick=time.perf_counter(); model=config.model if choices is None else choices[i]
        forecast=forecasts['P0' if model in ('M0','M1','R0') else 'P1']
        # Synthetic actuals are an offline stress path; only completed history enters planning.
        if config.error_scale==1:
            loads,pvs=l,v
        else:
            base=forecasts['P0']
            loads=np.maximum(0,base[:,0]+config.error_scale*(l-base[:,0]))
            pvs=np.maximum(0,base[:,1]+config.error_scale*(v-base[:,1]))
        pn=forecast[:,0]-forecast[:,1]
        inp=planning_input(loads-pvs,pn,i,model,config.alpha,config.scenarios)
        target=terminal_target(config.terminal,state)
        if model=='R0':
            opt=dict(g=np.maximum(inp,0),E=np.full(145,state),milp=False,gap=0.)
        else:
            opt=optimize(inp,p,state,target=target)
        close=config.initial if config.close and i==end-1 else None
        if close is not None and config.closure=='safe' and model!='R0':
            r=execute_closed(opt['g'],loads[i],pvs[i],state,close)
        elif config.controller=='F1':
            from q2_realtime_control import execute_mpc
            r=execute_mpc(opt['g'],loads[i],pvs[i],state,pn[i],p,terminal=close)
        else:
            r=execute(opt['g'],loads[i],pvs[i],state,terminal=close,storage=model!='R0')
        date=(pd.Timestamp('2025-01-01')+pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        pc=float(p@r['g']); ec=float(5*p@r['e'])
        day=dict(day=i,date=date,model=model,plan_cost=pc,emergency_cost=ec,cost=pc+ec,
            emergency_kwh=float(r['e'].sum()),initial=state,terminal=float(r['E'][-1]),
            planned_initial=float(opt['E'][0]),planned_terminal=float(opt['E'][-1]),
            planning_target=np.nan if target is None else target,milp=bool(opt['milp']),
            lp_absolute_gap_yuan=0. if opt['milp'] else float(opt['gap']),
            mip_relative_gap=float(opt['gap']) if opt['milp'] else 0.,seconds=time.perf_counter()-tick)
        days.append(day)
        frame=pd.DataFrame({key:r[key] for key in ('g','c','d','e','unused','spill')})
        frame.insert(0,'interval_id',np.arange(144)); frame.insert(0,'date',date)
        frame['E_start']=r['E'][:-1]; frame['E_end']=r['E'][1:]
        frame['load']=loads[i]; frame['pv']=pvs[i]; frame['price']=p
        chunks.append(frame); state=float(r['E'][-1])
    return pd.DataFrame(days),pd.concat(chunks,ignore_index=True)

def audit(daily,x,config):
    e=x.e.to_numpy(); E=x.E_start.to_numpy(); end=x.E_end.to_numpy()
    checks=dict(bus=float(abs(x.g-x.unused+x.pv-x.spill+x.e+x.d-x.load-x.c).max()),
        state=float(abs(x.E_end-x.E_start-.9*x.c+x.d/.9).max()),
        continuity=float(np.max(abs(E[1:]-end[:-1]))),
        bounds=float(max(0,1200-E.min(),1200-end.min(),E.max()-10800,end.max()-10800)),
        power=float(max(0,x.c.max()-5000/6,x.d.max()-5000/6)),
        overlap=int(((x.c>1e-6)&(x.d>1e-6)).sum()),
        negative=float(max(0,-x[['g','c','d','e','unused','spill']].min().min())),
        disposal=float(max(0,(x.unused-x.g).max(),(x.spill-x.pv).max())),
        cost=float(abs((x.price*x.g+5*x.price*x.e).sum()-daily.cost.sum())),
        initial=float(abs(E[0]-config.initial)),
        terminal=float(abs(end[-1]-config.initial)) if config.close else 0.)
    assert len(x)==len(daily)*144 and not x.duplicated(['date','interval_id']).any()
    assert all(z<1e-5 for z in checks.values()),checks
    if config.model!='R0' and config.terminal!='T4':
        assert np.max(abs(daily.planned_terminal-daily.planning_target))<1e-6
    assert np.max(abs(daily.planned_initial-daily.initial))<1e-6
    assert np.array_equal(x.interval_id,np.tile(np.arange(144),len(daily)))
    return checks

def save_run(label,config,start=31,end=365,inputs=None,choices=None):
    folder=OUT/'experiments'; folder.mkdir(parents=True,exist_ok=True)
    daily,detail=simulate(config,start,end,inputs,choices)
    checks=audit(daily,detail,config)
    daily.to_csv(folder/f'{label}_daily.csv',index=False)
    detail.to_csv(folder/f'{label}_dispatch.csv',index=False)
    result=dict(label=label,config=asdict(config),start=start,end=end,checks=checks,**summarize(daily,detail))
    (folder/f'{label}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('label','cost','emergency_kwh','last_E','max_seconds')}),flush=True)
    return result

def matrix(stage):
    inputs=load_inputs(); configs={}
    if stage=='terminal':
        configs={f'M1_{t}':Config(terminal=t) for t in ('T0','T1','T2','T3','T4')}
        configs.update({f'M1_{t}_open':Config(terminal=t,close=False) for t in ('T0','T1')})
    elif stage=='risk':
        configs={f'M1_{t}_a{a}':Config(terminal=t,alpha=a) for t in ('T0','T1') for a in (.7,.75,.85,.9)}
    elif stage=='stress':
        configs={f'M1_{t}_initial{e}':Config(terminal=t,initial=e) for t in ('T0','T1') for e in (3600.,8400.)}
        configs.update({f'M1_{t}_scale{k}':Config(terminal=t,error_scale=k) for t in ('T0','T1') for k in (.75,1.25)})
    elif stage=='models':
        configs={f'{m}_T0':Config(model=m) for m in ('M0','M2','M3','M4','R0')}
    for label,config in configs.items(): save_run(label,config,inputs=inputs)
    sources=[DATA/'data_csv/annual_actual_10min.csv',DATA/'data_csv/typical_day_10min.csv',OLD/'forecasts.npz',ROOT/'src/q2_solver.py',ROOT/'src/q2_review_experiments.py']
    (OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['terminal','risk','stress','models','trial'])
    args=parser.parse_args()
    if args.stage=='trial':
        save_run('trial_T0',Config(),31,38); save_run('trial_T1',Config(terminal='T1'),31,38)
    else: matrix(args.stage)
