"""Independent accounting, report tables and reference lower bound."""
from q2_solver import ROOT,OUT,DATA,read_data,optimize
import numpy as np, pandas as pd, json, hashlib, shutil, sys

def audit(x):
    checks={
      'rows':len(x),'duplicate_keys':int(x.duplicated(['date','interval_id']).sum()),
      'bus_residual_kwh':float(abs(x.g-x.unused+x.pv-x.spill+x.e+x.d-x.load-x.c).max()),
      'state_residual_kwh':float(abs(x.E_end-x.E_start-.9*x.c+x.d/.9).max()),
      'continuity_residual_kwh':float(np.max(abs(x.E_start.to_numpy()[1:]-x.E_end.to_numpy()[:-1]))),
      'power_excess_kwh':float(max(0,x.c.max()-5000/6,x.d.max()-5000/6)),
      'state_bound_excess_kwh':float(max(0,1200-x.E_start.min(),1200-x.E_end.min(),x.E_start.max()-10800,x.E_end.max()-10800)),
      'negative_flow_kwh':float(max(0,-x[['g','c','d','e','unused','spill']].min().min())),
      'overlap_count':int(((x.c>1e-6)&(x.d>1e-6)).sum()),
      'unused_excess_kwh':float(max(0,(x.unused-x.g).max())),
      'pv_spill_excess_kwh':float(max(0,(x.spill-x.pv).max())),
      'initial_error':float(abs(x.E_start.iloc[0]-6000)),
      'terminal_error':float(abs(x.E_end.iloc[-1]-6000))}
    assert checks['rows']==334*144
    assert all(v<1e-5 for k,v in checks.items() if k!='rows'),checks
    return checks

def intervals(day):
    vals=day.e.to_numpy(); result=[]; t=0
    def label(t): return f'{t//6:02d}:{t%6*10:02d}'
    while t<144:
        if vals[t]<=1e-7: t+=1; continue
        start=t
        while t<144 and vals[t]>1e-7: t+=1
        result.append([f'{label(start)}-{label(t)}',float(vals[start:t].sum())])
    return result

def lower_bound():
    a,L,V,p=read_data(); records=[]
    for month in range(2,13):
        ids=np.where(pd.to_datetime(a.date.unique()).month==month)[0]
        r=optimize((L[ids]-V[ids]).ravel(),np.tile(p,len(ids)),None,target=None)
        # Remove a conservative upper bound on the tiny throughput tie-break term.
        bound=r['objective']-1e-7*2*len(ids)*144*(5000/6)
        records.append({'month':month,'bound':bound,'primal_cost':float(r['g']@np.tile(p,len(ids))),
                        'seconds':r['seconds'],'initial':float(r['E'][0]),'terminal':float(r['E'][-1])})
    pd.DataFrame(records).to_csv(OUT/'perfect_information_monthly_bound.csv',index=False)
    print('Lower bound',sum(r['bound'] for r in records))

def summarize():
    labels=['M0','M1','M2','M3','M4','R0']; val={}
    for m in labels[:5]: val[m]=float(pd.read_csv(OUT/'validation'/f'{m}_daily.csv').cost.sum())
    selected=min(val,key=val.get)
    (OUT/'selection.json').write_text(json.dumps({'selected':selected,'selection_period':'2025-01-15/2025-01-31','validation_cost':val},indent=2),encoding='utf8')
    allchecks={}; metrics=[]; monthly=[]
    for path in sorted((OUT/'experiments').glob('*_daily.csv')):
        m=path.name.removesuffix('_daily.csv'); daily=pd.read_csv(path)
        x=pd.read_csv(OUT/'experiments'/f'{m}_dispatch.csv')
        allchecks[m]=audit(x)
        assert np.isclose((x.price*x.g+5*x.price*x.e).sum(),daily.cost.sum(),rtol=0,atol=1e-5)
        metrics.append({'model':m,'cost':float(daily.cost.sum()),'plan_cost':float(daily.plan_cost.sum()),
          'emergency_cost':float(daily.emergency_cost.sum()),'emergency_kwh':float(daily.emergency_kwh.sum()),
          'emergency_days':int((daily.emergency_kwh>1e-7).sum()),'p95_daily_cost':float(daily.cost.quantile(.95)),
          'tail5_mean':float(daily.cost.nlargest(17).mean()),'mean_seconds':float(daily.seconds.mean()),
          'max_seconds':float(daily.seconds.max()),'milp_days':int(daily.milp.sum()),
          'max_gap':float(daily.gap.max()),'unused_kwh':float(daily.unused_kwh.sum()),
          'spill_kwh':float(daily.spill_kwh.sum())})
        for mo,g in daily.groupby(daily.date.str[:7]):
            monthly.append({'model':m,'month':mo,'cost':float(g.cost.sum()),'emergency_kwh':float(g.emergency_kwh.sum())})
    metrics=pd.DataFrame(metrics); metrics.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_comparison.csv',index=False)
    x=pd.read_csv(OUT/'experiments'/f'{selected}_dispatch.csv')
    daily=pd.read_csv(OUT/'experiments'/f'{selected}_daily.csv')
    x.to_csv(OUT/'dispatch_10min.csv',index=False)
    daily.to_csv(OUT/'daily_summary.csv',index=False)
    battery=[]; er=[]; plan=[]; specified={}
    for date,day in x.groupby('date',sort=True):
        cost=float(day.price@day.g); emergencycost=float(5*day.price@day.e)
        plan.append(day.g.tolist()+[float(day.g.sum()),cost])
        blocks=[]
        for k in range(6):
            z=day.iloc[k*24:(k+1)*24]
            blocks.append([f'{k*4}:00-{(k+1)*4}:00',float(z.c.sum()),float(z.d.sum())])
            battery.append([date if k==0 else None,*blocks[-1],'0:00' if k==0 else '24:00' if k==1 else None,
                float(day.E_start.iloc[0]) if k==0 else float(day.E_end.iloc[-1]) if k==1 else None])
        events=intervals(day)
        if events:
            er.extend([[date if j==0 else None,*v] for j,v in enumerate(events)])
        else: er.append([date,'无',0.])
        if date in ['2025-03-20','2025-06-21','2025-09-23','2025-12-21']:
            specified[date]={'specified_g':day.iloc[[60,72,84,96,108,120]].g.tolist(),
              'plan_kwh':float(day.g.sum()),'emergency_kwh':float(day.e.sum()),
              'plan_cost':cost,'emergency_cost':emergencycost,'total_cost':cost+emergencycost,
              'blocks':blocks,'initial':float(day.E_start.iloc[0]),'terminal':float(day.E_end.iloc[-1]),
              'emergency_intervals':events}
    payload={'plan':plan,'battery':battery,'emergency':er,'selected':selected,'specified':specified}
    (OUT/'workbook_payload.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf8')
    # Paired moving-block bootstrap is descriptive after prior exploratory data use.
    rng=np.random.default_rng(2025); cis=[]
    ref=daily.cost.to_numpy()
    for m in labels:
        if m==selected: continue
        diff=pd.read_csv(OUT/'experiments'/f'{m}_daily.csv').cost.to_numpy()-ref
        for block in [7,14]:
            starts=rng.integers(0,len(diff)-block+1,size=(1000,int(np.ceil(len(diff)/block))))
            idx=(starts[:,:,None]+np.arange(block)).reshape(1000,-1)[:,:len(diff)]
            samples=diff[idx].mean(axis=1)
            cis.append({'comparator':m,'block_days':block,'mean_saving_yuan_per_day':float(diff.mean()),
                'lower95':float(np.quantile(samples,.025)),'upper95':float(np.quantile(samples,.975))})
    pd.DataFrame(cis).to_csv(OUT/'paired_cost_intervals.csv',index=False)
    f=np.load(OUT/'forecasts.npz'); _,L,V,_=read_data(); calibration=[]
    for conf in ['P0','P1']:
        pn=f[conf][:,0]-f[conf][:,1]; actual=L-V; errors=[]
        for i in range(31,365):
            past=np.arange(max(7,i-56),i); q=np.quantile(actual[past]-pn[past],.8,axis=0)+pn[i]
            errors.extend(actual[i]-q)
        errors=np.array(errors)
        calibration.append({'forecast':conf,'alpha':.8,'coverage':float((errors<=0).mean()),
                            'pinball_kwh':float(np.maximum(.8*errors,-.2*errors).mean())})
    pd.DataFrame(calibration).to_csv(OUT/'quantile_calibration.csv',index=False)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
      [DATA/'data_csv/annual_actual_10min.csv',DATA/'data_csv/typical_day_10min.csv',ROOT/'src/q2_solver.py']}
    (OUT/'verification.json').write_text(json.dumps({'physical':allchecks,'source_sha256':hashes},indent=2),encoding='utf8')
    print(json.dumps({'selected':selected,'validation':val,'comparison':metrics.to_dict('records')},indent=2))

if __name__=='__main__':
    if '--bound' in sys.argv: lower_bound()
    else: summarize()
