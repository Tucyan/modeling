"""Causal Q2 experiments. Energy variables are AC-side kWh."""
from pathlib import Path
import sysconfig
import sys, os, json, time, hashlib, argparse
ROOT=Path(__file__).resolve().parents[1]
# `.local-deps` is an optional cache. Only use it when compiled wheels match
# this interpreter; a copied cache from another Python minor version must not
# shadow packages installed from requirements.txt.
LOCAL_DEPS = ROOT / '.local-deps'
abi_suffix = (sysconfig.get_config_var('EXT_SUFFIX') or '').lower()
if abi_suffix and LOCAL_DEPS.exists() and any(str(p).lower().endswith(abi_suffix) for p in LOCAL_DEPS.rglob('*.pyd')):
    sys.path.insert(0, str(LOCAL_DEPS))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix
OUT=ROOT/'outputs/01a08b78-ed6b-7e60-b40d-9f266855d1ef/q2'
DATA=ROOT/'C题_标准化数据与修正模板/C_final_delivery'
CAP=5000/6

def optimize(net,price,initial,target=6000.,cap=CAP):
    net=np.atleast_2d(net); k,n=net.shape
    # Shared purchase, charge, discharge and state; scenario-specific emergency/disposal.
    size=4*n+1+2*k*n; A=lil_matrix((n+k*n,size))
    b=np.r_[np.zeros(n),net.ravel()]
    for t in range(n):
        A[t,[n+t,2*n+t,3*n+t,3*n+t+1]]=[-.9,1/.9,-1,1]
    for s in range(k):
        for t in range(n):
            A[n+s*n+t,[t,n+t,2*n+t,4*n+1+s*n+t,4*n+1+k*n+s*n+t]]=[1,-1,1,1,-1]
    A=A.tocsr()
    cost=np.r_[price,np.full(2*n,1e-7),np.zeros(n+1),np.tile(5*price/k,k),np.zeros(k*n)]
    lo=np.zeros(size); hi=np.full(size,np.inf)
    hi[n:3*n]=cap; lo[3*n:4*n+1]=1200; hi[3*n:4*n+1]=10800
    if initial is not None: lo[3*n]=hi[3*n]=initial
    if target is not None: lo[4*n]=hi[4*n]=target
    if k==1: hi[4*n+1:4*n+1+k*n]=0
    start=time.perf_counter()
    res=linprog(cost,A_eq=A,b_eq=b,bounds=list(zip(lo,hi)),method='highs',options={'time_limit':30})
    if not res.success: raise RuntimeError(res.message)
    x=res.x; binary=False
    finite=np.isfinite(hi)
    dual=float(b@res.eqlin.marginals+lo@res.lower.marginals+hi[finite]@res.upper.marginals[finite])
    gap=float(abs(res.fun-dual))
    if np.any((x[n:2*n]>1e-6)&(x[2*n:3*n]>1e-6)):
        binary=True
        B=lil_matrix((2*n,size+n))
        for t in range(n):
            B[t,n+t]=1; B[t,size+t]=-cap
            B[n+t,2*n+t]=1; B[n+t,size+t]=cap
        res=milp(np.r_[cost,np.zeros(n)],integrality=np.r_[np.zeros(size),np.ones(n)],
            bounds=Bounds(np.r_[lo,np.zeros(n)],np.r_[hi,np.ones(n)]),
            constraints=[LinearConstraint(hstack([A,csr_matrix((A.shape[0],n))]),b,b),
                         LinearConstraint(B.tocsr(),-np.inf,np.r_[np.zeros(n),np.full(n,cap)])],
            options={'time_limit':30,'mip_rel_gap':1e-8})
        if not res.success: raise RuntimeError(res.message)
        x=res.x[:size]; gap=float(res.mip_gap)
    assert np.max(abs(A@x-b))<1e-5
    assert not np.any((x[n:2*n]>1e-5)&(x[2*n:3*n]>1e-5))
    return dict(g=x[:n],c=x[n:2*n],d=x[2*n:3*n],E=x[3*n:4*n+1],
                objective=float(cost@x),seconds=time.perf_counter()-start,milp=binary,gap=gap)

def execute(g,load,pv,initial,terminal=None,storage=True):
    g,load,pv=map(np.asarray,(g,load,pv)); n=len(g)
    E=np.zeros(n+1); E[0]=initial
    c,d,e,unused,spill=[np.zeros(n) for _ in range(5)]
    for t in range(n):
        surplus=g[t]+pv[t]-load[t]
        if storage:
            delta=.9*max(surplus,0)-max(-surplus,0)/.9
            floor=1200 if terminal is None else max(1200,terminal-(n-t-1)*.9*CAP)
            ceiling=10800 if terminal is None else min(10800,terminal+(n-t-1)*CAP/.9)
            nextE=np.clip(E[t]+delta,max(floor,E[t]-CAP/.9),min(ceiling,E[t]+.9*CAP))
            c[t]=max(nextE-E[t],0)/.9; d[t]=max(E[t]-nextE,0)*.9
        balance=surplus+d[t]-c[t]
        e[t]=max(-balance,0); excess=max(balance,0)
        unused[t]=min(g[t],excess); spill[t]=max(excess-unused[t],0)
        E[t+1]=E[t]+.9*c[t]-d[t]/.9
    return dict(g=g,c=c,d=d,e=e,unused=unused,spill=spill,E=E)

def features(y,i,kind):
    t=np.arange(144); base=y[i-7] if kind=='load' else y[i-7:i].mean(axis=0)
    return np.column_stack([base,y[i-1],y[i-7],y[i-7:i].mean(axis=0),
        y[max(0,i-14):i].mean(axis=0),y[i-3:i].mean(axis=0),
        np.sin(t*2*np.pi/144),np.cos(t*2*np.pi/144),
        np.full(144,(i+2)%7/6),np.full(144,np.sin(i*2*np.pi/365)),
        np.full(144,np.cos(i*2*np.pi/365))])

def read_data():
    a=pd.read_csv(DATA/'data_csv/annual_actual_10min.csv')
    assert len(a)==365*144 and not a.duplicated(['date','interval_id']).any()
    assert np.array_equal(a.interval_id,np.tile(np.arange(144),365))
    p=pd.read_csv(DATA/'data_csv/typical_day_10min.csv').price_yuan_per_kwh.to_numpy()
    load=a.load_kw.to_numpy().reshape(365,144)/6
    pv=a.pv_actual_kw.to_numpy().reshape(365,144)/6
    assert np.isfinite(load).all() and np.isfinite(pv).all()
    return a,load,pv,p

def predict_all():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _,load,pv,_=read_data()
    pred=np.full((365,2,3,144),np.nan); selected=np.zeros((365,2),int); elapsed=[]
    for j,y in enumerate([load,pv]):
        kind=['load','pv'][j]; models=None
        for i in range(7,365):
            tick=time.perf_counter(); X=features(y,i,kind)
            pred[i,j,0]=X[:,0]
            if i>=14 and (models is None or (i-14)%7==0):
                train=range(max(7,i-56),i)
                xx=np.vstack([features(y,h,kind) for h in train])
                yy=np.concatenate([y[h]-features(y,h,kind)[:,0] for h in train])
                models=[make_pipeline(StandardScaler(),Ridge(alpha=100.)),
                    HistGradientBoostingRegressor(max_iter=40,max_leaf_nodes=7,max_bins=64,
                        l2_regularization=10.,learning_rate=.08,early_stopping=False,random_state=2025)]
                for model in models: model.fit(xx,yy)
            for m in [1,2]:
                pred[i,j,m]=np.maximum(0,X[:,0]+models[m-1].predict(X)) if models else X[:,0]
                if j==1: pred[i,j,m][np.max(y[max(0,i-28):i],axis=0)<1e-9]=0
            history=np.arange(max(7,i-14),i)
            if len(history)>=7:
                errors=np.mean(abs(pred[history,j]-y[history,None,:]),axis=(0,2))
                selected[i,j]=int(np.argmin(errors))
            elapsed.append(time.perf_counter()-tick)
    P0=pred[:,:,0,:]; P1=np.stack([pred[i,np.arange(2),selected[i]] for i in range(365)])
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'forecasts.npz',pred=pred,P0=P0,P1=P1,selected=selected)
    records=[]
    for j,y in enumerate([load,pv]):
        for m,name in enumerate(['baseline','ridge','boosting','online_selected']):
            pp=pred[31:,j,m] if m<3 else P1[31:,j]
            err=(pp-y[31:])*6
            records.append(dict(target=['load','pv'][j],model=name,mae_kw=float(abs(err).mean()),rmse_kw=float(np.sqrt((err**2).mean()))))
    pd.DataFrame(records).to_csv(OUT/'forecast_metrics.csv',index=False)
    pd.DataFrame({'day':np.arange(365),'load_model':selected[:,0],'pv_model':selected[:,1]}).to_csv(OUT/'forecast_selection.csv',index=False)
    print(json.dumps({'metrics':records,'max_predict_seconds':max(elapsed)},ensure_ascii=True),flush=True)

def run_policy(name,start=31,end=365,alpha=.8,scenarios=14):
    _,load,pv,p=read_data(); f=np.load(OUT/'forecasts.npz')
    config='P0' if name in ['M0','M1','R0'] else 'P1'
    forecast=f[config]; net=load-pv; pn=forecast[:,0]-forecast[:,1]
    E=6000.; rows=[]; days=[]
    for i in range(start,end):
        tick=time.perf_counter()
        past=np.arange(max(7,i-56),i); residual=net[past]-pn[past]
        q=np.quantile(residual,alpha,axis=0)
        if name=='R0':
            g=np.maximum(pn[i]+q,0); opt={'seconds':0.,'milp':False,'gap':0.}
        else:
            inp=pn[i]
            if name in ['M1','M3']: inp=inp+q
            if name=='M4':
                ids=np.unique(np.linspace(0,len(past)-1,min(scenarios,len(past)),dtype=int))
                inp=pn[i]+residual[ids]
            opt=optimize(inp,p,E)
            g=opt['g']
        r=execute(g,load[i],pv[i],E,terminal=6000. if i==end-1 else None,storage=name!='R0')
        day={'day':i,'date':str(pd.Timestamp('2025-01-01')+pd.Timedelta(days=i))[:10],
             'plan_cost':float(p@g),'emergency_cost':float(5*p@r['e']),
             'plan_kwh':float(g.sum()),'emergency_kwh':float(r['e'].sum()),
             'charge_kwh':float(r['c'].sum()),'discharge_kwh':float(r['d'].sum()),
             'unused_kwh':float(r['unused'].sum()),'spill_kwh':float(r['spill'].sum()),
             'initial':E,'terminal':float(r['E'][-1]),'solver_seconds':opt['seconds'],
             'seconds':time.perf_counter()-tick,'milp':opt['milp'],'gap':opt['gap']}
        day['cost']=day['plan_cost']+day['emergency_cost']; days.append(day)
        for t in range(144):
            rows.append([day['date'],t,*[float(r[key][t]) for key in ['g','c','d','e','unused','spill']],float(r['E'][t]),float(r['E'][t+1]),float(load[i,t]),float(pv[i,t]),float(p[t])])
        E=float(r['E'][-1])
    label=name if alpha==.8 and scenarios==14 else f'{name}_a{alpha}_s{scenarios}'
    folder=OUT/('validation' if start<31 else 'experiments'); folder.mkdir(exist_ok=True)
    pd.DataFrame(days).to_csv(folder/f'{label}_daily.csv',index=False)
    pd.DataFrame(rows,columns=['date','interval_id','g','c','d','e','unused','spill','E_start','E_end','load','pv','price']).to_csv(folder/f'{label}_dispatch.csv',index=False)
    print(json.dumps({'policy':label,'days':len(days),'cost':sum(d['cost'] for d in days),
        'emergency_kwh':sum(d['emergency_kwh'] for d in days),'last_E':E,
        'max_solve_seconds':max(d['solver_seconds'] for d in days)},ensure_ascii=True),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage'); p.add_argument('--start',type=int,default=31)
    p.add_argument('--end',type=int,default=365); p.add_argument('--alpha',type=float,default=.8)
    p.add_argument('--scenarios',type=int,default=14); args=p.parse_args()
    if args.stage=='predict': predict_all()
    else: run_policy(args.stage,args.start,args.end,args.alpha,args.scenarios)
