"""Baseline reproduction and independent review evidence."""
import json
from q2_review_experiments import *

def main():
    inputs=load_inputs(); config=Config(closure='legacy')
    d,x=simulate(config,inputs=inputs); checks=audit(d,x,config)
    old=pd.read_csv(OLD/'dispatch_10min.csv')
    errors={k:float(abs(x[k]-old[k]).max()) for k in ('g','c','d','e','E_start','E_end')}
    assert max(errors.values())<1e-6,errors
    # New safe closure must only change actual execution on the closing day.
    new=pd.read_csv(OUT/'experiments/M1_T0_dispatch.csv')
    prefix=float(abs(new.loc[:len(new)-145,'g']-x.loc[:len(x)-145,'g']).max())
    assert prefix<1e-6
    jan={}
    for m in ('M0','M1','M2','M3','M4'):
        c=Config(model=m); jd,jx=simulate(c,14,31,inputs); audit(jd,jx,c)
        jan[m]=float(jd.cost.sum())
    result=dict(legacy_reproduction_max_errors=errors,legacy_checks=checks,
        safe_prefix_purchase_error=prefix,january_safe_model_costs=jan,
        baseline_legacy_cost=float(d.cost.sum()),baseline_safe_cost=float((new.price*new.g+5*new.price*new.e).sum()))
    (OUT/'baseline_reproduction.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
