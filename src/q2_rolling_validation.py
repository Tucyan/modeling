"""Monthly causal policy selection; shadow states never replace actual states."""
import json
from dataclasses import replace
from q2_review_experiments import OUT,Config,load_inputs,simulate,audit,save_run,history_window,np,pd

def choose_from_scores(scores,last_history_day,decision_day):
    if last_history_day>=decision_day: raise ValueError('Selection contains unavailable observations')
    return min(scores,key=lambda name:(scores[name],name))

def main():
    inputs=load_inputs(); OUT.mkdir(parents=True,exist_ok=True)
    initial_scores={}
    for t in ('T0','T1'):
        config=Config(terminal=t)
        d,x=simulate(config,14,31,inputs); audit(d,x,config)
        initial_scores[t]=float(d.cost.sum())
        d.to_csv(OUT/f'january_{t}_daily.csv',index=False)
    selected_terminal=choose_from_scores(initial_scores,30,31)
    # Frozen protocol: 0.8, 56-day trailing shadow costs, M1/M3/M4; no annual tuning.
    selection=dict(terminal=selected_terminal,january_terminal_costs=initial_scores,
        alpha=.8,initial_month_model='M1',candidates=['M1','M3','M4'],
        lookback_days=56,first_history_day=14,closure='safe',
        final_policy=f'Rolling_{selected_terminal}',
        interpretation='Retrospectively designed causal backtest; not untouched holdout')
    (OUT/'selection_protocol.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
    print(json.dumps(selection),flush=True)
    for terminal in ('T0','T1'):
        choices={i:'M1' for i in range(31,365)}; records=[]
        for month in range(3,13):
            decision=int((pd.Timestamp(2025,month,1)-pd.Timestamp(2025,1,1)).days)
            start,end=history_window(decision); scores={}
            for model in ('M1','M3','M4'):
                config=Config(model=model,terminal=terminal)
                d,x=simulate(config,start,end,inputs); audit(d,x,config)
                scores[model]=float(d.cost.sum())
            winner=choose_from_scores(scores,end-1,decision)
            stop=365 if month==12 else int((pd.Timestamp(2025,month+1,1)-pd.Timestamp(2025,1,1)).days)
            choices.update({i:winner for i in range(decision,stop)})
            records.append(dict(month=month,decision_day=decision,start=start,end_exclusive=end,
                last_history_day=end-1,winner=winner,**scores))
            print(json.dumps(dict(terminal=terminal,month=month,winner=winner)),flush=True)
        pd.DataFrame(records).to_csv(OUT/f'rolling_selection_{terminal}.csv',index=False)
        save_run(f'Rolling_{terminal}',Config(terminal=terminal),inputs=inputs,choices=choices)

if __name__=='__main__':main()
