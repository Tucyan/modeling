"""Isolated Q2 revision: continuous shadow states and an open evaluation end."""
import json
from dataclasses import asdict
from q2_review_experiments import (ROOT, OUT as V2, Config, load_inputs, simulate,
    audit, summarize, planning_input, optimize, execute, np, pd)

OUT = ROOT / 'outputs/q2_incremental'

def select_month(frames, decision):
    start = max(14, decision - 56)
    scores = {}
    for model, frame in frames.items():
        past = frame.loc[(frame.day >= start) & (frame.day < decision)]
        if sorted(past.day.tolist()) != list(range(start, decision)):
            raise ValueError('Incomplete or duplicated historical window')
        scores[model] = float(past.cost.sum())
    return dict(decision_day=decision, start=start, end_exclusive=decision,
                winner=min(scores, key=lambda m:(scores[m],m)), **scores)

def calibration_metrics(actual, predicted, alpha):
    residual = np.asarray(actual) - np.asarray(predicted)
    return float(np.mean(residual <= 0)), float(np.mean(np.maximum(alpha*residual,(alpha-1)*residual)))

def save(label, config, start, inputs, choices=None):
    daily, detail = simulate(config, start, 365, inputs, choices)
    checks = audit(daily, detail, config)
    daily.to_csv(OUT/f'{label}_daily.csv',index=False)
    detail.to_csv(OUT/f'{label}_dispatch.csv',index=False)
    result = dict(label=label,config=asdict(config),checks=checks,**summarize(daily,detail))
    (OUT/f'{label}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(label=label,cost=result['cost'],last_E=result['last_E'])),flush=True)
    return daily,detail,result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=load_inputs()
    shadows={}
    for model in ('M1','M3','M4'):
        shadows[model],_,_=save('shadow_'+model,Config(model=model,close=False),14,inputs)
    records=[]; choices={}; oldchoices={i:'M1' for i in range(31,365)}
    for row in pd.read_csv(V2/'rolling_selection_T0.csv').itertuples():
        stop=365 if row.month==12 else (pd.Timestamp(2025,row.month+1,1)-pd.Timestamp(2025,1,1)).days
        oldchoices.update({i:row.winner for i in range(row.decision_day,stop)})
    for month in range(2,13):
        day=(pd.Timestamp(2025,month,1)-pd.Timestamp(2025,1,1)).days
        stop=365 if month==12 else (pd.Timestamp(2025,month+1,1)-pd.Timestamp(2025,1,1)).days
        row=select_month(shadows,day); row['month']=month
        choices.update({i:row['winner'] for i in range(day,stop)})
        records.append(row)
    protocol=dict(planning_terminal='T0:6000',actual_terminal='open',controller='F0',alpha=.8,
        initial=6000,shadow_start_day=14,actual_start_day=31,lookback=56,
        selection='Continuous candidate histories; February also selected from prior history',
        interpretation='Retrospectively designed causal backtest, not an untouched holdout')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    oldd,oldx,oldr=save('Rolling_v2_open',Config(close=False),31,inputs,oldchoices)
    daily,detail,result=save('Rolling_continuous_open',Config(close=False),31,inputs,choices)
    # Paired baseline: removing closure cannot change any preceding day.
    baseline=pd.read_csv(V2/'dispatch_10min.csv')
    cols=['g','c','d','e','E_start','E_end']
    prefix=float(abs(baseline[cols].iloc[:-144].to_numpy()-oldx[cols].iloc[:-144].to_numpy()).max())
    assert prefix<1e-6,prefix
    for row in records:
        day=row['decision_day']; start=row['start']
        row['actual_initial']=float(daily.loc[daily.day==day,'initial'].iloc[0])
        for model,frame in shadows.items():
            row[model+'_initial']=float(frame.loc[frame.day==day,'initial'].iloc[0])
            row[model+'_window_initial']=float(frame.loc[frame.day==start,'initial'].iloc[0])
            row[model+'_window_terminal']=float(frame.loc[frame.day==day-1,'terminal'].iloc[0])
    pd.DataFrame(records).to_csv(OUT/'monthly_selection.csv',index=False)
    # Independent reconstruction of all actual plans and feedback, not shadow cost splicing.
    l,v,p,forecasts=inputs; max_error=0.
    for row in daily.itertuples():
        forecast=forecasts['P0' if row.model=='M1' else 'P1']
        net=planning_input(l-v,forecast[:,0]-forecast[:,1],row.day,row.model,.8,14)
        opt=optimize(net,p,row.initial,target=6000.)
        run=execute(opt['g'],l[row.day],v[row.day],row.initial,terminal=None)
        z=detail.loc[detail.date==row.date]
        for key in ('g','c','d','e'):
            max_error=max(max_error,float(abs(run[key]-z[key].to_numpy()).max()))
    assert max_error<1e-6,max_error
    cal=[]
    dates=pd.date_range('2025-02-01','2025-12-31')
    seasons=np.array(['春' if 3<=d.month<=5 else '夏' if 6<=d.month<=8 else '秋' if 9<=d.month<=11 else '冬(2、12月)' for d in dates])
    for name,forecast in forecasts.items():
        pn=forecast[:,0]-forecast[:,1]
        q=np.array([planning_input(l-v,pn,i,'M1',.8,14) for i in range(31,365)])
        for season in ['总体','春','夏','秋','冬(2、12月)']:
            mask=np.ones(334,dtype=bool) if season=='总体' else seasons==season
            coverage,loss=calibration_metrics((l-v)[31:][mask],q[mask],.8)
            cal.append(dict(forecast=name,season=season,slots=int(mask.sum()*144),coverage=coverage,
                coverage_minus_target=coverage-.8,pinball_kwh=loss))
    pd.DataFrame(cal).to_csv(OUT/'quantile_calibration.csv',index=False)
    legacy=json.loads((V2/'experiments/Rolling_T0.json').read_text(encoding='utf8'))
    compare=pd.DataFrame([dict(label='Rolling_v2_closed',**{k:legacy[k] for k in ('cost','emergency_kwh','last_E')}),
        {k:oldr[k] for k in ('label','cost','emergency_kwh','last_E')},
        {k:result[k] for k in ('label','cost','emergency_kwh','last_E')}])
    compare.to_csv(OUT/'comparison.csv',index=False)
    verification=dict(days=334,slots=len(detail),prefix_error_kwh=prefix,
        reconstruction_error_kwh=max_error,physical=result['checks'],cost=result['cost'])
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf8')
    write_report(compare,pd.DataFrame(records),pd.DataFrame(cal),result)

def write_report(compare, selections, cal, result):
    def table(headers,rows):
        return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
    text='''# 第二问小幅修订：连续影子选模与开放评价末端

本补充对应独立目录outputs/q2_incremental，原v2论文、Excel及Q3接口保留。仅调整评价末端和历史选模环境，日前T0目标6000、0.8分位数、F0反馈、效率及容量约束保持一致。

## 思路构建与边界

题面给定的是1月1日初始6000 kWh，未要求年末恢复。新方案正常运行到12月31日，报告期末库存，不人为按5倍价补库存。T0每日规划末端6000仍是额外规划假设，并不要求实际每日或年末恢复；此前T0/T1对比仍作为该假设的实验依据。

M1/M3/M4影子系统均从1月15日6000 kWh连续运行，取消影子窗口末日特殊动作。月初只累加此前最多56天的真实历史执行费用，2月也使用此前1月历史选择；实际系统沿用1月待机后2月1日6000的既有初始化。实际月初继承自身SOC，重新计算计划，影子SOC和费用均不拼入实际账本。

连续影子历史消除了反复重置，却不保证与实际切换时SOC相同。窗口内各候选的初末库存也不同，原始历史费用排名可能受库存消耗影响；monthly_selection.csv同时列出窗口初末SOC和切换日影子/实际SOC。此处是历史策略选择启发式，不声称同状态最优动作选择或严格随机动态最优。初始1月实际待机、F0贪心和预测MAE筛选仍是现有限制。

## 费用消融与最终取舍

'''
    text+=table(['方案','费用/元','紧急电量/kWh','末端库存/kWh'],[[r.label,f'{r.cost:,.2f}',f'{r.emergency_kwh:,.2f}',f'{r.last_E:,.2f}'] for r in compare.itertuples()])
    a,b,c=compare.cost.tolist()
    text+=f'仅取消年末闭合的费用变化为{b-a:,.2f}元；在无闭合口径下改用连续影子选模的费用变化为{c-b:,.2f}元。第一项包含边界规则改变，不称为预测或调度效率收益；第二项也伴随库存差异，不能解释为等库存的确定节约。\n\n'
    text+='本轮推荐采用“连续影子选模＋实际开放末端”作为Q2修订方案，因为其评价规则更贴合题意，且消除了影子窗口反复重置。该规则在本次回测前确定，不按全年最低费用反向改选候选或分位数。本数据上连续影子与v2的月度模型标签完全相同，无闭合费用亦相同，因此连续影子修改的价值是纠正评价环境，未带来额外降费。正式v2接口暂保留以避免影响同步Q3；本补充及独立账本是修订方案的可复核依据。\n\n'
    text+=table(['月份','选中模型','实际月初SOC'],[[r.month,r.winner,f'{r.actual_initial:,.2f}'] for r in selections.itertuples()])
    text+='## 分位数校准诊断\n\n对净负载逐槽计算经验覆盖率P(实际≤预测分位数)及Pinball Loss；单位为每10分钟槽kWh。分位数只由当日以前56天残差构造，季节覆盖率是事后诊断，不参与回调参数。目标0.8源于单期无储能的经济解释，在本方案不是全日最优或联合可靠性保证。\n\n'
    text+=table(['预测','季节','槽数','覆盖率','Pinball/kWh'],[[r.forecast,r.season,r.slots,f'{r.coverage:.2%}',f'{r.pinball_kwh:.4f}'] for r in cal.itertuples()])
    text+='总体覆盖率P0为77.93%、P1为78.81%，均略低于80%目标；春季分别为74.05%、76.30%，存在更明显低覆盖。故将本方法称为历史残差分位数修正，不能声称已达到稳定80%覆盖。P1总体Pinball较低，但仍不能直接推出储能购电费最优。本轮仅补诊断，不用这张全年统计表反向调高春季参数。\n\n'
    text+='## 复现、附件与引用\n\n从仓库根目录运行`.venv/Scripts/python.exe src/q2_incremental.py`。检查全部334天能量、SOC、功率、互斥、费用以及历史信息下重建计划和执行，证据见附件。\n\n'
    for name in ('protocol.json','comparison.csv','monthly_selection.csv','quantile_calibration.csv','Rolling_continuous_open_daily.csv','Rolling_continuous_open_dispatch.csv','verification.json'):
        text+=f'- [{name}](../../outputs/q2_incremental/{name})\n'
    text+='\n方法引用沿用已核查的[Q2文献说明](../第二问优化文献核查.md)和[v2论文参考文献](第二问求解与结果分析.md#参考文献)：Hyndman与Athanasopoulos《Forecasting: Principles and Practice》第3版§5.10支持时间顺序验证；Gneiting、Lerch、Schulz（Solar Energy, 2023, 252:72–80）支持概率预测校准与评分；Beykirch、Janke、Steinke（PMAPS 2022）支持储能时间耦合的解释。它们不构成本题连续影子方案最优性的证明。\n'
    (ROOT/'docs/result/第二问小幅修订与校准补充.md').write_text(text,encoding='utf8')

if __name__=='__main__': main()
