"""Reconcile v2 runs, write template payload and Chinese manuscript."""
import json,shutil,hashlib,os
from pathlib import Path
from q2_review_experiments import *
from q2_analyze import intervals

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def fmt(x): return f'{x:,.2f}'
def relative_url(path, start=None):
    """Return a POSIX relative URL so the manuscript works after moving the repo."""
    start = Path(start or ROOT / 'docs' / 'result')
    return Path(os.path.relpath(Path(path), start=start)).as_posix()

def link(title, path, start=None):
    return f'[{title}]({relative_url(path, start)})'

def figures(monthly,comp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    families = {'msyh', 'microsoftyahei', 'simhei', 'simsun', 'notosanscjk'}
    selected = None
    for candidate in font_manager.findSystemFonts(fontext='ttf'):
        if Path(candidate).stem.lower().replace(' ', '') in families:
            font_manager.fontManager.addfont(candidate)
            selected = Path(candidate).stem
            break
    plt.rcParams['font.family'] = selected or 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    fig,ax=plt.subplots(figsize=(9,4))
    for label,name in [('M1_T0','冻结M1'),('M3_T0','固定M3（事后对照）'),('Rolling_T0','月度滚动方案')]:
        z=monthly[monthly.model==label]; ax.plot(z.month.str[-2:].astype(int),z.cost/10000,'o-',label=name)
    ax.set(xlabel='2025年月',ylabel='月总费用（万元）',xticks=range(2,13)); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'monthly_cost_v2.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for t,name in [('T0','固定6000'),('T1','规划日循环')]:
        alpha=[.7,.75,.8,.85,.9]
        labels=[f'M1_{t}'+('' if a==.8 else f'_a{a}') for a in alpha]
        axes[0].plot(alpha,[comp.loc[m,'cost']/10000 for m in labels],'o-',label=name)
        axes[1].plot(alpha,[comp.loc[m,'emergency_kwh']/10000 for m in labels],'o-',label=name)
    for ax in axes: ax.set_xlabel('风险分位数'); ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('输出期总费用（万元）');axes[1].set_ylabel('紧急购电量（万kWh）')
    fig.tight_layout();fig.savefig(OUT/'terminal_risk_v2.png',dpi=200);plt.close(fig)

def main():
    protocol=json.loads((OUT/'selection_protocol.json').read_text())
    label=protocol['final_policy']; folder=OUT/'experiments'
    daily=pd.read_csv(folder/f'{label}_daily.csv'); x=pd.read_csv(folder/f'{label}_dispatch.csv')
    selected=json.loads((folder/f'{label}.json').read_text())
    audit(daily,x,Config(**selected['config']))
    daily.to_csv(OUT/'daily_summary.csv',index=False); x.to_csv(OUT/'dispatch_10min.csv',index=False)
    all_results=[json.loads(p.read_text()) for p in folder.glob('*.json') if not p.stem.startswith('trial')]
    comparison=pd.DataFrame([{k:v for k,v in r.items() if k not in ('config','checks')} for r in all_results])
    comparison.to_csv(OUT/'comparison.csv',index=False); comp=comparison.set_index('label')
    monthly=[]
    for r in all_results:
        d=pd.read_csv(folder/f"{r['label']}_daily.csv")
        for month,z in d.groupby(d.date.str[:7]):
            monthly.append(dict(model=r['label'],month=month,cost=float(z.cost.sum()),emergency_kwh=float(z.emergency_kwh.sum())))
    monthly=pd.DataFrame(monthly); monthly.to_csv(OUT/'monthly_comparison.csv',index=False)
    figures(monthly,comp)
    monthly['season']=monthly.month.str[-2:].astype(int).map(lambda m:'春' if 3<=m<=5 else '夏' if 6<=m<=8 else '秋' if 9<=m<=11 else '冬（2、12月）')
    monthly.groupby(['model','season'],sort=False)[['cost','emergency_kwh']].sum().reset_index().to_csv(OUT/'seasonal_comparison.csv',index=False)
    specified={}; plan=[]; battery=[]; emergency=[]
    for date,z in x.groupby('date',sort=True):
        pc=float(z.price@z.g); ec=float(5*z.price@z.e)
        plan.append(z.g.tolist()+[float(z.g.sum()),pc]); blocks=[]
        for k in range(6):
            a=z.iloc[k*24:(k+1)*24]
            block=[f'{k*4}:00-{(k+1)*4}:00',float(a.c.sum()),float(a.d.sum())]; blocks.append(block)
            battery.append([date if k==0 else None,*block,'0:00' if k==0 else '24:00' if k==1 else None,float(z.E_start.iloc[0]) if k==0 else float(z.E_end.iloc[-1]) if k==1 else None])
        events=intervals(z)
        emergency.extend([[date if j==0 else None,*e] for j,e in enumerate(events)] if events else [[date,'无',0.]])
        if date in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
            specified[date]=dict(specified_g=z.iloc[[60,72,84,96,108,120]].g.tolist(),plan_kwh=float(z.g.sum()),plan_cost=pc,emergency_kwh=float(z.e.sum()),emergency_cost=ec,total_cost=pc+ec,initial=float(z.E_start.iloc[0]),terminal=float(z.E_end.iloc[-1]),blocks=blocks,emergency_intervals=events)
    payload=dict(selected=label,plan=plan,battery=battery,emergency=emergency,specified=specified)
    (OUT/'workbook_payload.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf8')
    paper=ROOT/'docs/result/第二问求解与结果分析.md'
    archive=OUT/'第二问原版论文_存档.md'
    if not archive.exists(): shutil.copy2(paper,archive)
    old=archive.read_text(encoding='utf8')
    predictors=old[old.index('### 3.1'):old.index('## 4. 日前优化模型')]
    model=old[old.index('### 4.1'):old.index('## 5. 实际执行与费用结算')]
    model=model.replace('另以28个场景检查敏感性。','旧版曾以28个场景检查敏感性；本次统一采用14个场景，旧版本费用不混入本次对比。')
    refs=old[old.index('## 参考文献'):]
    jan=json.loads((OUT/'baseline_reproduction.json').read_text())
    rotation=pd.read_csv(OUT/'rolling_selection_T0.csv')
    chosen=comp.loc[label]; base=comp.loc['M1_T0']; no=comp.loc['R0_T0']
    text=r'''# 问题二：基于历史风险校准与滚动选模的日前购电策略

## 1. 问题分析与思路构建

问题二每天0:00必须冻结144个10分钟区间的计划购电量。当日负载和光伏具有不确定性，计划电量无论实际是否用完均付费，供电缺口按同时刻正常电价的5倍补购。因此，多购造成已付费电量浪费，少购则暴露于高价补购；储能又把一个时段的决策传递到后续时段，不能仅用预测误差或单槽安全裕量评价整个方案。

据此构建“历史预测—净负载风险校准—日前购电决策—实时储能反馈—按月因果选模”的分层策略。日前储能轨迹是确定购电量的辅助规划变量；实际储能根据当前观测重新反馈，最终结果表记录实际执行充放电量。日前与实时分层及不平衡费用建模参考文献[1]；概率信息应匹配优化结构而非一味增加复杂度[2]；最终以实际购电总费用评价，预测MAE/RMSE只作辅助诊断[5]。

本节进一步检验“规划日循环是否比固定6000更好”“短期选出的模型是否需要随季节调整”，并将费用、风险和物理可行性同时作为判断依据。新增试验属于已观察过2025年数据后的因果历史回测，不能称为完全未接触的最终测试集，也不宣称实现随机真实费用的全局最优。

## 2. 信息集、时间边界与统一假设

### 2.1 每个决策时刻可用的信息

每天0点的信息集由附件1给定的重复日电价、前一日及更早的实际负载和光伏、各历史时点形成的预测/残差，以及当前实际储能构成。当天实测负载和光伏只在运行到相应时段时作为反馈输入，不作为0点已知数据；第二问不使用附件3预报和附件4波动电价。月初模型选择只使用当日以前已发生的历史。

假设当前10分钟区间的平均净负载可及时观测并用于该区间控制。这是离散时间近似，未模拟区间内波动或观测延迟。1月储能实际待机6000 kWh，1月数据用于初始训练和独立影子验证；2月1日至12月31日为334天正式输出期。

### 2.2 计量与物理假设

所有优化流量均在交流母线侧计量，单位kWh；功率kW仅在输入时乘1/6。充、放效率分别0.9，储能范围1200—10800 kWh，10分钟充放电上限5000/6 kWh。不售电、不缺供，允许弃光；多余合同可拒绝利用但不退款，与弃光分别核算。输入沿用标准化数据，未再次平滑、删异常或移动时间轴；未取得清洗前原附件，不能宣称独立复核了原始映射。

实际状态跨日连续。每日规划末端与实际末端是不同概念；评价期初末统一6000只是公平库存比较约定，不是题目要求的每日循环。用于恢复库存的额外购电按5倍价格计费，这也是本项目明确采用的结算解释。

## 3. 历史预测与风险水平

'''+predictors+r'''
0.8在本方案中称为“具有经济解释的基准风险水平”。分位数在有储能全日问题中不是最优性证明；后续敏感性也允许发现其他分位数费用更低，不能把0.8附近稳定作为预定结论。P1内部按此前14天MAE选预测器，调度模型标签则按历史实际运行费用选取，这两层选择依据不同。

## 4. 日前优化、实时执行与终端处理

'''+model+r'''
### 4.3 实际执行与费用

对已冻结计划，当前可用盈余为 $x_{i,t}=g_{i,t}+V^{act}_{i,t}-L^{act}_{i,t}$。常规运行日采用F0反馈：盈余时在容量、功率限制下充电，缺口时在储能下界、功率限制下放电。记执行后的盈余 $B^{sur}_{i,t}=x_{i,t}+d^{act}_{i,t}-c^{act}_{i,t}$，则

$$
e_{i,t}=\max\{0,-B^{sur}_{i,t}\},\quad
w_{i,t}=\min\{g_{i,t},\max(B^{sur}_{i,t},0)\},\quad
s_{i,t}=\max(B^{sur}_{i,t},0)-w_{i,t}.
$$

$$
g_{i,t}-w_{i,t}+V^{act}_{i,t}-s_{i,t}+e_{i,t}+d^{act}_{i,t}=L^{act}_{i,t}+c^{act}_{i,t},\qquad
E^{act}_{i,t+1}=E^{act}_{i,t}+0.9c^{act}_{i,t}-d^{act}_{i,t}/0.9.
$$

$$
C_i=\sum_t p_tg_{i,t}+5\sum_t p_te_{i,t},\qquad C=\sum_i C_i.
$$

其中 $w$为已付费未利用合同电量，$s$为实际弃光。储能不得向外网售电或人为丢弃电量，因此还检查 $s\le V^{act}$。F0侧重即时可靠供电，未显式评价未来峰谷电价的实时边际价值；这些价值部分通过日前购电计划体现。

### 4.4 规划日循环对照与评价末端修正

基准T0设 $E^0_{i,144}=6000$。规划日循环T1设

$$
E^0_{i,0}=E^{act}_{i,0},\qquad E^0_{i,144}=E^{act}_{i,0},\qquad
E^{act}_{i,0}=E^{act}_{i-1,144}.
$$

T1不要求实际每日首尾相等，两组各自传递实际SOC，不能将T0的状态直接拿给T1使用。循环目标可减少固定库存追踪，但可能延续低库存；固定6000有库存恢复作用，也可能增加购电。两者都属于额外终端假设，优劣由实验检验。

本次短窗口复核发现，旧评价末端包络仅考虑额定放电上限，可能在实际负载较小时强制超负载放电，将电池电量误计为弃光。新闭合日规则提前管理库存：若当前储能高于目标 $E^*$，取 $c=0$，$d=\min\{5000/6,L^{act},0.9(E-E^*)\}$；若不高于目标，则沿用盈缺反馈，但将下一状态上界限制为 $\min\{E^*,E+0.9(5000/6)\}$，下界限制为 $\max\{1200,E^*-(143-t)0.9(5000/6),E-(5000/6)/0.9\}$。缺少的电量仍按5倍价格补购。到期若实际需求不足以消纳必要放电，则明确判不可达，不伪造状态或弃光。

该保守规则只用于每次影子窗口的最后一天和正式输出期12月31日，不在每月底对实际系统重置状态。它会改变末日成本，不能把新旧版本的全部差额归因于模型优化；相关成本影响单列报告。

## 5. 模型比较与最终方案确定

### 5.1 实验组的作用

'''
    text+=table(['组别','预测','风险处理','日前目标','用途'],[
        ['M0','P0历史基准','无','计划费','确定性对照'],['M1','P0','分位数修正','计划费','经济风险基准'],
        ['M2','P1在线选择','无','计划费','预测消融'],['M3','P1','分位数修正','计划费','预测＋风险对照'],
        ['M4','P1','14个历史场景，共享储能','计划费＋平均5倍补购费','受限场景对照'],['R0','P0','分位数修正','直接按修正净负载购电','无储能参照']])
    text+='### 5.2 1月预选与终端假设选择\n\n所有1月影子试验在1月15—31日连续运行，同起末6000，采用修正后的闭合日规则，按真实历史执行费用比较。\n\n'
    text+=table(['模型（T0）','1月影子总费用（元）'],[[m,fmt(v)] for m,v in jan['january_safe_model_costs'].items()])
    text+=table(['M1终端规则','1月影子总费用（元）'],[[t,fmt(v)] for t,v in protocol['january_terminal_costs'].items()])
    text+='两项比较均支持2月以M1、固定6000规划目标启动，分位数0.8保持预设。后续T2/T3或0.75的输出期费用即使更低，也不追溯替换这一初始选择。\n\n### 5.3 每月滚动选模\n\n从3月起，每月1日以之前最多56天、最早1月15日起的历史进行M1/M3/M4影子回放，选择累计总费用最小者，平局按模型名排序；选中标签固定至下月。每个候选影子窗口同起末6000，但真实系统沿用上一天实际状态，每天0点重新求解计划。历史窗口只决定模型标签，影子SOC不写入真实运行。该时间推进方式遵循滚动起点验证原则[7]。\n\n'
    text+=table(['运行月份','影子起始日','影子末日','选中模型'],[['2月','2025-01-15','2025-01-31','M1']]+[[f'{r.month}月',(pd.Timestamp('2025-01-01')+pd.Timedelta(days=r.start)).strftime('%Y-%m-%d'),(pd.Timestamp('2025-01-01')+pd.Timedelta(days=r.last_history_day)).strftime('%Y-%m-%d'),r.winner] for r in rotation.itertuples()])
    text+='**最终交付方案：固定6000规划目标（T0）＋月度历史费用滚动选择M1/M3/M4＋当前区间F0储能反馈＋评价末日物理可行库存恢复。** 这是在本次复审中固定规则后重放的因果策略，不冒充2025年之前已注册的策略；全年固定M3只是事后对照，未被反向用于2月1日选模。\n\n'
    text+='### 5.4 输出期总费用与风险\n\n'
    labels=['R0_T0','M0_T0','M1_T0','M2_T0','M3_T0','M4_T0','Rolling_T0','Rolling_T1']
    text+=table(['方案','总费用（元）','紧急购电量（kWh）','紧急天数','合并事件数'],[[m,fmt(comp.loc[m,'cost']),fmt(comp.loc[m,'emergency_kwh']),int(comp.loc[m,'emergency_days']),int(comp.loc[m,'emergency_events'])] for m in labels])
    text+=f"主方案总费用为**{fmt(chosen.cost)}元**，其中计划费{fmt(chosen.plan_cost)}元、紧急费{fmt(chosen.emergency_cost)}元。相对同闭合规则冻结M1节约{fmt(base.cost-chosen.cost)}元（{100*(base.cost-chosen.cost)/base.cost:.3f}%），相对无储能R0节约{fmt(no.cost-chosen.cost)}元（{100*(no.cost-chosen.cost)/no.cost:.2f}%）。主方案紧急购电{int(chosen.emergency_days)}天，{int(chosen.emergency_events)}个合并事件，{int(chosen.emergency_slots)}个非零10分钟槽，三者不是同一统计量。\n\n"
    text+=f"相对冻结M1，紧急购电总量减少{fmt(base.emergency_kwh-chosen.emergency_kwh)} kWh，但补购天数由{int(base.emergency_days)}增至{int(chosen.emergency_days)}，事件数由{int(base.emergency_events)}增至{int(chosen.emergency_events)}。本方案以总费用为首要指标，并非所有风险指标同时改善。\n\n![月度费用对比]({relative_url(OUT/'monthly_cost_v2.png')})\n\n"
    text+=f"固定M3费用{fmt(comp.loc['M3_T0','cost'])}元，仍比滚动策略低{fmt(chosen.cost-comp.loc['M3_T0','cost'])}元。这表明滚动选择解决了使用历史动态选模的可实施性，但并未取得事后最优候选的全部收益；不能据此宣称滚动策略一定优于所有固定策略。\n\n"
    text+='## 6. 终端与参数敏感性\n\n### 6.1 固定6000与规划日循环\n\n'
    text+=table(['规则','总费用（元）','紧急电量（kWh）','低于2400日初SOC天数'],[[t,fmt(comp.loc['M1_'+t,'cost']),fmt(comp.loc['M1_'+t,'emergency_kwh']),int(comp.loc['M1_'+t,'low_initial_days'])] for t in ('T0','T1','T2','T3','T4')])
    text+=f"T1比T0多支出{fmt(comp.loc['M1_T1','cost']-base.cost)}元（{100*(comp.loc['M1_T1','cost']/base.cost-1):.2f}%），紧急购电增加{fmt(comp.loc['M1_T1','emergency_kwh']-base.emergency_kwh)} kWh。本数据下，规划日循环没有改善总成本，故未替换T0。2400仅是低库存统计阈值，不是新增控制约束。T2/T3/T4只作事后敏感性，不据此追选正式终端。\n\n"
    text+=table(['月份','T1减T0费用（元）'],[[mo,fmt(float(monthly[(monthly.model=='M1_T1')&(monthly.month==mo)].cost.iloc[0]-monthly[(monthly.model=='M1_T0')&(monthly.month==mo)].cost.iloc[0]))] for mo in sorted(monthly.month.unique())])
    text+='### 6.2 末端库存与版本差异\n\n'
    text+=table(['M1配置','输出期费用（元）','末端库存（kWh）'],[[m,fmt(comp.loc[m,'cost']),fmt(comp.loc[m,'last_E'])] for m in ('M1_T0','M1_T0_open','M1_T1','M1_T1_open')])
    text+=f"保守闭合日规则使T0费用相对旧规则增加{fmt(jan['baseline_safe_cost']-jan['baseline_legacy_cost'])}元，差异来自12月31日执行；此前计划已验证一致。旧M1费用{fmt(jan['baseline_legacy_cost'])}元与新方案采用不同闭合策略，不能把两者差额全部称为滚动选模收益。无闭合组库存不同，只报告费用与库存，不直接宣称节约。\n\n"
    text+='### 6.3 风险水平、初值与误差压力\n\n'
    text+=table(['分位数','T0费用（元）','T1费用（元）','T0紧急电量（kWh）'],[[str(a),fmt(comp.loc['M1_T0'+('' if a==.8 else f'_a{a}'),'cost']),fmt(comp.loc['M1_T1'+('' if a==.8 else f'_a{a}'),'cost']),fmt(comp.loc['M1_T0'+('' if a==.8 else f'_a{a}'),'emergency_kwh'])] for a in (.7,.75,.8,.85,.9)])
    text+='本次0.75的事后费用低于0.8；提高风险分位数减少补购，但增加付费未利用电量。费用曲线较平缓不能解释为风险电量也稳定；0.8为预设经济基准，不以这张输出期表反向调参。\n\n'
    text+=f"![终端与风险参数敏感性]({relative_url(OUT/'terminal_risk_v2.png')})\n\n"
    stress=[m for m in comp.index if '_initial' in m or '_scale' in m]
    text+=table(['压力配置','费用（元）','紧急电量（kWh）','期末SOC（kWh）'],[[m,fmt(comp.loc[m,'cost']),fmt(comp.loc[m,'emergency_kwh']),fmt(comp.loc[m,'last_E'])] for m in sorted(stress)])
    text+='初值3600/8400组均恢复各自初值，只能同初值配对比较，正式初值仍为6000。误差压力为P0预测加原历史误差的0.75或1.25倍，并对负载、光伏非负截断；预测缓存保持固定，只有此前合成实际进入残差校准。它是合成扰动下的离线稳健性诊断，不是额外真实观测。\n\n'
    trial=json.loads((folder/'trial_MPC.json').read_text()); trial0=json.loads((folder/'trial_T0.json').read_text())
    text+='### 6.4 实时MPC与场景增强的取舍\n\n'
    text+=f"F1每个时段固定已付费计划、用当前实测和未来预测最小化剩余日内5倍补购费用，并只执行首个动作；两槽测试验证了为高价缺口保留储能的行为。2月1—7日共同闭合规则下，F0费用{fmt(trial0['cost'])}元，F1费用{fmt(trial['cost'])}元，F1增加{100*(trial['cost']/trial0['cost']-1):.2f}%，最大单日耗时{trial['max_seconds']:.3f}秒。短窗口未显示收益，按试验门槛停止扩大；未据此宣称所有MPC均不适用。F1不改变日前计划，允许按5倍价格补购用于储能，但仍不得售电或丢弃电池能量。\n\n"
    text+='完整场景自适应储能已形成节点变量、非预见性与树外执行的设计说明；当前未实现和验证完整场景树，故不进入正式结果。M4已具有场景专属补购变量，可解释为受限两阶段补救模型，不能笼统说它不是随机规划；其储能动作共享，不能代替具有场景节点自适应储能的多阶段模型。\n\n'
    text+='MPC试验的最后一天与F0采用同一库存闭合反馈，其余六天才使用剩余日MPC；两组每日购电均从自身实际日初状态重新计算，不能视为整周购电量逐槽完全相同的实验。\n\n'
    text+='## 7. 四个指定日期的具体策略\n\n购电为0点冻结计划，充放电及SOC为实际执行值。展示保留两位小数，附件保存完整计算精度。\n\n'
    for date,s in specified.items():
        text+=f'### {date}\n\n'
        text+=table(['时间段','计划购电量（kWh）'],[[t,fmt(q)] for t,q in zip(['10:00—10:10','12:00—12:10','14:00—14:10','16:00—16:10','18:00—18:10','20:00—20:10'],s['specified_g'])])
        text+=table(['指标','数值'],[[k,fmt(s[v])] for k,v in [('全天计划购电量（kWh）','plan_kwh'),('全天计划购电费（元）','plan_cost'),('紧急购电量（kWh）','emergency_kwh'),('紧急购电费（元）','emergency_cost'),('全天总费用（元）','total_cost'),('0:00储电量（kWh）','initial'),('24:00储电量（kWh）','terminal')]])
        text+=table(['时间段','实际充电量（kWh）','实际放电量（kWh）'],[[b[0],fmt(b[1]),fmt(b[2])] for b in s['blocks']])
    text+='### 指定日紧急购电汇总（对应表3）\n\n'
    er=[]
    for date,s in specified.items(): er.extend([[date,*[event[0],fmt(event[1])]] for event in s['emergency_intervals']] or [[date,'无','0.00']])
    text+=table(['日期','紧急购电区间','购电量（kWh）'],er)
    dates=[d for d,s in specified.items() if s['emergency_kwh']>EPS]
    text+=f"四个指定日中，发生紧急购电的日期为：{'、'.join(dates) if dates else '无'}；其余指定日为0。相邻非零10分钟槽合并且不跨日，统计阈值为1e-7 kWh。此结论由新方案数据自动生成，不沿用旧方案的指定日结论。\n\n"
    text+='## 8. 验证、适用边界与附件\n\n'
    text+=f"主方案48,096条10分钟轨迹已检查母线平衡、SOC递推、功率及状态边界、非负、充放电互斥、实际跨日连续和期末库存。该组日前转MILP天数为{int(chosen.milp_days)}，LP最大原始—对偶绝对差为{chosen.max_lp_absolute_gap_yuan:.3g}元；不将这一证据解释为真实随机总费用最优。导出Excel另由独立复核器读取全部计划、334天储能及紧急事件，并对每一天重建0点计划，详见交付验证附件。\n\n"
    text+='局限包括一年样本、1月初选季节单一、已进行历史探索、当前槽平均值可及时观测的近似、固定目标和保守闭合日策略、简单实时反馈，以及M4未包含自适应储能。没有宣称完整随机规划/MPC已实现全年提升，也没有把更低的事后结果当作事前可知选择。\n\n'
    for title,filename in [('正式结果result2.xlsx','result2.xlsx'),('完整10分钟执行账本','dispatch_10min.csv'),('每日费用与规划/实际SOC','daily_summary.csv'),('模型与敏感性对比','comparison.csv'),('逐月比较','monthly_comparison.csv'),('分季比较','seasonal_comparison.csv'),('事前信息约束与最终选择协议','selection_protocol.json'),('月度选择依据','rolling_selection_T0.csv'),('旧版重现及闭合规则差异','baseline_reproduction.json'),('导出Excel独立交付验证','delivery_verification.json'),('输入及代码哈希','source_hashes.json')]: text+='- '+link(title,OUT/filename)+'\n'
    text+='\n'+link('优化文献核查',ROOT/'docs/第二问优化文献核查.md')+'；'+link('场景自适应储能设计及门槛',ROOT/'docs/第二问场景自适应模型说明.md')+'。\n\n'+refs
    paper.write_text(text,encoding='utf8')
    copy_text = text
    linked_targets = [OUT / name for name in (
        'monthly_cost_v2.png', 'terminal_risk_v2.png', 'result2.xlsx',
        'dispatch_10min.csv', 'daily_summary.csv', 'comparison.csv',
        'monthly_comparison.csv', 'seasonal_comparison.csv',
        'selection_protocol.json', 'rolling_selection_T0.csv',
        'baseline_reproduction.json', 'delivery_verification.json',
        'source_hashes.json')]
    linked_targets.extend([
        ROOT / 'docs' / '第二问优化文献核查.md',
        ROOT / 'docs' / '第二问场景自适应模型说明.md',
    ])
    for target in linked_targets:
        copy_text = copy_text.replace(
            f']({relative_url(target, paper.parent)})',
            f']({relative_url(target, OUT)})')
    (OUT/'第二问求解与结果分析.md').write_text(copy_text,encoding='utf8')
    print(json.dumps(dict(selected=label,cost=float(chosen.cost),specified=specified),ensure_ascii=True),flush=True)

if __name__=='__main__': main()
