"""Generate Q2 manuscript from independently reconciled experiment outputs."""
import sys,runpy
from pathlib import Path
if __name__=='__main__' and '--legacy' not in sys.argv and (Path(__file__).resolve().parents[1]/'outputs/q2_review_v2/selection_protocol.json').exists():
    runpy.run_path(str(Path(__file__).with_name('q2_review_report.py')),run_name='__main__')
    raise SystemExit(0)
from q2_solver import ROOT, OUT
import json, math, os
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
families = {'msyh', 'microsoftyahei', 'simhei', 'simsun', 'notosanscjk'}
selected_font = None
for candidate in font_manager.findSystemFonts(fontext='ttf'):
    if Path(candidate).stem.lower().replace(' ', '') in families:
        font_manager.fontManager.addfont(candidate)
        selected_font = Path(candidate).stem
        break
plt.rcParams['font.family'] = selected_font or 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)+'\n'
def fmt(v): return f'{v:,.2f}'

sel=json.loads((OUT/'selection.json').read_text(encoding='utf8')); name=sel['selected']
payload=json.loads((OUT/'workbook_payload.json').read_text(encoding='utf8'))
comp=pd.read_csv(OUT/'comparison.csv').set_index('model')
daily=pd.read_csv(OUT/'daily_summary.csv'); detail=pd.read_csv(OUT/'dispatch_10min.csv')
forecast=pd.read_csv(OUT/'forecast_metrics.csv')
month=pd.read_csv(OUT/'monthly_comparison.csv')
checks=json.loads((OUT/'verification.json').read_text(encoding='utf8'))['physical'][name]
qcal=pd.read_csv(OUT/'quantile_calibration.csv')
bound=pd.read_csv(OUT/'perfect_information_monthly_bound.csv').bound.sum()
chosen=comp.loc[name]; nos=comp.loc['R0']; point=comp.loc['M0']
paper=ROOT/'docs/result/第二问求解与结果分析.md'
def relative_url(path, start=None):
    start = Path(start or paper.parent)
    return Path(os.path.relpath(Path(path), start=start)).as_posix()
base=relative_url(ROOT)
out=relative_url(OUT)

fig,ax=plt.subplots(figsize=(10,4.5))
for m in ['R0','M0','M1','M3','M4']:
    z=month[month.model==m];ax.plot(z.month.str[-2:].astype(int),z.cost/10000,'o-',label=m)
ax.set(xlabel='2025年月',ylabel='月度实际总购电费（万元）',title='同一信息与执行口径下的月度费用')
ax.set_xticks(range(2,13)); ax.legend(ncol=5);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'monthly_cost.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(4,2,figsize=(12,12))
for j,date in enumerate(payload['specified']):
    x=detail[detail.date==date];h=np.arange(144)/6
    axes[j,0].plot(h,(x.load-x.pv)*6,label='实际净负载',lw=1)
    axes[j,0].plot(h,x.g*6,label='计划购电',lw=1)
    axes[j,0].plot(h,x.e*6,label='紧急购电',lw=1)
    axes[j,0].set(title=date,ylabel='功率（kW）')
    axes[j,1].plot(np.arange(145)/6,np.r_[x.E_start.iloc[0],x.E_end],color='#427755')
    axes[j,1].axhline(1200,color='gray',ls='--');axes[j,1].axhline(10800,color='gray',ls='--')
    axes[j,1].set(title=date+' 储能状态',ylabel='储电量（kWh）',ylim=(600,11400))
    for a in axes[j]: a.set_xlim(0,24);a.grid(alpha=.2);a.set_xlabel('小时')
axes[0,0].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'specified_days.png',dpi=180);plt.close(fig)

text=r"""# 问题二：预测误差下的日前购电与储能调控

## 1. 问题分析与建模思路

问题二要求在每天0:00确定未来144个10分钟区间的购电量。与问题一不同，当天负载与光伏尚未发生，实际供电不足时需按当时正常价格的5倍紧急购电；正常购电费始终按计划量结算。因此，本问不能将附件2当天实测曲线作为日前已知输入，也不能直接逐日重复问题一的完美预测调度。

本文将问题分为“历史预测—误差校准—日前储能优化—实时反馈与费用核算”四个部分。预测负责给出未来负载与光伏，调度模型利用电价差和储能约束安排购电，误差校准降低低估净负载引起的紧急购电，实际执行器在观测当前区间净负载后调节储能。

这一结构参考日前储能计划与不平衡成本研究[1]及日前/实时分层控制[6]。考虑到预测误差与最终运行成本并不完全一致[5]，本文以实际总购电费为主要选型依据，预测精度为辅助依据。

## 2. 数据、信息边界与基本假设

采用已标准化附件1电价与附件2负载、光伏数据，全年365天共52560条，每天144条。只读核查未发现日期—时段重复或负载、光伏缺失，未再平滑或删除实际值。当前没有清洗前原附件，因此不宣称独立复核了原始数据到标准化数据的全部映射。

1. 每个区间为10分钟，功率kW乘1/6得到电量kWh。第二问只使用附件1重复日电价，不使用附件3光伏预报或附件4波动电价。
2. 充、放电均在交流母线侧计量；单向效率分别0.9，往返效率0.81。储能上下界1200、10800 kWh，单槽充放电上限5000/6 kWh。
3. 不售电、不缺供，允许弃光。多余合同电量可拒绝实际利用，但仍支付全部计划费用；它与弃光单独核算。这是对计划收费规则的运行解释。
4. 假设区间内控制足够快，可依据当前10分钟平均净负载进行储能调节；控制器不读取之后区间的实测值。这是离散近似，未模拟区间内瞬时波动及观测延迟。
5. 1月实际储能待机保持题设初值6000 kWh，1月历史数据用于训练与影子验证。2月1日从6000开始连续执行。1月影子验证是离线比较，不与实际储能轨迹混淆。
6. 每日优化设定计划末端目标6000 kWh，防止短视地耗空储能；实际日末不强制等于该值，而是连续传递。评价期末借助可达状态包络恢复6000，相关紧急购电全部计费。该终端处理是公平评价约定，不是把问题一日循环条件套用到问题二。

## 3. 预测模型与误差构造

### 3.1 强历史基准

负载采用上周同槽，光伏采用过去7天同槽均值：

$$
\widehat L_{i,t}=L^{\mathrm{act}}_{i-7,t},\qquad
\widehat V_{i,t}=\frac17\sum_{j=1}^{7}V^{\mathrm{act}}_{i-j,t},\qquad
\widehat N_{i,t}=\widehat L_{i,t}-\widehat V_{i,t}.
$$

采用周期基准的理由是既有数据呈现明显周周期；文献[3]支持用周期和日历效应建模，但具体基准的有效性由本题历史数据验证。

### 3.2 改进候选及无未来信息的在线选择

在基准上拟合残差，比较Ridge回归和直方图梯度提升树。输入共11项：基准值、前一日、前一周、过去7天/14天/3天同槽均值、日内正余弦、星期编码及年度正余弦。所有历史统计均截止前一日。年度正余弦只是平滑时间特征，不表示已学到多年的季节规律。

每7天重训一次，最多使用此前56天；Ridge在训练集内标准化，惩罚参数100；树模型40轮、最多7叶、学习率0.08、L2系数10、最多64箱，关闭随机早停，种子2025。参数在本轮预先固定，并未开展大规模网格搜索。Ridge属于正则化回归候选，不声称复现文献[3]的LASSO。

改进配置P1分别对负载、光伏依据过去14天样本外MAE选择基准、回归或树模型；历史不足时回退基准。非负预测截断为零，光伏某槽过去28天一直为零时将预测设为零，不使用目标日实际值识别夜间。

### 3.3 净负载残差分位数

保存当时发布的历史预测，并计算样本外误差：

$$
\varepsilon_{j,t}=N^{\mathrm{act}}_{j,t}-\widehat N_{j,t},\qquad
\widetilde N_{i,t}=\widehat N_{i,t}+Q_\alpha\{\varepsilon_{j,t}:j<i,\ j\text{位于最近56天}\}.
$$

误差最早从1月8日开始积累。本文主参数取$\alpha=0.8$，并做0.7、0.9敏感性分析。负载与光伏误差先合成净负载误差，避免将两者的独立上界简单相加。

在单时段、无储能且多购不退款的特例中，额外购入1 kWh必付正常价格，只有需求超出计划时才能避免5倍补购，因此内部最优超额概率为1/5，给出80%分位数启发。但有储能时边际价值受跨时段约束影响，80%既不是全日最优性定理，也不是全天无紧急购电的可靠性保证。概率预测需另行检查评分与校准[4]。

## 4. 日前优化模型

### 4.1 风险修正LP/MILP

当天0点已知实际初始储能，求解：

$$
\min\ \sum_{t=0}^{143}p_tg_{i,t},
$$

$$
g_{i,t}+d^0_{i,t}-c^0_{i,t}-u^0_{i,t}=\widetilde N_{i,t},
\qquad
E^0_{i,t+1}=E^0_{i,t}+0.9c^0_{i,t}-d^0_{i,t}/0.9,
$$

$$
1200\le E^0_{i,t}\le10800,\quad
0\le c^0_{i,t}\le(5000/6)z_{i,t},\quad
0\le d^0_{i,t}\le(5000/6)(1-z_{i,t}),
$$

$$
g_{i,t},u^0_{i,t}\ge0,\quad z_{i,t}\in\{0,1\},\quad
E^0_{i,0}=E^{\mathrm{act}}_{i,0},\quad E^0_{i,144}=6000.
$$

其中$u^0$是净负载等效模型的预测剩余电量，不当作实际弃光；$c^0,d^0,E^0$是规划值，输出附件中的充放电则为实际执行值。确定性对照用$\widehat N$替换$\widetilde N$。

数值实现添加每kWh充放电$10^{-7}$元的择优项，仅用于抑制无意义吞吐，不计入实际购电费用，也不解释为电池退化成本。先求LP松弛；若出现同时充放电，则转完整MILP，时间上限30秒、相对间隙阈值$10^{-8}$。每个解再次检查互斥和物理残差，不能将问题一的LP结论无条件推广。

### 4.2 受限场景对照

M4由最近56天中均匀抽取最多14个完整残差日轨迹，叠加当前点预测；场景等权，日前购电、计划充放电和储能状态在各场景共享，紧急补购及剩余电量按场景区分：

$$
\min\ \sum_t p_tg_{i,t}
 +\frac1{|\Omega_i|}\sum_{\omega\in\Omega_i}\sum_t5p_te^\omega_{i,t},
\qquad
g_{i,t}+d^0_{i,t}-c^0_{i,t}+e^\omega_{i,t}-u^\omega_{i,t}
=N^\omega_{i,t}.
$$

其余储能约束同上，$e^\omega,u^\omega\ge0$。另以28个场景检查敏感性。

这是共享控制的受限随机规划，不是完整多阶段自适应随机规划。虽然输入保存整日轨迹，但共享控制下，目标只利用各时段经验分布，不能据此证明已充分利用时间联合相关性，也不能将其表现视为对所有随机规划方法的否定。文献[1,2]支持更完整的概率/场景建模，本文仅实现其中计算量较小的对照。

## 5. 实际执行与费用结算

### 5.1 因果储能反馈

计划量$g$冻结后，当前区间观测形成：

$$
x_{i,t}=g_{i,t}+V^{\mathrm{act}}_{i,t}-L^{\mathrm{act}}_{i,t}.
$$

若$x\ge0$，在充电功率及剩余容量允许范围内充电；若$x<0$，在放电功率及剩余电量允许范围内放电。剩余缺口以紧急购电弥补。本文所有候选共用这一反馈器，隔离日前模型的影响。它是可实施的局部规则，但不保证全局最优实时控制。

令充放电后的母线盈余为$b=x+d^{\mathrm{act}}-c^{\mathrm{act}}$，则：

$$
e_{i,t}=\max(-b,0),\quad
w_{i,t}=\min\{g_{i,t},\max(b,0)\},\quad
s_{i,t}=\max(b,0)-w_{i,t}.
$$

$w$为已付费未利用合同电量，$s$为实际弃光。此规则优先利用光伏。实际平衡及储能递推为：

$$
g-w+V^{\mathrm{act}}-s+e+d^{\mathrm{act}}
=L^{\mathrm{act}}+c^{\mathrm{act}},\qquad
E^{\mathrm{act}}_{i,t+1}=E^{\mathrm{act}}_{i,t}+0.9c^{\mathrm{act}}_{i,t}
-d^{\mathrm{act}}_{i,t}/0.9.
$$

评价窗口最后一天对下一边界施加能到达6000 kWh的上下包络：若还剩$r$槽，则下界为$\max(1200,6000-r\,0.9\bar c)$，上界为$\min(10800,6000+r\,\bar d/0.9)$。以当前物理可达区间裁剪动作，最后一槽边界必为6000；不足部分依然通过5倍补购付费，不人为修改储能状态。

### 5.2 实际费用

$$
C_i^{\mathrm{tot}}=\sum_tp_tg_{i,t}+5\sum_tp_te_{i,t}.
$$

所有电量已经是kWh，不再乘时间步长。附件“计划购电量”的全天购电费填写第一项；包含紧急购电的总费用在本文和每日明细中单列，防止混淆。

## 6. 实验设计与最终方案确定

先在1月15—31日开展17天影子验证。所有候选从6000 kWh开始，连续运行，验证末端恢复6000；预测、校准和模型选择只用该日之前数据。以这17天累计实际费用选定方案，并在2月1日冻结调度方案标签。输出期为2月1日至12月31日334天，各对照统一初末状态及执行器。

由于此前已经观察全年基准统计，本研究属于按时间信息边界实施的历史回测，不能称为完全未接触数据的盲测。本文也不把全年事后最优候选反向用作2月1日的选型理由。

"""
names={'M0':'P0＋确定性','M1':'P0＋分位数','M2':'P1＋确定性','M3':'P1＋分位数','M4':'P1＋受限场景'}
text+=table(['方案','方法','1月验证费用（元）'],[[m,names[m],fmt(v)] for m,v in sel['validation_cost'].items()])
text+=f"验证费用最低的是**{name}：{names[name]}**，因此本次result2.xlsx采用该方案。风险分位数为预设0.8，不用全年敏感性成绩回调主参数。\n\n"
text+="### 6.1 预测结果\n\n"
text+=table(['目标','预测器','MAE（kW）','RMSE（kW）'],[[r.target,r.model,fmt(r.mae_kw),fmt(r.rmse_kw)] for r in forecast.itertuples()])
text+="负载改进有效，但光伏在线选择未超过强历史基准。说明算法复杂度不能替代数据适配性；P1的负载和光伏分别选择，未强行融合所有模型。\n\n### 6.2 全年费用与风险对照\n\n"
text+=table(['方案','总费用（元）','紧急费用（元）','紧急电量（kWh）','紧急购电天数'],[[m,fmt(comp.loc[m,'cost']),fmt(comp.loc[m,'emergency_cost']),fmt(comp.loc[m,'emergency_kwh']),int(comp.loc[m,'emergency_days'])] for m in ['R0','M0','M1','M2','M3','M4']])
text+=f"""相对同预测、同风险参数的无储能R0，主方案节约{fmt(nos.cost-chosen.cost)}元，降幅{100*(nos.cost-chosen.cost)/nos.cost:.2f}%。这一配对主要体现储能及相应调度的价值。相对M0，风险修正节约{fmt(point.cost-chosen.cost)}元，降幅{100*(point.cost-chosen.cost)/point.cost:.2f}%，紧急购电量下降{100*(point.emergency_kwh-chosen.emergency_kwh)/point.emergency_kwh:.2f}%。

全年事后最低费用为M3，比交付主方案少{fmt(chosen.cost-comp.loc['M3','cost'])}元（主方案费用的{100*(chosen.cost-comp.loc['M3','cost'])/chosen.cost:.2f}%）。这表明1月短验证窗存在季节代表性不足；M3值得后续采用预先设计的滚动策略选型进一步验证。本文保留M1作为预先选定、无选型前视的正式交付，不把它宣称为全局最优或全年最佳候选。

![月度购电费比较]({out}/monthly_cost.png)

M4紧急电量较低不等于总费用最低；多计划购电也会增加费用。该结论再次说明应同时核算两类费用，不能单独追求紧急购电为零。

### 6.3 风险参数、场景数与下界

"""
text+=table(['情形','总费用（元）','紧急电量（kWh）'],[[m,fmt(comp.loc[m,'cost']),fmt(comp.loc[m,'emergency_kwh'])] for m in ['M1_a0.7_s14','M1','M1_a0.9_s14','M4','M4_a0.8_s28']])
text+="提高分位数降低紧急购电，但过度保守会增加已付费计划量；敏感性仅用于解释，不据此重选已冻结方案。场景数比较只覆盖14与28个，不能据此宣称场景近似已经收敛。\n\n"
text+=table(['预测配置','目标分位数','经验覆盖率','pinball loss（kWh）'],[[r.forecast,r.alpha,f'{100*r.coverage:.2f}%',fmt(r.pinball_kwh)] for r in qcal.itertuples()])
text+=f"""覆盖率是逐时经验覆盖率，包含光伏夜间零值和时序非平稳影响，不是全天联合保证。

另求解知道实际负载与光伏的完美信息模型，并放松各月初末状态的衔接，得到费用下界{fmt(bound)}元。月边界可自由取可行电量，因此该值比严格全年连续完美信息最优值更宽松。主方案相对该下界高{100*(chosen.cost-bound)/bound:.2f}%；这是对松弛下界的差距，不能当作实际可实现改进空间。下界已扣除数值择优项的保守上界。

主方案日费用95%分位数为{fmt(chosen.p95_daily_cost)}元，最贵17天（约5%）平均费用为{fmt(chosen.tail5_mean)}元。所有算法耗时为本机本轮观测，受并发和系统负载影响，不代表硬件无关性能。

"""
text+=table(['方案','平均调度及执行耗时（秒/日）','最大耗时（秒/日）','转MILP天数'],[[m,f"{comp.loc[m,'mean_seconds']:.4f}",f"{comp.loc[m,'max_seconds']:.4f}",int(comp.loc[m,'milp_days'])] for m in names])
ci=pd.read_csv(OUT/'paired_cost_intervals.csv')
text+="主方案相对各对照的配对日费用差采用7天移动块自助法（1000次、随机种子2025），正值代表主方案节约；14天块结果保存于附件。区间是历史样本的描述性不确定性，不消除既有探索或政策选择偏差。\n\n"
text+=table(['对照','平均日节约（元）','95%区间（元/日）'],[[r.comparator,fmt(r.mean_saving_yuan_per_day),f'[{fmt(r.lower95)}, {fmt(r.upper95)}]'] for r in ci[ci.block_days==7].itertuples()])
text+="## 7. 四个指定日期的具体结果\n\n下列计划购电为0点冻结值，充放电和储能为实际执行值。为对应题目表1，同时列出计划购电费及含紧急购电的总费用。显示保留两位小数，附件保存完整数值。\n\n"
for date,s in payload['specified'].items():
    text+=f"### {date}\n\n"
    v=s['specified_g']
    text+=table(['时间段','计划购电量（kWh）','时间段','计划购电量（kWh）','时间段','计划购电量（kWh）'],
      [['10:00-10:10',fmt(v[0]),'12:00-12:10',fmt(v[1]),'14:00-14:10',fmt(v[2])],
       ['16:00-16:10',fmt(v[3]),'18:00-18:10',fmt(v[4]),'20:00-20:10',fmt(v[5])]])
    text+=table(['指标','数值'],[['全天计划购电量（kWh）',fmt(s['plan_kwh'])],['全天紧急购电量（kWh）',fmt(s['emergency_kwh'])],['全天计划购电费（元）',fmt(s['plan_cost'])],['全天紧急购电费（元）',fmt(s['emergency_cost'])],['全天总费用（元）',fmt(s['total_cost'])]])
    blocks=s['blocks']
    text+=table(['时间段','充电量（kWh）','放电量（kWh）','时间段','充电量（kWh）','放电量（kWh）'],
       [[*([blocks[k][0]]+[fmt(z) for z in blocks[k][1:]]),*([blocks[k+1][0]]+[fmt(z) for z in blocks[k+1][1:]])] for k in [0,2,4]])
    text+=table(['0:00储电量（kWh）','24:00储电量（kWh）'],[[fmt(s['initial']),fmt(s['terminal'])]])
text+="### 指定日期紧急购电汇总（对应表3）\n\n相邻且均发生紧急购电的10分钟区间合并，电量求和；不跨日合并。\n\n"
events=[payload['specified'][d]['emergency_intervals'] for d in payload['specified']]
headers=[]
for d in payload['specified']: headers.extend([d+' 时间段','购电量（kWh）'])
rows=[]
for j in range(max(1,max(map(len,events)))):
    row=[]
    for es in events:
        row.extend([es[j][0],fmt(es[j][1])] if j<len(es) else (['无','0.00'] if j==0 else ['—','—']))
    rows.append(row)
text+=table(headers,rows)
text+=f"\n![指定日期购电及储能轨迹]({out}/specified_days.png)\n\n"
text+="## 8. 约束核验与适用边界\n\n"
text+=table(['核验项','结果'],[[k,v] for k,v in checks.items()])
text+=f"""对全部48096个输出区间独立重算母线平衡、储能递推、功率与容量边界、充放电互斥及跨日连续性，结果均在$10^{{-5}}$ kWh容差内。初末状态均为6000 kWh。主方案全年充电{fmt(daily.charge_kwh.sum())} kWh、放电{fmt(daily.discharge_kwh.sum())} kWh；因初末电量相等，充放电差{fmt(daily.charge_kwh.sum()-daily.discharge_kwh.sum())} kWh对应储能损耗。另核对全部费用分项及弃光/未利用合同电量的来源边界。

求解器最优仅指当日给定预测和假设下的优化问题，不意味着随机真实运行费用达到全局最优。局限包括：1月验证窗较短、未使用气象输入、固定终端目标、局部实时反馈、光伏夜间规则的历史近似，以及M4共享控制限制。未检验效率另一解释、观测延迟或更完整多阶段MPC，不对这些场景宣称相同收益。

## 9. 附件与复现材料

"""
links=[('题目要求完整结果 result2.xlsx','result2.xlsx'),('实际10分钟购电、充放电及状态明细','dispatch_10min.csv'),
 ('每日电量与费用汇总','daily_summary.csv'),('全部方案与敏感性比较','comparison.csv'),
 ('逐月比较','monthly_comparison.csv'),('预测误差比较','forecast_metrics.csv'),
 ('选型依据','selection.json'),('独立物理审计及数据哈希','verification.json'),
 ('导出Excel逐格核验','workbook_verification.json'),('配对费用差区间','paired_cost_intervals.csv')]
for label,file in links: text+=f"- [{label}]({out}/{file})\n"
text+=f"\n[完整文献核查与前期实验设计]({base}/docs/第二问文献依据与实验设计.md)。数据和模板保持原样，复现说明见[{ 'src/README.md' }]({base}/src/README.md)。结果模板采用静态数值，修改输入后需重算并重新导出。\n\n"
text+=r"""## 参考文献

[1] Appino R R, González Ordiano J Á, Mikut R, Hagenmeyer V, Faulwasser T. Storage Scheduling with Stochastic Uncertainties: Feasibility and Cost of Imbalances[C]. PSCC, 2018. [公开作者稿及元数据](https://arxiv.org/abs/1805.02525)，[全文](https://arxiv.org/pdf/1805.02525)。用于两阶段计划、储能偏差与不平衡费用的建模依据。

[2] Beykirch M, Janke T, Steinke F. Bidding and Scheduling in Energy Markets: Which Probabilistic Forecast Do We Need?[C]. PMAPS, 2022. [公开作者稿及元数据](https://arxiv.org/abs/2203.13159)。用于说明储能状态关联与预测分布需求，未据此宣称本次受限场景已实现完整联合优化。

[3] Ziel F, Liu B. Lasso estimation for GEFCom2014 probabilistic electric load forecasting[J]. International Journal of Forecasting, 2016, 32(3): 1029–1037. DOI: [10.1016/j.ijforecast.2016.01.001](https://doi.org/10.1016/j.ijforecast.2016.01.001). [公开作者稿](https://arxiv.org/abs/1603.01376)。支持周期、日历与正则化思想，本文Ridge不是原模型复现。

[4] Gneiting T, Lerch S, Schulz B. Probabilistic solar forecasting: Benchmarks, post-processing, verification[J]. Solar Energy, 2023, 252: 72–80. DOI: [10.1016/j.solener.2022.12.054](https://doi.org/10.1016/j.solener.2022.12.054). [开放全文](https://publikationen.bibliothek.kit.edu/1000155949/150349405)。用于基准、概率评分与校准依据。

[5] Donti P L, Amos B, Kolter J Z. Task-based End-to-end Model Learning in Stochastic Optimization[C]. Advances in Neural Information Processing Systems 30, 2017. [公开作者稿及出版信息](https://arxiv.org/abs/1703.04529)。用于下游成本导向评价，本文未实现其端到端神经优化。

[6] Gupta R K, Sossan F, Paolone M. Grid-aware Distributed Model Predictive Control of Heterogeneous Resources in a Distribution Network: Theory and Experimental Validation[EB/OL]. 2020（公开作者稿年份）. arXiv:2008.02848. [公开作者稿](https://arxiv.org/abs/2008.02848)。用于日前/实时分层架构，本文未引入原文网络参数与多设备控制。

[7] Hyndman R J, Athanasopoulos G. Forecasting: Principles and Practice[M/OL]. 3rd ed., §5.10 Time series cross-validation. [作者公开教材](https://otexts.com/fpp3/tscv.html)。本文时间顺序验证遵循该原则，不使用随机时段划分。

文献访问日期：2026-09-11。论文与作者稿用于支持方法及适用边界，所有本题费用、精度与收益数字均来自本项目计算，不引用其他系统的收益比例作为本题结果。
"""
text=text.replace('u^0','a^0').replace(r'u^\omega',r'a^\omega')
text=text.replace(r'b=x+d^{\mathrm{act}}-c^{\mathrm{act}}',r'B^{\mathrm{sur}}=x+d^{\mathrm{act}}-c^{\mathrm{act}}')
text=text.replace(r'\max(-b,0)',r'\max(-B^{\mathrm{sur}},0)').replace(r'\max(b,0)',r'\max(B^{\mathrm{sur}},0)')
text=text.replace('还剩$r$槽','还剩$n^{\\mathrm{rem}}$槽').replace(r'6000-r\,',r'6000-n^{\mathrm{rem}}\,').replace(r'6000+r\,',r'6000+n^{\mathrm{rem}}\,')
text=text.replace('### 6.2 全年费用','### 6.2 输出期费用').replace('全年事后','输出期事后').replace('主方案全年充电','主方案输出期充电')
event_dates=[date for date,values in payload['specified'].items() if values['emergency_kwh']>1e-7]
event_summary='、'.join(event_dates) if event_dates else '无'
text=text.replace('## 8. 约束核验与适用边界',
    f"四个指定日中，发生紧急购电的日期为：{event_summary}；其余指定日为0。完整输出期有{int((daily.emergency_kwh>1e-7).sum())}天发生紧急购电，不能用指定日代替输出期风险评价。\n\n## 8. 约束核验与适用边界")
paper.parent.mkdir(exist_ok=True,parents=True);paper.write_text(text,encoding='utf8')
print('Saved manuscript',paper,'lines',len(text.splitlines()))
