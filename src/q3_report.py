"""Generate the reproducible Question 3 model and results paper section."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd

from q3_workbook_data import contiguous_emergency_ranges, interval_time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/q3_final'
PAPER = ROOT / 'docs/result/第三问求解与结果分析.md'
SPECIFIED_DATES = ('2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21')


def table(headers, rows):
    return ('| ' + ' | '.join(headers) + ' |\n|' + '|'.join('---' for _ in headers)
            + '|\n' + ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows)
            + '\n')


def fmt(value):
    return f'{float(value):,.2f}'


def _selected_row(payload):
    label = f"calibrated_{payload['selected']['release_name']}"
    return next(row for row in payload['comparison'] if row['label'] == label)


def render_report(payload):
    selected = payload['selected']
    chosen = _selected_row(payload)
    release_name = selected['release_name']
    updates = selected['updates']
    update_text = '仅0点' if not updates else '0点及' + '、'.join(f'{h}点' for h in updates)
    comparison = payload['comparison']
    calibrated = [row for row in comparison if str(row['label']).startswith('calibrated_')]
    q2_cost = float(payload['q2_cost'])
    delta_q2 = float(chosen['cost']) - q2_cost
    origin = pd.Timestamp('2025-01-01')
    history_start = origin + pd.Timedelta(days=payload['calibration_start_day'])
    history_end = origin + pd.Timedelta(days=payload['calibration_end_exclusive'] - 1)
    last_history = origin + pd.Timedelta(days=payload['last_history_day'])
    decision_date = origin + pd.Timedelta(days=payload['decision_day'])
    calibration_dates = f'{history_start:%Y-%m-%d} 至 {history_end:%Y-%m-%d}'
    update_parameter_note = (
        '日内参数用于所选发布时间的历史残差分位数校准。' if updates else
        'R0不调用日内参数，该参数仅保留在预选协议中以记录候选定义。')
    baseline = next((row for row in calibrated if row['label'] == 'calibrated_R0'), None)
    information_text = f'事前选型选择了 {update_text}（{release_name}）。'
    if baseline is not None:
        if updates:
            compared = chosen
        else:
            compared = min((row for row in calibrated if row['label'] != 'calibrated_R0'),
                           key=lambda row: float(row['cost']), default=None)
        if compared is not None:
            difference = float(compared['cost']) - float(baseline['cost'])
            change = ('费用持平' if abs(difference) <= 1e-6 else
                      f"费用{'减少' if difference < 0 else '增加'} {fmt(abs(difference))} 元")
            information_text += (
                f"在输出期的事后对照中，{compared['label']} 相对 R0 的总{change}。"
                '该对照包含计划费、调整相关费用和紧急购电费，不能反向改变事前选型。')
        else:
            information_text += '未提供日内策略的输出期对照，不能判断其费用效果。'
    else:
        information_text += '未提供 R0 的输出期对照，不能判断日内调整的净节费。'
    information_text += ('\n\n按事前冻结协议采用所选的日内调整。' if updates else
                         '\n\n按事前冻结协议保持0点合同，不作日内调整。')
    information_text += (
        '这只描述当前候选与协议下的策略选择；并非每个发布时间都值得或必须调整，'
        '也不能由此断言其他预报没有信息价值。若结算、预测分布或控制器改变，应重新评价。')
    refund_row = next((row for row in comparison if row['label'] == f'N1_{release_name}'), None)
    if refund_row is None:
        refund_text = f'未提供所选 {release_name} 策略的 N1 对照结果，不作退款效果判断。'
    else:
        refund_delta = float(refund_row['cost']) - float(chosen['cost'])
        refund_change = ('费用持平' if abs(refund_delta) <= 1e-6 else
                         f"费用{'减少' if refund_delta < 0 else '增加'} {fmt(abs(refund_delta))} 元")
        refund_text = (f"退款 N1 下，同一所选策略 {release_name} 的总费用为 "
                       f"{fmt(refund_row['cost'])} 元，相对 N0 的{refund_change}。")
        if not updates:
            refund_text += ('该策略没有日内调整，不能据此识别退款对日内选型的价值。'
                            if abs(refund_delta) <= 1e-6 else
                            '该策略没有日内调整，仍出现费用差异，应核查对照输入及结算实现。')
        else:
            refund_text += '此处固定原选型作结算对照，并未在N1下重新选择风险参数和发布时间。'

    text = fr'''# 第三问求解与结果分析

## 1. 问题解释与结算口径

第三问允许微网在每天 0:00 建立原始购电合同，并在 6:00、12:00、18:00 获得新光伏预报后修改尚未执行的合同。本文严格区分原计划 $g_{{i,t}}$、某次发布后的合同总量 $q^{{[\tau]}}_{{i,t}}$、相对原计划净调整 $\delta^{{[\tau]}}_{{i,t}}$、相对上一版本的变化 $\Delta q^{{[\tau]}}_{{i,t}}$、最终合同和实际接收购电量。

主结算采用 N0：不退款，下调部分另付对应原计划电费 50% 的违约费。单槽常规合同费用为

$$
K^{{N0}}(g,q)=p\left(g+1.5[q-g]_++0.5[g-q]_+\right).
$$

紧急购电另按 $5pe$ 计费。中间合同版本可以再次修订，但不重复累计收费；完整版本另存审计账本。退款 N1 仅作敏感性，绝不与 N0 主结果混合。

## 2. 三层滚动模型

每天包含 144 个 10 分钟区间，所有流量均为交流母线侧 kWh。电池内部储电量满足

$$
E_{{i,t+1}}=E_{{i,t}}+0.9c_{{i,t}}-\frac{{d_{{i,t}}}}{{0.9}},
\qquad 1200\le E_{{i,t}}\le10800,
$$

且 $0\le c_{{i,t}},d_{{i,t}}\le5000/6$。母线平衡为

$$
q_{{i,t}}-w_{{i,t}}+e_{{i,t}}+V_{{i,t}}-s_{{i,t}}+d_{{i,t}}
=L_{{i,t}}+c_{{i,t}}.
$$

0:00 层建立原始合同；日内发布层只修改尚未执行区间；10 分钟执行层以真实负载、真实光伏和连续电池状态进行反馈。已付费多余合同可以拒收但不退款，不允许售电或把电池电量假记为弃光。评价期 2025-02-01 至 2025-12-31 共 334 天，期初和期末储电量均为 6000 kWh，实际状态跨日连续。

## 3. 预测与风险校准

负载使用第二问已构造的因果在线选择预测；光伏使用附件 3 在当次发布时可见的 24 小时整点预报，并按小时内恒定功率映射为 10 分钟电量。对每个发布时间和剩余时距，从此前最多 56 个已完成日期构造净负荷残差经验分布。任何决策均不读取当日未来实际负载或实际光伏，未来实际只用于事后执行与评分。

风险候选与八种发布时间组合先在 {calibration_dates} 进行连续影子回测，最后历史日为 {last_history:%Y-%m-%d}，在 {decision_date:%Y-%m-%d} 的0点决策前冻结选择。选中参数为 0 点分位数 {selected['risk_pair']['alpha0']:.2f}、日内分位数 {selected['risk_pair']['alpha_update']:.2f}；选中发布时间为 **{update_text}（{release_name}）**。{update_parameter_note}

这种滚动预测—优化结构与储能随机 MPC 的思想一致，但本文没有把其他市场规则或设备参数搬入本题。[Patrinos、Trimboli、Bemporad（2011）](https://cse.lab.imtlucca.it/~bemporad/publications/papers/cdc11-dispatch.pdf)

## 4. 发布时间组合与信息价值

'''
    rows = []
    for row in calibrated:
        rows.append([row['label'], fmt(row['cost']), fmt(row['emergency_kwh']),
                     int(row['emergency_days']), int(row['emergency_events'])])
    text += table(['策略', '总费用（元）', '紧急购电（kWh）', '紧急天数', '合并事件数'], rows)
    text += f'''{information_text}

## 5. 正式策略费用

正式策略总费用为 **{fmt(chosen['cost'])} 元**，其中 0 点计划费 {fmt(chosen['plan_cost'])} 元、上调费 {fmt(chosen['up_cost'])} 元、下调违约费 {fmt(chosen['down_fee'])} 元、紧急购电费 {fmt(chosen['emergency_cost'])} 元。紧急购电量为 {fmt(chosen['emergency_kwh'])} kWh，发生于 {int(chosen['emergency_days'])} 天、{int(chosen['emergency_events'])} 个合并事件；已付费拒收量为 {fmt(chosen['unused_kwh'])} kWh，弃光为 {fmt(chosen['spill_kwh'])} kWh。

第二问 v2 的既有正式策略费用为 {fmt(q2_cost)} 元。问题三正式策略比其{'高' if delta_q2 >= 0 else '低'} {fmt(abs(delta_q2))} 元（{abs(delta_q2)/q2_cost*100:.2f}%）。两者合同机会集合和预报输入不同，该比较用于定位效果，不把第二问策略冒充为第三问可行解。

{refund_text}

## 6. 指定日期结果

以下按题目表 1—表 3 的信息结构报告 2025-03-20、2025-06-21、2025-09-23、2025-12-21。Excel 保存完整精度，正文保留两位小数。

'''
    for date, result in payload.get('specified', {}).items():
        text += f'### {date}\n\n'
        text += table(['时刻', '0点计划购电（kWh）', '最终合同（kWh）'], [
            [time, fmt(g), fmt(q)] for time, g, q in result['selected_slots']])
        text += table(['时间段', '实际充电量（kWh）', '实际放电量（kWh）'], [
            [block['period'], fmt(block['charge']), fmt(block['discharge'])]
            for block in result['blocks']])
        text += table(['指标', '数值'], [
            ['0:00储电量（kWh）', fmt(result['initial'])],
            ['24:00储电量（kWh）', fmt(result['terminal'])],
            ['全天计划购电量（kWh）', fmt(result['plan_kwh'])],
            ['全天最终合同量（kWh）', fmt(result['q_kwh'])],
            ['全天总费用（元）', fmt(result['cost'])],
            ['紧急购电量（kWh）', fmt(result['emergency_kwh'])],
        ])
        events = result['emergency'] or [{'period': '无', 'kwh': 0.}]
        text += table(['紧急购电区间', '购电量（kWh）'], [
            [event['period'], fmt(event['kwh'])] for event in events])

    text += '''## 7. 验证与适用边界

正式交付从磁盘重新读取 48,096 条执行记录，检查合同版本重建、N0 费用、母线平衡、SOC 递推、充放电互斥、功率/容量边界、拒收和弃光边界、跨日连续以及评价末日闭合。`result3.xlsx` 的计划与调整矩阵、充放电汇总和全部紧急购电事件已经与 CSV 逐项复核；验证结果见独立 JSON。

本文是固定协议后的历史因果回放，不是未经接触 2025 年数据的前瞻试验。限制包括一年样本、1 月季节代表性有限、H0 小时映射、有限风险候选和简单执行反馈。完整场景树未达到本轮必要性门槛，不进入正式结果。概率预测应同时评价校准和尖锐性，点误差改善也不能直接等同于调度费用改善。[Gneiting 等（2023）](https://arxiv.org/abs/2105.05251)

## 8. 复现与附件

- [正式结果 result3.xlsx](../../outputs/q3_final/result3.xlsx)
- [10分钟执行账本](../../outputs/q3_final/dispatch_10min.csv)
- [每日费用汇总](../../outputs/q3_final/daily_summary.csv)
- [合同版本账本](../../outputs/q3_final/contract_versions.csv)
- [候选策略比较](../../outputs/q3_final/comparison.csv)
- [1月事前选择协议](../../outputs/q3_final/selection_protocol.json)
- [独立交付核验](../../outputs/q3_final/delivery_verification.json)

复现顺序为：运行 `src/q3_experiments.py --stage formal`，生成工作簿载荷并运行 `src/q3_workbook.mjs`，生成本文，最后运行 `src/q3_verify_delivery.py`。`result3.xlsx` 保留模板四张工作表，并把省略号示例扩展为 334 天完整结果。
'''
    return text


def load_payload():
    protocol = json.loads((OUT / 'selection_protocol.json').read_text(encoding='utf-8'))
    comparison = pd.read_csv(OUT / 'comparison.csv').to_dict('records')
    daily = pd.read_csv(OUT / 'daily_summary.csv')
    dispatch = pd.read_csv(OUT / 'dispatch_10min.csv')
    specified = {}
    selected_slots = ((60, '10:00—10:10'), (72, '12:00—12:10'),
                      (84, '14:00—14:10'), (96, '16:00—16:10'),
                      (108, '18:00—18:10'), (120, '20:00—20:10'))
    for date in SPECIFIED_DATES:
        frame = dispatch[dispatch.date == date].sort_values('interval_id')
        day = daily[daily.date == date].iloc[0]
        blocks = []
        for block in range(6):
            part = frame.iloc[block*24:(block+1)*24]
            blocks.append({'period': f'{block*4}:00—{(block+1)*4}:00',
                           'charge': float(part.c.sum()),
                           'discharge': float(part.d.sum())})
        events = [{'period': f"{interval_time(event['start'])}—{interval_time(event['end'])}",
                   'kwh': event['kwh']}
                  for event in contiguous_emergency_ranges(frame)]
        specified[date] = dict(
            selected_slots=[(label, float(frame.iloc[slot].g), float(frame.iloc[slot].q))
                            for slot, label in selected_slots],
            blocks=blocks, initial=float(frame.iloc[0].E_start),
            terminal=float(frame.iloc[-1].E_end), plan_kwh=float(frame.g.sum()),
            q_kwh=float(frame.q.sum()), cost=float(day.cost),
            emergency_kwh=float(frame.e.sum()), emergency=events)
    q2 = json.loads((ROOT / 'outputs/q2_review_v2/experiments/Rolling_T0.json').read_text())
    return dict(winner=protocol['winner'], selected=protocol['selected'],
                comparison=comparison, specified=specified, q2_cost=q2['cost'],
                **{key: protocol[key] for key in ('calibration_start_day',
                   'calibration_end_exclusive', 'last_history_day', 'decision_day')})


def main():
    payload = load_payload()
    text = render_report(payload)
    PAPER.parent.mkdir(parents=True, exist_ok=True)
    PAPER.write_text(text, encoding='utf-8')
    print(json.dumps({'path': str(PAPER.relative_to(ROOT)),
                      'sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'selected': payload['selected']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
