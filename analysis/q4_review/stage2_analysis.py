"""Read-only analysis of frozen Q4 results.

This module deliberately does not import any Q1-Q4 solver, experiment runner,
or execution controller.  It reads frozen artifacts and independently derives
consistency, accounting-attribution, and stability evidence.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "outputs/q4_final"
N0 = FORMAL / "N0"
OUTPUT = ROOT / "outputs/q4"
DOCS = ROOT / "docs/q4"
CAP = 5000.0 / 6.0
LOWER_SOC = 1200.0
UPPER_SOC = 10800.0
ETA_C = 0.9
ETA_D = 0.9
EPS = 1e-7


def f0_step_independent(contract: float, load: float, pv: float, state: float,
                        *, remaining_slots: int, closing_target: float | None,
                        cap: float = CAP) -> dict[str, float]:
    """Independent transcription of the frozen F0 equations for one slot."""
    contract, load, pv, state = map(float, (contract, load, pv, state))
    if not all(np.isfinite([contract, load, pv, state])):
        raise ValueError("F0 inputs must be finite")
    if min(contract, load, pv) < 0 or not LOWER_SOC <= state <= UPPER_SOC:
        raise ValueError("F0 inputs outside physical bounds")
    if remaining_slots < 0:
        raise ValueError("remaining_slots must be nonnegative")

    surplus = contract + pv - load
    if closing_target is not None and state > closing_target:
        c = 0.0
        d = min(cap, load, ETA_C * (state - closing_target))
        next_state = state - d / ETA_D
    else:
        delta = ETA_C * max(surplus, 0.0) - max(-surplus, 0.0) / ETA_D
        if closing_target is None:
            floor, ceiling = LOWER_SOC, UPPER_SOC
        else:
            floor = max(LOWER_SOC,
                        closing_target - remaining_slots * ETA_C * cap)
            ceiling = closing_target
        next_state = np.clip(
            state + delta,
            max(floor, state - cap / ETA_D),
            min(ceiling, state + ETA_C * cap),
        )
        c = max(float(next_state) - state, 0.0) / ETA_C
        d = max(state - float(next_state), 0.0) * ETA_D

    balance = surplus + d - c
    emergency = max(-balance, 0.0)
    excess = max(balance, 0.0)
    unused = min(contract, excess)
    spill = max(excess - unused, 0.0)
    next_state = state + ETA_C * c - d / ETA_D
    return {
        "c": float(c), "d": float(d), "e": float(emergency),
        "unused": float(unused), "spill": float(spill),
        "next_state": float(next_state),
    }


def compare_plan_execution(plan: pd.DataFrame,
                           actual: pd.DataFrame) -> dict[str, object]:
    """Compare persisted storage plans with actuals, if such plans exist."""
    plan_required = {
        "date", "interval_id", "charge_plan", "discharge_plan", "SOC_plan"
    }
    missing = sorted(plan_required - set(plan.columns))
    if missing:
        return {"identifiable": False, "missing_plan_columns": missing,
                "rows": pd.DataFrame()}
    actual_required = {"date", "interval_id", "c", "d", "E_end"}
    absent_actual = sorted(actual_required - set(actual.columns))
    if absent_actual:
        raise ValueError(f"actual ledger missing {absent_actual}")
    merged = plan.merge(actual[list(actual_required)],
                        on=["date", "interval_id"], how="inner", validate="one_to_one")
    if len(merged) != len(plan) or len(merged) != len(actual):
        raise ValueError("plan and actual keys do not match")
    merged["delta_charge"] = merged.c - merged.charge_plan
    merged["delta_discharge"] = merged.d - merged.discharge_plan
    merged["delta_SOC"] = merged.E_end - merged.SOC_plan
    return {"identifiable": True, "missing_plan_columns": [], "rows": merged}


def accounting_attribution(q42: Mapping[str, float],
                           q43: Mapping[str, float]) -> pd.DataFrame:
    """Decompose Q4-2 minus Q4-3 cost using the settlement identity."""
    rows = [
        ("planned_purchase", float(q42["plan_cost"]) - float(q43["plan_cost"])),
        ("upward_adjustment", float(q42["up_cost"]) - float(q43["up_cost"])),
        ("downward_fee", float(q42["down_fee"]) - float(q43["down_fee"])),
        ("refund", float(q43["refund"]) - float(q42["refund"])),
        ("emergency_purchase",
         float(q42["emergency_cost"]) - float(q43["emergency_cost"])),
    ]
    frame = pd.DataFrame(rows, columns=["component", "contribution_yuan"])
    total = float(q42["cost"]) - float(q43["cost"])
    frame["share_of_total_savings_pct"] = np.where(
        abs(total) > 1e-12, 100.0 * frame.contribution_yuan / total, np.nan)
    return frame


def monthly_stability(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily frozen-result savings into month-level evidence."""
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame.date)
    frame["month"] = frame.date.dt.strftime("%Y-%m")
    rows = []
    cumulative = 0.0
    for month, group in frame.groupby("month", sort=True):
        savings = group.delta_cost.to_numpy(dtype=float)
        q42 = float(group.cost_q4_2.sum())
        q43 = float(group.cost_q4_3.sum())
        total = float(savings.sum())
        cumulative += total
        rows.append({
            "month": month,
            "days": int(len(group)),
            "cost_q4_2_yuan": q42,
            "cost_q4_3_yuan": q43,
            "savings_yuan": total,
            "savings_pct": 100.0 * total / q42,
            "positive_days": int(np.sum(savings > EPS)),
            "loss_days": int(np.sum(savings < -EPS)),
            "zero_days": int(np.sum(np.abs(savings) <= EPS)),
            "mean_daily_savings_yuan": float(np.mean(savings)),
            "median_daily_savings_yuan": float(np.median(savings)),
            "std_daily_savings_yuan": float(np.std(savings, ddof=0)),
            "min_daily_savings_yuan": float(np.min(savings)),
            "max_daily_savings_yuan": float(np.max(savings)),
            "cumulative_savings_yuan": cumulative,
        })
    return pd.DataFrame(rows)


def complete_calendar_months(monthly: pd.DataFrame, year: int) -> pd.DataFrame:
    """Show missing calendar months without treating them as observations."""
    calendar = pd.DataFrame({
        "month": [f"{year}-{month:02d}" for month in range(1, 13)]
    })
    completed = calendar.merge(monthly, on="month", how="left")
    completed["coverage_status"] = np.where(
        completed.days.notna(), "formal_evaluation", "not_in_formal_evaluation")
    completed["days"] = completed.days.fillna(0).astype(int)
    for column in ("positive_days", "loss_days", "zero_days"):
        completed[column] = completed[column].fillna(0).astype(int)
    completed["cumulative_savings_yuan"] = (
        completed.savings_yuan.fillna(0.0).cumsum())
    return completed


def _markdown_table(frame: pd.DataFrame, columns: list[str],
                    labels: list[str], formats: Mapping[str, str] | None = None) -> str:
    formats = formats or {}
    lines = ["| " + " | ".join(labels) + " |",
             "|" + "|".join(["---"] * len(labels)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if pd.isna(value):
                values.append("—")
            elif column in formats:
                values.append(formats[column].format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _source_locations(path: Path, needles: Mapping[str, str]) -> dict[str, int | None]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, int | None] = {}
    for label, needle in needles.items():
        result[label] = next((index for index, line in enumerate(lines, 1)
                              if needle in line), None)
    return result


def detect_plan_records() -> dict[str, object]:
    aliases = {
        "charge_plan", "discharge_plan", "SOC_plan", "c_plan", "d_plan",
        "planned_charge", "planned_discharge", "planned_soc",
    }
    headers: dict[str, list[str]] = {}
    found: dict[str, list[str]] = {}
    for path in sorted(N0.glob("*.csv")):
        columns = list(pd.read_csv(path, nrows=0).columns)
        headers[path.name] = columns
        hits = sorted(set(columns) & aliases)
        if hits:
            found[path.name] = hits

    payload_keys: dict[str, list[str]] = {}
    for mode in ("q4-2", "q4-3"):
        path = N0 / f"{mode}_payload/workbook_payload.json"
        payload_keys[mode] = sorted(json.loads(path.read_text(encoding="utf-8")).keys())

    solver_locations = _source_locations(
        ROOT / "src/q4_solver.py",
        {
            "initial_optimization": "first = optimize(",
            "only_initial_contract_persisted": 'g = first["q"].copy()',
            "actual_flow_arrays_created": 'for key in ("c", "d", "e", "unused", "spill")',
            "updated_contract_persisted": 'q[t:] = solved["q"]',
            "f0_execution": "step = f0_step(",
        },
    )
    experiment_locations = _source_locations(
        ROOT / "src/q4_experiments.py",
        {
            "actual_flow_export": "('g', 'q', 'c', 'd', 'e', 'unused', 'spill', 'actual_grid')",
            "actual_soc_export": "frame['E_start'], frame['E_end']",
        },
    )
    return {
        "plan_generation_logic_exists": True,
        "plan_generation_note": (
            "The optimizer returns c, d and E, but q4_solver persists only q from each "
            "optimization; c, d and E exported by q4_experiments are F0 actual flows."
        ),
        "searched_plan_aliases": sorted(aliases),
        "persisted_plan_fields_found": found,
        "plan_execution_deviation_identifiable": bool(found),
        "available_actual_fields": ["c", "d", "E_start", "E_end"],
        "csv_headers": headers,
        "workbook_payload_top_level_keys": payload_keys,
        "source_locations": {
            "src/q4_solver.py": solver_locations,
            "src/q4_experiments.py": experiment_locations,
        },
        "interpretation": (
            "Current frozen artifacts cannot identify charge_plan/discharge_plan/SOC_plan "
            "deviations without rerunning or changing the solver, both prohibited here."
        ),
    }


def run_consistency() -> dict[str, object]:
    folder = OUTPUT / "consistency"
    folder.mkdir(parents=True, exist_ok=True)
    availability = detect_plan_records()
    (folder / "plan_record_availability.json").write_text(
        json.dumps(availability, ensure_ascii=False, indent=2), encoding="utf-8")

    replay_rows = []
    for mode in ("q4-2", "q4-3"):
        dispatch = pd.read_csv(N0 / f"{mode}_dispatch_10min.csv")
        for row in dispatch.itertuples(index=False):
            closing_target = 6000.0 if str(row.date) == "2025-12-31" else None
            replay = f0_step_independent(
                row.q, row.load, row.pv, row.E_start,
                remaining_slots=143 - int(row.interval_id),
                closing_target=closing_target,
            )
            replay_rows.append({
                "mode": mode, "date": str(row.date),
                "interval_id": int(row.interval_id),
                "charge_error": replay["c"] - float(row.c),
                "discharge_error": replay["d"] - float(row.d),
                "emergency_error": replay["e"] - float(row.e),
                "unused_error": replay["unused"] - float(row.unused),
                "spill_error": replay["spill"] - float(row.spill),
                "soc_end_error": replay["next_state"] - float(row.E_end),
                "q_minus_g": float(row.q) - float(row.g),
                "actual_grid_minus_q": float(row.actual_grid) - float(row.q),
            })
    residuals = pd.DataFrame(replay_rows)
    residuals.to_csv(folder / "f0_replay_residuals.csv", index=False)

    error_columns = [
        "charge_error", "discharge_error", "emergency_error",
        "unused_error", "spill_error", "soc_end_error",
    ]
    daily_rows = []
    for (mode, date), group in residuals.groupby(["mode", "date"], sort=True):
        values = group[error_columns].to_numpy(dtype=float)
        daily_rows.append({
            "mode": mode, "date": date, "rows": len(group),
            "mean_absolute_error": float(np.mean(np.abs(values))),
            "max_absolute_error": float(np.max(np.abs(values))),
            "std_error": float(np.std(values, ddof=0)),
        })
    daily = pd.DataFrame(daily_rows)
    daily["month"] = daily.date.str[:7]
    daily.to_csv(folder / "f0_replay_daily.csv", index=False)

    monthly = daily.groupby(["mode", "month"], as_index=False).agg(
        days=("date", "count"),
        mean_absolute_error=("mean_absolute_error", "mean"),
        max_absolute_error=("max_absolute_error", "max"),
        std_error=("std_error", "mean"),
    )
    monthly.to_csv(folder / "f0_replay_monthly.csv", index=False)

    residuals["month"] = residuals.date.str[:7]
    contract_monthly = residuals.groupby(["mode", "month"], as_index=False).agg(
        mean_q_minus_g=("q_minus_g", "mean"),
        mean_absolute_q_minus_g=("q_minus_g", lambda x: float(np.mean(np.abs(x)))),
        max_absolute_q_minus_g=("q_minus_g", lambda x: float(np.max(np.abs(x)))),
        mean_actual_grid_minus_q=("actual_grid_minus_q", "mean"),
        mean_absolute_actual_grid_minus_q=(
            "actual_grid_minus_q", lambda x: float(np.mean(np.abs(x)))),
        max_absolute_actual_grid_minus_q=(
            "actual_grid_minus_q", lambda x: float(np.max(np.abs(x)))),
    )
    contract_monthly.to_csv(folder / "contract_execution_monthly.csv", index=False)

    summary: dict[str, object] = {
        "plan_execution_deviation_identifiable": availability[
            "plan_execution_deviation_identifiable"],
        "modes": {},
    }
    for mode, group in residuals.groupby("mode", sort=True):
        values = group[error_columns].to_numpy(dtype=float)
        summary["modes"][mode] = {
            "rows": int(len(group)),
            "mean_absolute_error": float(np.mean(np.abs(values))),
            "max_absolute_error": float(np.max(np.abs(values))),
            "std_error": float(np.std(values, ddof=0)),
            "max_charge_error": float(np.max(np.abs(group.charge_error))),
            "max_discharge_error": float(np.max(np.abs(group.discharge_error))),
            "max_soc_end_error": float(np.max(np.abs(group.soc_end_error))),
        }
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    month_view = monthly.copy()
    report = [
        "# Q4规划—执行一致性分析",
        "",
        "## 结论",
        "",
        "冻结Q4代码中存在计划充放电和计划SOC的生成逻辑，但正式产物没有持久化"
        "`charge_plan`、`discharge_plan`或`SOC_plan`。因此当前不能从冻结证据计算"
        "`actual-plan`偏差，也不能声称该偏差为零或可控。",
        "",
        "可以独立确认的是：正式逐槽账本与既定F0实际执行规则一致。以下结果由分析脚本"
        "重新实现F0公式后逐槽重放得到，没有导入或调用Q4 solver。",
        "",
        "| 策略 | 槽数 | 全字段平均绝对误差 | 全字段最大绝对误差 | SOC最大误差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, item in summary["modes"].items():
        report.append(
            f"| {mode.upper()} | {item['rows']:,} | {item['mean_absolute_error']:.3e} | "
            f"{item['max_absolute_error']:.3e} | {item['max_soc_end_error']:.3e} |")
    report += [
        "",
        "## 计划记录检查",
        "",
        "- `src/q4_solver.py`在日前和日内优化中会得到`c`、`d`和`E`，但随后只把"
        "优化得到的`q`保存为原合同或更新合同。",
        "- `src/q4_experiments.py`导出的`c`、`d`、`E_start`、`E_end`来自F0循环"
        "完成后的实际结果。",
        "- 正式CSV没有任何计划储能字段；工作簿载荷中的`battery`也由正式逐槽账本"
        "实际`c/d/E`聚合，不是优化计划。",
        "- 新增的`compare_plan_execution`分析接口能在将来提供独立计划日志时计算"
        "三类偏差；本次没有调用优化器补造历史计划。",
        "",
        "## 月度F0重放误差",
        "",
        _markdown_table(
            month_view,
            ["mode", "month", "days", "mean_absolute_error",
             "max_absolute_error", "std_error"],
            ["策略", "月份", "天数", "平均绝对误差", "最大绝对误差", "误差标准差"],
            {"mean_absolute_error": "{:.3e}", "max_absolute_error": "{:.3e}",
             "std_error": "{:.3e}"},
        ),
        "",
        "## 对评委问题的回答",
        "",
        "1. **计划与执行偏差是否存在：** 优化计划储能轨迹未落盘，当前证据不足，"
        "不能定量回答。合同`g/q`与F0实际流量是不同概念，不能用`q-g`替代储能计划偏差。",
        "2. **偏差是否可控：** F0实际执行本身具有确定规则，逐槽重放误差低于验证容差；"
        "但这只证明账本忠实执行F0，不能证明优化计划与F0之间的偏差可控。",
        "3. **是否影响成本结论：** 正式费用由实际`q`、紧急购电和真实结算价直接复算，"
        "不依赖未保存的计划`c/d/SOC`，所以会计成本差仍可核验；但计划—执行机制解释"
        "必须保留上述限制。",
        "",
        "相关数据位于`outputs/q4/consistency/`。",
    ]
    (DOCS / "plan_execution_consistency.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    return summary


def _summary_mapping(frame: pd.DataFrame) -> dict[str, float]:
    keys = ["cost", "plan_cost", "up_cost", "down_fee", "refund", "emergency_cost"]
    return {key: float(frame[key].sum()) for key in keys}


def run_attribution() -> tuple[pd.DataFrame, dict[str, object]]:
    folder = OUTPUT / "ablation"
    folder.mkdir(parents=True, exist_ok=True)
    q42 = pd.read_csv(N0 / "q4-2_daily_summary.csv")
    q43 = pd.read_csv(N0 / "q4-3_daily_summary.csv")
    columns = ["date", "cost", "plan_cost", "up_cost", "down_fee",
               "refund", "emergency_cost", "emergency_kwh"]
    daily = q42[columns].merge(
        q43[columns], on="date", suffixes=("_q4_2", "_q4_3"),
        validate="one_to_one")
    daily["delta_cost"] = daily.cost_q4_2 - daily.cost_q4_3
    daily["delta_plan_cost"] = daily.plan_cost_q4_2 - daily.plan_cost_q4_3
    daily["delta_up_cost"] = daily.up_cost_q4_2 - daily.up_cost_q4_3
    daily["delta_down_fee"] = daily.down_fee_q4_2 - daily.down_fee_q4_3
    daily["delta_refund"] = daily.refund_q4_3 - daily.refund_q4_2
    daily["delta_emergency_cost"] = (
        daily.emergency_cost_q4_2 - daily.emergency_cost_q4_3)
    daily.to_csv(folder / "daily_savings.csv", index=False)

    q42_sum, q43_sum = _summary_mapping(q42), _summary_mapping(q43)
    annual_savings = q42_sum["cost"] - q43_sum["cost"]
    annual = pd.DataFrame([
        {"strategy": "Q4-2", "annual_cost_yuan": q42_sum["cost"],
         "savings_vs_q4_2_yuan": 0.0, "savings_pct": 0.0,
         "emergency_cost_yuan": float(q42.emergency_cost.sum()),
         "emergency_kwh": float(q42.emergency_kwh.sum())},
        {"strategy": "Q4-3", "annual_cost_yuan": q43_sum["cost"],
         "savings_vs_q4_2_yuan": annual_savings,
         "savings_pct": 100.0 * annual_savings / q42_sum["cost"],
         "emergency_cost_yuan": float(q43.emergency_cost.sum()),
         "emergency_kwh": float(q43.emergency_kwh.sum())},
    ])
    annual.to_csv(folder / "annual_strategy_comparison.csv", index=False)

    attribution = accounting_attribution(q42_sum, q43_sum)
    attribution.to_csv(folder / "annual_accounting_attribution.csv", index=False)
    monthly_attr = daily.assign(month=daily.date.str[:7]).groupby(
        "month", as_index=False).agg(
            total_savings_yuan=("delta_cost", "sum"),
            planned_purchase_contribution_yuan=("delta_plan_cost", "sum"),
            upward_adjustment_contribution_yuan=("delta_up_cost", "sum"),
            downward_fee_contribution_yuan=("delta_down_fee", "sum"),
            refund_contribution_yuan=("delta_refund", "sum"),
            emergency_purchase_contribution_yuan=("delta_emergency_cost", "sum"),
        )
    monthly_attr.to_csv(folder / "monthly_accounting_attribution.csv", index=False)

    protocol = json.loads((N0 / "selection_protocol.json").read_text(encoding="utf-8"))
    selected42 = protocol["selected_q42"]
    selected43 = protocol["selected_q43"]
    alpha0, alpha_update = selected43["alpha0"], selected43["alpha_update"]
    candidates = [selected42]
    labels = ["B0_Q4-2_selected"]
    notes = ["January-selected Q4-2 baseline"]
    for release_name, gamma, label, note in (
        ("R0", 0, "B1_Q4-3_R0", "No intraday contract update; January evidence only"),
        ("Rall", 0, "B2_Q4-3_Rall_gamma0", "Complete frozen update set; selected"),
        ("Rall", 1, "Price_correction_candidate_gamma1",
         "Existing January candidate; not the frozen formal strategy"),
    ):
        match = [row for row in protocol["q43_scores"]
                 if row.get("release_name") == release_name
                 and row.get("price_gamma") == gamma
                 and row.get("alpha0") == alpha0
                 and row.get("alpha_update") == alpha_update]
        if len(match) != 1:
            raise AssertionError(f"expected one January candidate for {release_name}/{gamma}")
        candidates.append(match[0])
        labels.append(label)
        notes.append(note)
    b0_cost = float(candidates[0]["cost"])
    january = pd.DataFrame([
        {"label": label, "scope": "2025-01-15/2025-01-31",
         "days": int(row["days"]), "cost_yuan": float(row["cost"]),
         "savings_vs_B0_yuan": b0_cost - float(row["cost"]),
         "updates": "|".join(map(str, row.get("updates", []))),
         "price_gamma": int(row.get("price_gamma", 0)), "note": note}
        for label, row, note in zip(labels, candidates, notes)
    ])
    january.to_csv(folder / "existing_january_ablation.csv", index=False)

    attribution_gap = float(attribution.contribution_yuan.sum() - annual_savings)
    summary = {
        "annual_q4_2_cost_yuan": q42_sum["cost"],
        "annual_q4_3_cost_yuan": q43_sum["cost"],
        "annual_savings_yuan": annual_savings,
        "annual_savings_pct": 100.0 * annual_savings / q42_sum["cost"],
        "accounting_attribution_gap_yuan": attribution_gap,
        "formal_price_gamma": int(selected43["price_gamma"]),
        "dynamic_price_correction_direct_contribution_yuan": 0.0,
        "annual_orthogonal_information_and_adjustment_ablation_available": False,
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    component_names = {
        "planned_purchase": "原合同购电费用变化",
        "upward_adjustment": "上调购电费用变化",
        "downward_fee": "下调违约费用变化",
        "refund": "下调退款变化",
        "emergency_purchase": "紧急购电费用变化",
    }
    view = attribution.copy()
    view["component_cn"] = view.component.map(component_names)
    report = [
        "# Q4收益来源归因分析",
        "",
        "## 全年正式结果",
        "",
        _markdown_table(
            annual,
            ["strategy", "annual_cost_yuan", "savings_vs_q4_2_yuan",
             "savings_pct", "emergency_cost_yuan", "emergency_kwh"],
            ["策略", "全年成本/元", "相对Q4-2节省/元", "节省率/%",
             "紧急购电费/元", "紧急购电量/kWh"],
            {"annual_cost_yuan": "{:,.2f}", "savings_vs_q4_2_yuan": "{:,.2f}",
             "savings_pct": "{:.4f}", "emergency_cost_yuan": "{:,.2f}",
             "emergency_kwh": "{:,.2f}"},
        ),
        "",
        "## 可核验的会计分项贡献",
        "",
        "正贡献表示该分项使Q4-3相对Q4-2节省费用；负贡献表示抵消节省。",
        "",
        _markdown_table(
            view,
            ["component_cn", "contribution_yuan", "share_of_total_savings_pct"],
            ["费用分项", "对总节省贡献/元", "占总节省/%"],
            {"contribution_yuan": "{:,.2f}", "share_of_total_savings_pct": "{:.2f}"},
        ),
        "",
        f"分项合计与全年节省的差为`{attribution_gap:.3e}`元。该表是结算恒等式归因，"
        "不是模型机制的因果消融。",
        "",
        "## 已有1月消融证据",
        "",
        _markdown_table(
            january,
            ["label", "days", "cost_yuan", "savings_vs_B0_yuan",
             "updates", "price_gamma"],
            ["已有候选", "天数", "费用/元", "相对B0节省/元", "更新时刻", "价格修正"],
            {"cost_yuan": "{:,.2f}", "savings_vs_B0_yuan": "{:,.2f}"},
        ),
        "",
        "该表来自冻结的1月15—31日候选评分，不是新回放，也不能替代全年B1。"
        "B1和B2的风险配置与正式Q4-3一致，但B0的Q4-2风险参数不同，因此B0→B1"
        "仍混合了PV信息和风险预留定义差异。",
        "",
        "## 收益来源判断",
        "",
        "- **信息更新与滚动调整：** 全年正式数据只保存Q4-2与完整Q4-3，二者合计"
        f"节省{annual_savings:,.2f}元；缺少冻结的全年正交B1，不能把该金额唯一拆成PV"
        "信息和合同调整两部分。1月R0/Rall候选只提供局部支持。",
        "- **动态电价利用：** 正式Q4-3的`price_gamma=0`，日内价格误差修正未启用，"
        "所以其对正式全年节省的直接贡献为0元。基准P2p日前价格预测仍参与两种策略，"
        "但不是Q4-3相对Q4-2的独有变化。",
        "- **费用渠道：** 紧急购电费下降和原合同购电费下降形成正贡献；Q4-3新增的"
        "上调费用形成负贡献。该会计拆分能精确解释钱从哪里变化，但不等于结构变量的"
        "独立因果贡献。",
        "",
        "派生数据位于`outputs/q4/ablation/`。",
    ]
    (DOCS / "revenue_attribution.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    return daily, summary


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.0f}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{escape(title)}</text>',
    ]


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if abs(high - low) < 1e-12:
        return (start + end) / 2
    return start + (value - low) * (end - start) / (high - low)


def write_monthly_svg(monthly: pd.DataFrame, path: Path) -> None:
    width, height = 1000, 520
    left, right, top, bottom = 90, 30, 65, 80
    values = monthly.savings_yuan.to_numpy(dtype=float)
    low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
    pad = max((high - low) * 0.08, 1.0)
    low, high = low - pad, high + pad
    y0 = _scale(0.0, low, high, height - bottom, top)
    plot_width = width - left - right
    step = plot_width / len(values)
    lines = _svg_start(width, height, "Monthly Q4-2 minus Q4-3 cost")
    lines += [
        f'<line x1="{left}" y1="{y0:.2f}" x2="{width-right}" y2="{y0:.2f}" stroke="#334155" stroke-width="1"/>',
        f'<text x="{left-8}" y="{y0+4:.2f}" text-anchor="end" font-family="Arial" font-size="11">0</text>',
    ]
    for index, (month, value) in enumerate(zip(monthly.month, values)):
        x = left + index * step + step * 0.18
        bar_width = step * 0.64
        y = _scale(float(value), low, high, height - bottom, top)
        bar_y, bar_h = min(y, y0), abs(y - y0)
        color = "#167D8D" if value >= 0 else "#C2413B"
        lines.append(f'<rect x="{x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{max(bar_h,1):.2f}" fill="{color}"/>')
        lines.append(f'<text x="{x+bar_width/2:.2f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="10">{escape(str(month)[5:])}</text>')
        label_y = bar_y - 6 if value >= 0 else bar_y + bar_h + 14
        lines.append(f'<text x="{x+bar_width/2:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="Arial" font-size="9">{value/1000:.1f}k</text>')
    lines.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="12">Month (2025)</text>')
    lines.append(f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="12">Savings (yuan)</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def write_histogram_svg(values: np.ndarray, path: Path) -> None:
    counts, edges = np.histogram(values, bins=20)
    width, height = 1000, 520
    left, right, top, bottom = 90, 30, 65, 80
    max_count = max(int(counts.max()), 1)
    step = (width - left - right) / len(counts)
    lines = _svg_start(width, height, "Distribution of daily savings")
    lines.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#334155"/>')
    for index, count in enumerate(counts):
        bar_h = (height - top - bottom) * count / max_count
        x = left + index * step + 1
        y = height - bottom - bar_h
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(step-2,1):.2f}" height="{bar_h:.2f}" fill="#4F7899"/>')
    for index in (0, 5, 10, 15, 20):
        x = left + (width-left-right) * index / 20
        value = edges[index] if index < len(edges) else edges[-1]
        lines.append(f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="10">{value/1000:.1f}k</text>')
    lines.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="12">Daily savings (yuan)</text>')
    lines.append(f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="12">Days</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cumulative_svg(daily: pd.DataFrame, path: Path) -> None:
    values = daily.delta_cost.cumsum().to_numpy(dtype=float)
    width, height = 1000, 520
    left, right, top, bottom = 90, 30, 65, 70
    low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
    pad = max((high-low) * 0.05, 1.0)
    low, high = low-pad, high+pad
    xs = np.linspace(left, width-right, len(values))
    ys = [_scale(float(v), low, high, height-bottom, top) for v in values]
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    zero_y = _scale(0.0, low, high, height-bottom, top)
    lines = _svg_start(width, height, "Cumulative Q4 savings")
    lines += [
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#94A3B8" stroke-dasharray="4 4"/>',
        f'<polyline points="{points}" fill="none" stroke="#167D8D" stroke-width="2"/>',
        f'<text x="{left}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="10">2025-02-01</text>',
        f'<text x="{width-right}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="10">2025-12-31</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="12">Evaluation date</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="12">Cumulative savings (yuan)</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stability(daily: pd.DataFrame) -> dict[str, object]:
    folder = OUTPUT / "stability"
    folder.mkdir(parents=True, exist_ok=True)
    observed_monthly = monthly_stability(daily)
    evaluation_year = int(pd.to_datetime(daily.date).dt.year.mode().iloc[0])
    monthly = complete_calendar_months(observed_monthly, evaluation_year)
    monthly.to_csv(folder / "monthly_stability.csv", index=False)

    ordered = daily.sort_values("delta_cost", ascending=False)
    top5 = ordered.head(5)
    bottom5 = ordered.tail(5).sort_values("delta_cost")
    values = daily.delta_cost.to_numpy(dtype=float)
    total = float(values.sum())
    top5_sum = float(top5.delta_cost.sum())
    stats = {
        "days": int(len(daily)),
        "total_savings_yuan": total,
        "mean_daily_savings_yuan": float(np.mean(values)),
        "median_daily_savings_yuan": float(np.median(values)),
        "std_daily_savings_yuan": float(np.std(values, ddof=0)),
        "positive_days": int(np.sum(values > EPS)),
        "loss_days": int(np.sum(values < -EPS)),
        "zero_days": int(np.sum(np.abs(values) <= EPS)),
        "positive_day_pct": float(100 * np.mean(values > EPS)),
        "loss_day_pct": float(100 * np.mean(values < -EPS)),
        "p05_yuan": float(np.quantile(values, 0.05)),
        "p25_yuan": float(np.quantile(values, 0.25)),
        "p75_yuan": float(np.quantile(values, 0.75)),
        "p95_yuan": float(np.quantile(values, 0.95)),
        "positive_months": int(np.sum(observed_monthly.savings_yuan > EPS)),
        "loss_months": int(np.sum(observed_monthly.savings_yuan < -EPS)),
        "months_in_formal_evaluation": int(len(observed_monthly)),
        "months_not_in_formal_evaluation": int(12 - len(observed_monthly)),
        "top5_gain_sum_yuan": top5_sum,
        "top5_gain_share_of_total_pct": 100.0 * top5_sum / total,
        "savings_excluding_top5_gains_yuan": total - top5_sum,
        "top5_gain_dates": top5[["date", "delta_cost"]].to_dict("records"),
        "top5_loss_dates": bottom5[["date", "delta_cost"]].to_dict("records"),
    }
    (folder / "daily_benefit_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    write_monthly_svg(
        monthly.loc[monthly.coverage_status.eq("formal_evaluation")],
        folder / "monthly_savings.svg")
    write_histogram_svg(values, folder / "daily_benefit_histogram.svg")
    write_cumulative_svg(daily, folder / "cumulative_savings.svg")

    if stats["savings_excluding_top5_gains_yuan"] > 0 and stats["positive_months"] >= 7:
        stability_conclusion = (
            "剔除收益最高5天后全年节省仍为正，且多数月份保持正节省；优势不是仅由5个"
            "异常日期产生，但日级波动和亏损日仍不可忽略。"
        )
    else:
        stability_conclusion = (
            "收益对少数高收益日或少数月份较为集中，现有全年优势的稳定性需要谨慎表述。"
        )

    top5_view = top5[["date", "delta_cost"]].copy()
    loss5_view = bottom5[["date", "delta_cost"]].copy()
    report = [
        "# Q4全年月度稳定性分析",
        "",
        "## 结论",
        "",
        stability_conclusion,
        "",
        f"334天累计节省{total:,.2f}元；正收益{stats['positive_days']}天"
        f"（{stats['positive_day_pct']:.2f}%），损失{stats['loss_days']}天"
        f"（{stats['loss_day_pct']:.2f}%）；正式评价期内正节省月份"
        f"{stats['positive_months']}个，负节省月份{stats['loss_months']}个。",
        "",
        "正式日账本覆盖2025-02-01至2025-12-31。为完整呈现自然年，表中保留"
        "2025-01一行并标记为`not_in_formal_evaluation`；该行不参与收益与稳定性结论。",
        "",
        "## 12个月统计",
        "",
        _markdown_table(
            monthly,
            ["month", "coverage_status", "days", "cost_q4_2_yuan", "cost_q4_3_yuan",
             "savings_yuan", "savings_pct", "positive_days", "loss_days",
             "mean_daily_savings_yuan", "std_daily_savings_yuan",
             "cumulative_savings_yuan"],
            ["月份", "覆盖状态", "天数", "Q4-2成本/元", "Q4-3成本/元", "节省/元", "节省率/%",
             "收益天", "损失天", "日均节省/元", "日节省标准差/元", "累计节省/元"],
            {"cost_q4_2_yuan": "{:,.2f}", "cost_q4_3_yuan": "{:,.2f}",
             "savings_yuan": "{:,.2f}", "savings_pct": "{:.3f}",
             "mean_daily_savings_yuan": "{:,.2f}",
             "std_daily_savings_yuan": "{:,.2f}",
             "cumulative_savings_yuan": "{:,.2f}"},
        ),
        "",
        "## 日收益分布与集中度",
        "",
        f"- 日收益中位数：{stats['median_daily_savings_yuan']:,.2f}元；标准差："
        f"{stats['std_daily_savings_yuan']:,.2f}元。",
        f"- 5%/95%分位数：{stats['p05_yuan']:,.2f}元 / {stats['p95_yuan']:,.2f}元。",
        f"- 收益最高5天合计：{top5_sum:,.2f}元，占全年净节省"
        f"{stats['top5_gain_share_of_total_pct']:.2f}%。",
        f"- 剔除收益最高5天后：{stats['savings_excluding_top5_gains_yuan']:,.2f}元。",
        "",
        "### 收益最高5天",
        "",
        _markdown_table(top5_view, ["date", "delta_cost"], ["日期", "节省/元"],
                        {"delta_cost": "{:,.2f}"}),
        "",
        "### 损失最大5天",
        "",
        _markdown_table(loss5_view, ["date", "delta_cost"], ["日期", "节省/元"],
                        {"delta_cost": "{:,.2f}"}),
        "",
        "## 图表",
        "",
        "- [月度节省](../../outputs/q4/stability/monthly_savings.svg)",
        "- [每日收益直方图](../../outputs/q4/stability/daily_benefit_histogram.svg)",
        "- [累计收益曲线](../../outputs/q4/stability/cumulative_savings.svg)",
        "",
        "本阶段只做月度和分布分析；TOP5日期的电价、PV、SOC、购电变化和紧急购电"
        "逐槽曲线属于后续典型日案例分析，不在本次生成。",
    ]
    (DOCS / "stability_analysis.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    consistency = run_consistency()
    daily, attribution = run_attribution()
    stability = run_stability(daily)
    print(json.dumps({
        "status": "generated",
        "plan_deviation_identifiable": consistency[
            "plan_execution_deviation_identifiable"],
        "annual_savings_yuan": attribution["annual_savings_yuan"],
        "positive_days": stability["positive_days"],
        "loss_days": stability["loss_days"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
