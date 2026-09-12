"""Build paper-ready Q4 figures and review prose from frozen artifacts only."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
N0 = ROOT / "outputs/q4_final/N0"
N1 = ROOT / "outputs/q4_final/N1"
DERIVED = ROOT / "outputs/q4"
FIGURES = DERIVED / "paper_figures"
REPORT = ROOT / "docs/q4/Q4_final_review_report.md"
FONT = "Microsoft YaHei,Arial,sans-serif"
BLUE = "#176B87"
ORANGE = "#D97706"
GREEN = "#16856B"
RED = "#C2413B"
SLATE = "#334155"
GRID = "#CBD5E1"
PALE = "#F8FAFC"


def select_typical_dates(daily: pd.DataFrame) -> dict[str, str]:
    """Select extrema mechanically so case choice cannot be cherry-picked."""
    ordered = daily.copy()
    ordered["date"] = ordered.date.astype(str)
    return {
        "maximum_gain": str(ordered.loc[ordered.delta_cost.idxmax(), "date"]),
        "maximum_loss": str(ordered.loc[ordered.delta_cost.idxmin(), "date"]),
    }


def annual_cost_comparison(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the frozen daily settlement table into paper-level totals."""
    q42 = float(daily.cost_q4_2.sum())
    q43 = float(daily.cost_q4_3.sum())
    emergency42 = float(daily.emergency_cost_q4_2.sum()) if "emergency_cost_q4_2" in daily else np.nan
    emergency43 = float(daily.emergency_cost_q4_3.sum()) if "emergency_cost_q4_3" in daily else np.nan
    return pd.DataFrame([
        {"strategy": "Q4-2", "annual_cost_yuan": q42,
         "savings_vs_q4_2_yuan": 0.0, "emergency_cost_yuan": emergency42},
        {"strategy": "Q4-3", "annual_cost_yuan": q43,
         "savings_vs_q4_2_yuan": q42 - q43, "emergency_cost_yuan": emergency43},
    ])


def svg_document(width: int, height: int, title: str,
                 elements: Iterable[str]) -> str:
    body = "\n".join(elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n'
        f'<title>{escape(title)}</title>\n'
        f'<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>'
    )


def _text(x: float, y: float, value: object, size: int = 13,
          anchor: str = "start", weight: int = 400, color: str = SLATE,
          rotate: float | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}"{transform}>{escape(str(value))}</text>')


def _line(x1: float, y1: float, x2: float, y2: float, color: str = GRID,
          width: float = 1.0, dash: str | None = None) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}"{dashed}/>')


def _polyline(x: np.ndarray, y: np.ndarray, color: str, width: float = 2.2,
              dash: str | None = None) -> str:
    points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y))
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round"{dashed}/>')


def _scale(values: np.ndarray, low: float, high: float,
           out_low: float, out_high: float) -> np.ndarray:
    if abs(high - low) < 1e-12:
        return np.full_like(values, (out_low + out_high) / 2, dtype=float)
    return out_low + (values - low) * (out_high - out_low) / (high - low)


def _panel_axes(x: float, y: float, width: float, height: float, title: str,
                ylabel: str, ymin: float, ymax: float,
                xlabel: str = "时刻") -> tuple[list[str], tuple[float, float, float, float]]:
    left, right, top, bottom = x + 78, x + width - 24, y + 46, y + height - 52
    parts = [
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{PALE}" rx="8"/>',
        _text(x + width / 2, y + 27, title, 15, "middle", 600),
    ]
    for frac in np.linspace(0, 1, 5):
        yy = bottom - frac * (bottom - top)
        value = ymin + frac * (ymax - ymin)
        parts.append(_line(left, yy, right, yy))
        parts.append(_text(left - 8, yy + 4, f"{value:,.1f}", 10, "end"))
    for hour in (0, 6, 12, 18, 24):
        xx = left + (hour / 24) * (right - left)
        parts.append(_line(xx, top, xx, bottom, "#E2E8F0"))
        parts.append(_text(xx, bottom + 18, str(hour), 10, "middle"))
    parts.extend([
        _line(left, bottom, right, bottom, SLATE),
        _line(left, top, left, bottom, SLATE),
        _text((left + right) / 2, y + height - 12, xlabel, 11, "middle"),
        _text(x + 18, (top + bottom) / 2, ylabel, 11, "middle", rotate=-90),
    ])
    return parts, (left, right, top, bottom)


def _save_svg(path: Path, width: int, height: int, title: str,
              parts: list[str]) -> None:
    path.write_text(svg_document(width, height, title, parts), encoding="utf-8")


def write_annual_cost_figure(annual: pd.DataFrame, path: Path) -> None:
    width, height = 1200, 610
    parts = [_text(width / 2, 34, "Q4正式评价期成本对比（N0）", 23, "middle", 700)]
    panels = [
        (55, "总运行费用", "annual_cost_yuan", "万元", 1 / 10000),
        (625, "紧急购电费用", "emergency_cost_yuan", "万元", 1 / 10000),
    ]
    colors = [BLUE, ORANGE]
    for panel_x, title, column, unit, factor in panels:
        x0, y0, pw, ph = panel_x, 70, 520, 475
        parts.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="{PALE}" rx="10"/>')
        parts.append(_text(x0 + pw / 2, y0 + 32, title, 17, "middle", 600))
        values = annual[column].to_numpy(dtype=float) * factor
        maximum = float(np.nanmax(values)) * 1.14
        base, top = y0 + ph - 70, y0 + 70
        for tick in np.linspace(0, maximum, 5):
            yy = base - tick / maximum * (base - top)
            parts.append(_line(x0 + 70, yy, x0 + pw - 25, yy))
            parts.append(_text(x0 + 62, yy + 4, f"{tick:,.0f}", 10, "end"))
        for index, row in annual.reset_index(drop=True).iterrows():
            value = float(row[column]) * factor
            bar_x = x0 + 145 + index * 180
            bar_h = value / maximum * (base - top)
            parts.append(f'<rect x="{bar_x}" y="{base-bar_h:.2f}" width="92" height="{bar_h:.2f}" fill="{colors[index]}" rx="4"/>')
            parts.append(_text(bar_x + 46, base - bar_h - 10, f"{value:,.2f}", 12, "middle", 600, colors[index]))
            parts.append(_text(bar_x + 46, base + 24, row.strategy, 13, "middle", 600))
        parts.append(_text(x0 + 18, (top + base) / 2, unit, 11, "middle", rotate=-90))
    savings = float(annual.loc[annual.strategy.eq("Q4-3"), "savings_vs_q4_2_yuan"].iloc[0])
    parts.append(_text(width / 2, 585, f"Q4-3相对Q4-2节省 {savings:,.2f} 元（1.5204%）", 15, "middle", 600, GREEN))
    _save_svg(path, width, height, "Q4 annual cost comparison", parts)


def write_monthly_figure(monthly: pd.DataFrame, path: Path) -> None:
    observed = monthly.loc[monthly.coverage_status.eq("formal_evaluation")].copy()
    values = observed.savings_yuan.to_numpy(dtype=float) / 1000.0
    width, height = 1200, 620
    left, right, top, bottom = 90, 1160, 80, 525
    low, high = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    pad = max((high - low) * 0.12, 5.0)
    low, high = low - pad, high + pad
    zero_y = float(_scale(np.array([0.0]), low, high, bottom, top)[0])
    parts = [_text(width / 2, 35, "Q4-3相对Q4-2月度节省", 23, "middle", 700)]
    for tick in np.linspace(low, high, 6):
        yy = float(_scale(np.array([tick]), low, high, bottom, top)[0])
        parts.append(_line(left, yy, right, yy))
        parts.append(_text(left - 10, yy + 4, f"{tick:,.0f}", 10, "end"))
    parts.append(_line(left, zero_y, right, zero_y, SLATE, 1.4))
    step = (right - left) / len(values)
    for index, (month, value) in enumerate(zip(observed.month, values)):
        x = left + index * step + step * 0.16
        yv = float(_scale(np.array([value]), low, high, bottom, top)[0])
        y, bar_h = min(yv, zero_y), abs(yv - zero_y)
        color = GREEN if value >= 0 else RED
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{step*0.68:.2f}" height="{max(bar_h,1):.2f}" fill="{color}"/>')
        label_y = y - 7 if value >= 0 else y + bar_h + 15
        parts.append(_text(x + step * 0.34, label_y, f"{value:,.1f}", 10, "middle", 600, color))
        parts.append(_text(x + step * 0.34, bottom + 22, str(month)[5:], 11, "middle"))
    parts.extend([
        _text(width / 2, height - 30, "月份（2025年；1月不在正式评价期）", 12, "middle"),
        _text(24, (top + bottom) / 2, "节省（千元）", 12, "middle", rotate=-90),
        _text(width - 30, 57, "绿色：节省；红色：成本增加", 11, "end"),
    ])
    _save_svg(path, width, height, "Monthly Q4 savings", parts)


def write_distribution_figure(daily: pd.DataFrame, path: Path) -> None:
    values = daily.delta_cost.to_numpy(dtype=float) / 1000.0
    counts, edges = np.histogram(values, bins=24)
    width, height = 1200, 620
    left, right, top, bottom = 90, 1160, 78, 520
    maximum = int(counts.max()) * 1.15
    parts = [_text(width / 2, 35, "每日节省金额分布（334天）", 23, "middle", 700)]
    for tick in np.linspace(0, maximum, 6):
        yy = bottom - tick / maximum * (bottom - top)
        parts.append(_line(left, yy, right, yy))
        parts.append(_text(left - 10, yy + 4, f"{tick:.0f}", 10, "end"))
    step = (right - left) / len(counts)
    for index, count in enumerate(counts):
        center = (edges[index] + edges[index + 1]) / 2
        bar_h = count / maximum * (bottom - top)
        color = GREEN if center >= 0 else RED
        parts.append(f'<rect x="{left+index*step+1:.2f}" y="{bottom-bar_h:.2f}" width="{max(step-2,1):.2f}" height="{bar_h:.2f}" fill="{color}" opacity="0.88"/>')
    zero_x = float(_scale(np.array([0.0]), float(edges[0]), float(edges[-1]), left, right)[0])
    median_x = float(_scale(np.array([np.median(values)]), float(edges[0]), float(edges[-1]), left, right)[0])
    parts.extend([
        _line(zero_x, top, zero_x, bottom, SLATE, 1.4),
        _line(median_x, top, median_x, bottom, ORANGE, 2, "6 4"),
        _text(median_x + 5, top + 20, f"中位数 {np.median(values):.3f}千元", 11, "start", 600, ORANGE),
        _text(width / 2, height - 30, "每日节省 = Q4-2成本 − Q4-3成本（千元）", 12, "middle"),
        _text(22, (top + bottom) / 2, "天数", 12, "middle", rotate=-90),
    ])
    _save_svg(path, width, height, "Daily savings distribution", parts)


def _day_frame(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    selected = frame.loc[frame.date.astype(str).eq(date)].copy()
    return selected.sort_values("interval_id").reset_index(drop=True)


def write_soc_figure(q42: pd.DataFrame, q43: pd.DataFrame,
                     typical: dict[str, str], path: Path) -> None:
    width, height = 1500, 760
    parts = [_text(width / 2, 34, "典型日期储能SOC实际执行轨迹", 23, "middle", 700)]
    cases = [("maximum_gain", "最大收益日"), ("maximum_loss", "最大损失日")]
    for index, (key, label) in enumerate(cases):
        date = typical[key]
        a, b = _day_frame(q42, date), _day_frame(q43, date)
        values = np.concatenate([a.E_end.to_numpy(float), b.E_end.to_numpy(float)])
        low, high = float(values.min()), float(values.max())
        pad = max((high - low) * 0.08, 100.0)
        panel_x = 35 + index * 730
        axes, (left, right, top, bottom) = _panel_axes(
            panel_x, 70, 700, 610, f"{label}：{date}", "SOC（kWh）", low - pad, high + pad)
        parts.extend(axes)
        slots = np.arange(144)
        xx = _scale(slots, 0, 143, left, right)
        parts.append(_polyline(xx, _scale(a.E_end.to_numpy(float), low-pad, high+pad, bottom, top), BLUE))
        parts.append(_polyline(xx, _scale(b.E_end.to_numpy(float), low-pad, high+pad, bottom, top), ORANGE))
        parts.extend([
            _line(left + 20, top + 18, left + 55, top + 18, BLUE, 3),
            _text(left + 62, top + 22, "Q4-2", 11),
            _line(left + 125, top + 18, left + 160, top + 18, ORANGE, 3),
            _text(left + 167, top + 22, "Q4-3", 11),
        ])
    _save_svg(path, width, height, "SOC trajectories on typical dates", parts)


def _plot_series(parts: list[str], frame: pd.DataFrame, columns: list[str],
                 colors: list[str], labels: list[str], panel_x: float, panel_y: float,
                 panel_w: float, panel_h: float, title: str, ylabel: str,
                 dashed: set[int] | None = None) -> None:
    dashed = dashed or set()
    arrays = [frame[column].to_numpy(float) for column in columns]
    low = min(float(array.min()) for array in arrays)
    high = max(float(array.max()) for array in arrays)
    if low >= 0:
        low = 0.0
    pad = max((high - low) * 0.08, 1e-3)
    axes, (left, right, top, bottom) = _panel_axes(
        panel_x, panel_y, panel_w, panel_h, title, ylabel, low - pad, high + pad)
    parts.extend(axes)
    slots = np.arange(len(frame))
    xx = _scale(slots, 0, max(len(frame) - 1, 1), left, right)
    for index, array in enumerate(arrays):
        yy = _scale(array, low - pad, high + pad, bottom, top)
        parts.append(_polyline(xx, yy, colors[index], 2.0, "7 4" if index in dashed else None))
    legend_x = left + 12
    for index, (label, color) in enumerate(zip(labels, colors)):
        lx = legend_x + index * 130
        parts.append(_line(lx, top + 17, lx + 28, top + 17, color, 2.5,
                           "7 4" if index in dashed else None))
        parts.append(_text(lx + 34, top + 21, label, 10))


def write_typical_figure(q42: pd.DataFrame, q43: pd.DataFrame,
                         typical: dict[str, str], path: Path) -> None:
    width, height = 1540, 1420
    parts = [_text(width / 2, 34, "最大收益日与最大损失日运行特征", 23, "middle", 700)]
    cases = [("maximum_gain", "最大收益日"), ("maximum_loss", "最大损失日")]
    for column_index, (key, case_label) in enumerate(cases):
        date = typical[key]
        a, b = _day_frame(q42, date), _day_frame(q43, date)
        frame = pd.DataFrame({
            "price": a.price,
            "pv": a.pv,
            "load": a.load,
            "q42": a.q,
            "q43": b.q,
            "adjustment": b.q - b.g,
            "e42": a.e,
            "e43": b.e,
        })
        x = 30 + column_index * 755
        parts.append(_text(x + 362, 67, f"{case_label}：{date}", 17, "middle", 700,
                           GREEN if key == "maximum_gain" else RED))
        _plot_series(parts, frame, ["price"], [SLATE], ["实际结算价"],
                     x, 85, 725, 300, "电价曲线", "元/kWh")
        _plot_series(parts, frame, ["pv", "load"], [GREEN, SLATE], ["PV", "负载"],
                     x, 405, 725, 300, "光伏与负载", "kWh/槽")
        _plot_series(parts, frame, ["q42", "q43", "adjustment"],
                     [BLUE, ORANGE, RED], ["Q4-2购电", "Q4-3购电", "Q4-3调整"],
                     x, 725, 725, 300, "合同购电与调整", "kWh/槽", {2})
        _plot_series(parts, frame, ["e42", "e43"], [BLUE, ORANGE],
                     ["Q4-2紧急", "Q4-3紧急"], x, 1045, 725, 300,
                     "紧急购电", "kWh/槽")
    parts.append(_text(width / 2, 1395,
                       "典型日期由334天日收益极值自动选择；图中实际价格仅用于事后结算展示。",
                       12, "middle"))
    _save_svg(path, width, height, "Typical-date operating analysis", parts)


def _format_table(frame: pd.DataFrame, columns: list[str], labels: list[str],
                  decimals: dict[str, int] | None = None) -> str:
    decimals = decimals or {}
    lines = ["| " + " | ".join(labels) + " |",
             "|" + "|".join(["---"] * len(labels)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        formatted = []
        for column, value in zip(columns, row):
            if pd.isna(value):
                formatted.append("—")
            elif column in decimals:
                formatted.append(f"{float(value):,.{decimals[column]}f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def _load_frozen() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(DERIVED / "ablation/daily_savings.csv")
    q42 = pd.read_csv(N0 / "q4-2_dispatch_10min.csv")
    q43 = pd.read_csv(N0 / "q4-3_dispatch_10min.csv")
    return daily, q42, q43


def _n1_rows() -> dict[str, dict[str, float]]:
    rows = json.loads((N1 / "summary.json").read_text(encoding="utf-8"))
    return {row["mode"]: row for row in rows}


def _write_report(annual: pd.DataFrame, monthly: pd.DataFrame,
                  typical: dict[str, str], typical_summary: pd.DataFrame) -> None:
    q42_cost = float(annual.loc[annual.strategy.eq("Q4-2"), "annual_cost_yuan"].iloc[0])
    q43_cost = float(annual.loc[annual.strategy.eq("Q4-3"), "annual_cost_yuan"].iloc[0])
    q42_emergency = float(annual.loc[annual.strategy.eq("Q4-2"), "emergency_cost_yuan"].iloc[0])
    q43_emergency = float(annual.loc[annual.strategy.eq("Q4-3"), "emergency_cost_yuan"].iloc[0])
    savings = q42_cost - q43_cost
    stability = json.loads((DERIVED / "stability/daily_benefit_statistics.json").read_text(encoding="utf-8"))
    attribution = pd.read_csv(DERIVED / "ablation/annual_accounting_attribution.csv")
    january = pd.read_csv(DERIVED / "ablation/existing_january_ablation.csv")
    consistency = json.loads((DERIVED / "consistency/summary.json").read_text(encoding="utf-8"))
    n1 = _n1_rows()
    n1_q43 = float(n1["q4-3"]["cost"])
    n1_savings = float(n1["q4-2"]["cost"]) - n1_q43

    annual_view = annual.copy()
    annual_view["savings_pct"] = 100 * annual_view.savings_vs_q4_2_yuan / q42_cost
    monthly_view = monthly.loc[monthly.coverage_status.eq("formal_evaluation")].copy()
    attr_labels = {
        "planned_purchase": "原合同购电费用",
        "upward_adjustment": "上调购电费用",
        "downward_fee": "下调违约费用",
        "refund": "下调退款",
        "emergency_purchase": "紧急购电费用",
    }
    attribution["component_cn"] = attribution.component.map(attr_labels)

    report = [
        "# Q4最终审查与论文支撑报告",
        "",
        "## 摘要",
        "",
        "本报告在Q4正式模型完全冻结的前提下，整合规划—执行一致性、收益归因、"
        "全年稳定性和已有敏感性证据。正式评价期为2025-02-01至2025-12-31，共334天、"
        "48,096个10分钟时段。Q4-3相对Q4-2降低费用"
        f"{savings:,.2f}元，降幅{100*savings/q42_cost:.4f}%。该优势可以由正式账本"
        "复算，但日级与月级表现存在波动，论文应将其表述为全年平均经济性改善，而不是"
        "逐日一致占优。",
        "",
        "## 一、模型保持说明",
        "",
        "本阶段属于模型外部验证增强。Q4目标函数、物理约束、风险参数、优化流程、"
        "F0实际执行控制器和正式结果均未改变，也未重新训练预测模型。Q4-2仍采用P2p价格"
        "预测和风险参数0.80；Q4-3仍采用风险参数(0.70, 0.65)，并仅在6:00、12:00和"
        "18:00使用当时已发布预测调整剩余合同量。正式`price_gamma=0`，未来真实价格未"
        "参与决策；账本中的实际价格仅用于交付后的费用结算和本报告的事后展示。",
        "",
        "正式结果目录`outputs/q4_final/`保持只读。本报告、图表和中间统计均写入"
        "`docs/q4/`、`analysis/q4_review/`和`outputs/q4/paper_figures/`。",
        "",
        "## 二、执行一致性验证",
        "",
        "源码审计确认优化器内部能够生成计划充电、计划放电和计划SOC，但正式结果只持久化"
        "合同购电量，未保存计划储能轨迹。因此，冻结历史证据不能计算"
        "`actual_charge-charge_plan`、`actual_discharge-discharge_plan`和"
        "`actual_SOC-SOC_plan`，也不能将这些偏差宣称为零。新增分析接口可以在未来获得"
        "独立计划日志后执行比较，本次没有调用求解器补造历史轨迹。",
        "",
        "可独立验证的是实际执行账本与F0规则的一致性。分析程序未导入Q4 solver，而是独立"
        "重写单时段F0公式，对Q4-2和Q4-3各48,096个时段逐槽复算充电、放电、紧急购电、"
        "弃电及SOC。结果如下。",
        "",
        "| 策略 | 复核时段 | 平均绝对误差 | 最大绝对误差 | SOC最大误差 |",
        "|---|---:|---:|---:|---:|",
        f"| Q4-2 | 48,096 | {consistency['modes']['q4-2']['mean_absolute_error']:.3e} | "
        f"{consistency['modes']['q4-2']['max_absolute_error']:.3e} | "
        f"{consistency['modes']['q4-2']['max_soc_end_error']:.3e} |",
        f"| Q4-3 | 48,096 | {consistency['modes']['q4-3']['mean_absolute_error']:.3e} | "
        f"{consistency['modes']['q4-3']['max_absolute_error']:.3e} | "
        f"{consistency['modes']['q4-3']['max_soc_end_error']:.3e} |",
        "",
        "该结果证明正式账本忠实执行既定F0规则，但不等价于证明优化计划储能轨迹与F0轨迹"
        "一致。最终费用直接由实际合同、紧急购电和结算价复算，因此计划轨迹缺失不破坏"
        "费用对账结论，却构成机制解释方面的证据局限。",
        "",
        "![图4 典型日期SOC轨迹](../../outputs/q4/paper_figures/figure_4_soc_trajectories.svg)",
        "",
        "**图4说明：** 最大收益日和最大损失日均由334天日收益极值自动确定；曲线为正式"
        "账本中的实际SOC，不是未保存的计划SOC。",
        "",
        "## 三、经济收益来源分析",
        "",
        _format_table(annual_view,
                      ["strategy", "annual_cost_yuan", "savings_vs_q4_2_yuan",
                       "savings_pct", "emergency_cost_yuan"],
                      ["策略", "全年费用/元", "相对Q4-2节省/元", "节省率/%", "紧急购电费/元"],
                      {"annual_cost_yuan": 2, "savings_vs_q4_2_yuan": 2,
                       "savings_pct": 4, "emergency_cost_yuan": 2}),
        "",
        "![图1 全年成本对比](../../outputs/q4/paper_figures/figure_1_annual_cost_comparison.svg)",
        "",
        "Q4-3的净节省不是单一费用项造成。按正式结算恒等式分解，正贡献表示该项使Q4-3"
        "相对Q4-2更便宜，负贡献表示抵消节省。",
        "",
        _format_table(attribution, ["component_cn", "contribution_yuan",
                                    "share_of_total_savings_pct"],
                      ["费用分项", "贡献/元", "占净节省/%"],
                      {"contribution_yuan": 2, "share_of_total_savings_pct": 2}),
        "",
        f"原合同购电费用下降贡献203,378.32元，紧急购电费用由{q42_emergency:,.2f}元降至"
        f"{q43_emergency:,.2f}元，贡献328,261.07元；与此同时，合同上调费用增加"
        "308,831.92元，抵消部分收益。五个分项与净节省的对账误差低于1e-6元。",
        "",
        "该分解回答的是费用从哪里变化，不能替代机制因果消融。全年正式数据只有Q4-2与完整"
        "Q4-3，缺少正交的全年“仅信息更新、不调整合同”策略，因此无法把信息更新与滚动调整"
        "拆成唯一比例。冻结的1月15—31日候选显示，相同Q4-3风险配置下，Rall/γ=0成本为"
        f"{float(january.loc[january.label.eq('B2_Q4-3_Rall_gamma0'), 'cost_yuan'].iloc[0]):,.2f}元，"
        "R0成本更高；但这是17天局部证据，且不能外推成全年独立贡献。正式"
        "`price_gamma=0`，所以日内价格误差修正对正式全年节省的直接贡献为0。",
        "",
        "## 四、全年稳定性分析",
        "",
        f"334天中，Q4-3在{stability['positive_days']}天节省费用，占"
        f"{stability['positive_day_pct']:.2f}%；在{stability['loss_days']}天费用更高，占"
        f"{stability['loss_day_pct']:.2f}%。正式评价期11个月中，正节省月份"
        f"{stability['positive_months']}个、负节省月份{stability['loss_months']}个。1月不在"
        "正式评价期，不填造成本。",
        "",
        _format_table(monthly_view, ["month", "days", "savings_yuan", "savings_pct",
                                     "positive_days", "loss_days", "cumulative_savings_yuan"],
                      ["月份", "天数", "节省/元", "节省率/%", "收益天", "损失天", "累计节省/元"],
                      {"savings_yuan": 2, "savings_pct": 3,
                       "cumulative_savings_yuan": 2}),
        "",
        "![图2 月度收益](../../outputs/q4/paper_figures/figure_2_monthly_savings.svg)",
        "",
        "![图3 日收益分布](../../outputs/q4/paper_figures/figure_3_daily_savings_distribution.svg)",
        "",
        f"日收益中位数为{stability['median_daily_savings_yuan']:,.2f}元，标准差为"
        f"{stability['std_daily_savings_yuan']:,.2f}元。收益最高5天合计"
        f"{stability['top5_gain_sum_yuan']:,.2f}元，占全年净节省"
        f"{stability['top5_gain_share_of_total_pct']:.2f}%；剔除这5天后仍节省"
        f"{stability['savings_excluding_top5_gains_yuan']:,.2f}元。因此优势并非完全由5天"
        "产生，但收益集中度较高，不能宣称逐日稳定占优。",
        "",
        "![图5 典型日期分析](../../outputs/q4/paper_figures/figure_5_typical_dates.svg)",
        "",
        _format_table(typical_summary,
                      ["case", "date", "cost_q4_2", "cost_q4_3", "delta_cost",
                       "emergency_kwh_q4_2", "emergency_kwh_q4_3"],
                      ["案例", "日期", "Q4-2费用/元", "Q4-3费用/元", "节省/元",
                       "Q4-2紧急购电/kWh", "Q4-3紧急购电/kWh"],
                      {"cost_q4_2": 2, "cost_q4_3": 2, "delta_cost": 2,
                       "emergency_kwh_q4_2": 2, "emergency_kwh_q4_3": 2}),
        "",
        "典型日期不是人工挑选，而是固定规则选择最大收益日"
        f"{typical['maximum_gain']}和最大损失日{typical['maximum_loss']}。图中实际价格仅作为"
        "事后结算环境展示，不参与当时决策。",
        "",
        "## 五、敏感性分析",
        "",
        "### 5.1 结算规则敏感性",
        "",
        "| 结算规则 | Q4-2费用/元 | Q4-3费用/元 | Q4-3相对节省/元 | 节省率/% |",
        "|---|---:|---:|---:|---:|",
        f"| N0 | {q42_cost:,.2f} | {q43_cost:,.2f} | {savings:,.2f} | {100*savings/q42_cost:.4f} |",
        f"| N1 | {float(n1['q4-2']['cost']):,.2f} | {n1_q43:,.2f} | {n1_savings:,.2f} | "
        f"{100*n1_savings/float(n1['q4-2']['cost']):.4f} |",
        "",
        "Q4-3在N0和N1两种已有结算规则下均优于Q4-2，说明优势方向对结算口径具有一定"
        "稳健性。但N1费用较低来自结算规则变化，不应被解释为模型结构改进。",
        "",
        "### 5.2 风险参数与更新时间",
        "",
        "正式参数由1月15—31日的因果影子回测预先选择，2—12月未参与调参。已有候选覆盖"
        "Q4-2风险分位数0.50、0.70、0.80、0.90，Q4-3多组(α0, αupdate)、8种更新时间"
        "集合和γ∈{0,1}。这些结果支持所选配置并非未经比较的单点设定，但评价窗口仅17天，"
        "不能等价于跨年度参数稳定性。",
        "",
        "### 5.3 终端SOC",
        "",
        "正式规划采用6,000 kWh终端目标，正式回放在评价末日闭合。当前没有软约束或48小时"
        "跨日窗口的冻结结果；遵守本阶段禁止修改模型和禁止大规模重算的约束，本报告不为"
        "这两种方案填造成本。论文可将其列为后续实验设计：保持其余参数不变，分别比较硬"
        "约束、绝对偏差惩罚和48小时滚动窗口的费用、末端SOC偏差与运行时间。",
        "",
        "## 六、模型局限性",
        "",
        "1. 计划充放电和计划SOC未持久化，无法对历史计划—执行偏差作直接定量验证。",
        "2. 全年正式结果缺少正交B1，信息更新与合同调整的因果贡献不能唯一拆分。",
        "3. 价格模型P0p/P1p没有对应的全年调度成本，因此不能证明预测MAE排序必然等于经济"
        "成本排序。",
        "4. 收益存在集中性：44.01%的评价日费用上升，5个高收益日贡献63.05%的净节省。",
        "5. 正式证据来自2025年单年度因果回测，尚不能替代跨年度或不同市场环境的外部验证。",
        "6. 终端SOC软约束与48小时窗口只有实验设计，没有现成结果。",
        "7. 实际价格、实际PV和实际负载在本报告中用于事后结算和解释；这证明信息边界合理，"
        "但仍应与真实在线部署的预测漂移风险区分。",
        "",
        "## 七、论文写作建议",
        "",
        "### 建议保留的正文内容",
        "",
        "1. 用图1及全年成本表作为主结果，报告绝对节省、相对节省和紧急购电下降。",
        "2. 用会计分项表解释收益渠道，并明确它不是机制因果消融。",
        "3. 用图2、图3及TOP5集中度说明优势的月度、日级和尾部稳定性。",
        "4. 用图4说明实际SOC轨迹受物理边界约束，同时在图注中写明它是执行轨迹而非计划轨迹。",
        "5. 用图5同时展示最大收益和最大损失案例，避免只展示有利日期。",
        "6. 在稳健性小节保留N0/N1比较；终端SOC替代方案只放在局限性或未来工作，不报告"
        "未经运行的数字。",
        "",
        "### 推荐的论文表述",
        "",
        "> 在334天因果回测中，Q4-3相对Q4-2降低总运行费用222,807.47元（1.5204%）。"
        "费用分解表明，收益主要由原合同购电费用和紧急购电费用下降形成，但部分收益被"
        "合同上调费用抵消。Q4-3在55.99%的日期和6个评价月份中占优；剔除最高收益5天后"
        "仍保持82,324.47元净节省，说明全年优势不完全依赖极少数日期，但其日级与月级"
        "波动仍需谨慎解释。",
        "",
        "### 不建议使用的表述",
        "",
        "- 不写“规划与执行完全一致”；应写“实际账本与F0规则一致，计划轨迹未落盘”。",
        "- 不把222,807.47元全部归因于PV预测更新或动态价格；正式动态价格修正贡献为0。",
        "- 不写“所有月份稳定节省”或“每日均占优”。",
        "- 不报告软终端约束、48小时窗口或P0p/P1p全年经济成本的虚构数值。",
        "",
        "## 八、论文图表索引",
        "",
        "| 图号 | 文件 | 建议位置 |",
        "|---|---|---|",
        "| 图1 | `figure_1_annual_cost_comparison.svg` | Q4主结果 |",
        "| 图2 | `figure_2_monthly_savings.svg` | 稳定性分析 |",
        "| 图3 | `figure_3_daily_savings_distribution.svg` | 尾部与分布分析 |",
        "| 图4 | `figure_4_soc_trajectories.svg` | 执行一致性与物理解释 |",
        "| 图5 | `figure_5_typical_dates.svg` | 典型日案例 |",
        "",
        "全部图表均由冻结账本或已核验派生CSV生成，未调用Q4 solver。SVG为矢量格式，可直接"
        "插入论文或转换为PDF。",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    daily, q42, q43 = _load_frozen()
    typical = select_typical_dates(daily)
    annual = annual_cost_comparison(daily)
    annual.to_csv(FIGURES / "annual_cost_comparison.csv", index=False)

    monthly = pd.read_csv(DERIVED / "stability/monthly_stability.csv")
    write_annual_cost_figure(annual, FIGURES / "figure_1_annual_cost_comparison.svg")
    write_monthly_figure(monthly, FIGURES / "figure_2_monthly_savings.svg")
    write_distribution_figure(daily, FIGURES / "figure_3_daily_savings_distribution.svg")
    write_soc_figure(q42, q43, typical, FIGURES / "figure_4_soc_trajectories.svg")
    write_typical_figure(q42, q43, typical, FIGURES / "figure_5_typical_dates.svg")

    summary_rows = []
    case_labels = {"maximum_gain": "最大收益日", "maximum_loss": "最大损失日"}
    for key, date in typical.items():
        row = daily.loc[daily.date.astype(str).eq(date)].iloc[0]
        a, b = _day_frame(q42, date), _day_frame(q43, date)
        summary_rows.append({
            "case": case_labels[key], "date": date,
            "cost_q4_2": float(row.cost_q4_2), "cost_q4_3": float(row.cost_q4_3),
            "delta_cost": float(row.delta_cost),
            "emergency_kwh_q4_2": float(a.e.sum()),
            "emergency_kwh_q4_3": float(b.e.sum()),
            "soc_min_q4_2": float(a.E_end.min()), "soc_max_q4_2": float(a.E_end.max()),
            "soc_min_q4_3": float(b.E_end.min()), "soc_max_q4_3": float(b.E_end.max()),
            "q4_3_absolute_adjustment_kwh": float(np.abs(b.q - b.g).sum()),
        })
    typical_summary = pd.DataFrame(summary_rows)
    typical_summary.to_csv(FIGURES / "typical_dates_summary.csv", index=False)

    figure_files = [
        "figure_1_annual_cost_comparison.svg",
        "figure_2_monthly_savings.svg",
        "figure_3_daily_savings_distribution.svg",
        "figure_4_soc_trajectories.svg",
        "figure_5_typical_dates.svg",
    ]
    captions = [
        "Q4-2与Q4-3正式评价期总成本及紧急购电费用对比",
        "Q4-3相对Q4-2的月度节省",
        "334天每日节省金额分布",
        "最大收益日与最大损失日SOC实际执行轨迹",
        "最大收益日与最大损失日的电价、PV、购电调整和紧急购电",
    ]
    manifest = {
        "generation_policy": "read-only frozen artifacts; no solver calls",
        "formal_evaluation": {"start": "2025-02-01", "end": "2025-12-31",
                              "days": 334, "slots_per_strategy": 48096},
        "typical_dates": typical,
        "figures": [{"file": file, "caption": caption}
                    for file, caption in zip(figure_files, captions)],
        "source_files": [
            "outputs/q4_final/N0/q4-2_dispatch_10min.csv",
            "outputs/q4_final/N0/q4-3_dispatch_10min.csv",
            "outputs/q4/ablation/daily_savings.csv",
            "outputs/q4/stability/monthly_stability.csv",
        ],
    }
    (FIGURES / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(annual, monthly, typical, typical_summary)
    print(json.dumps({"status": "generated", "figures": len(figure_files),
                      "typical_dates": typical,
                      "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
