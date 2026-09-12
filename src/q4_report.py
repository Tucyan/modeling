"""Generate the four required Q4 date tables from a final dispatch ledger."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from q4_solver import settle


TARGET_DATES = (
    "2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"
)
SLOT_IDS = (60, 72, 84, 96, 108, 120)
SLOT_LABELS = ("10:00-10:10", "12:00-12:10", "14:00-14:10",
               "16:00-16:10", "18:00-18:10", "20:00-20:10")
EPS = 1e-7


def _fmt(value):
    return f"{float(value):.6f}"


def _time(boundary):
    minutes = 10 * int(boundary)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _events(day):
    values = day.e.to_numpy(dtype=float)
    active = np.flatnonzero(values > EPS)
    if not len(active):
        return []
    chunks = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
    return [
        f"{_time(chunk[0])}-{_time(chunk[-1] + 1)}：{_fmt(values[chunk].sum())} kWh"
        for chunk in chunks
    ]


def date_summary(dispatch, date, mode, settlement="N0"):
    """Return all paper-facing fields for one date and one Q4 mode."""
    if mode not in ("q4-2", "q4-3"):
        raise ValueError("mode must be q4-2 or q4-3")
    if settlement not in ("N0", "N1"):
        raise ValueError("settlement must be N0 or N1")
    day = dispatch[dispatch.date.astype(str) == str(date)].sort_values("interval_id")
    if len(day) != 144 or not np.array_equal(day.interval_id, np.arange(144)):
        raise ValueError(f"{date} must contain 144 ordered intervals")
    g = day.g.to_numpy(dtype=float)
    q = day.q.to_numpy(dtype=float)
    price = day.price.to_numpy(dtype=float)
    contract = settle(g, q, price, settlement)
    emergency = 5 * price * day.e.to_numpy(dtype=float)
    slot_values = [
        dict(label=label, planned=float(g[slot]), final=float(q[slot]),
             actual=float(day.actual_grid.iloc[slot]))
        for slot, label in zip(SLOT_IDS, SLOT_LABELS)
    ]
    blocks = []
    for block in range(6):
        part = day.iloc[block * 24:(block + 1) * 24]
        blocks.append(dict(
            label=f"{block * 4}:00-{(block + 1) * 4}:00",
            charge=float(part.c.sum()), discharge=float(part.d.sum())))
    return dict(
        date=str(date), mode=mode, settlement=settlement,
        slot_values=slot_values, blocks=blocks,
        planned_total=float(g.sum()), final_total=float(q.sum()),
        actual_grid_total=float(day.actual_grid.sum()),
        planned_cost=float(np.dot(price, g)),
        contract_cost=float(contract.sum()),
        emergency_cost=float(emergency.sum()),
        total_cost=float(contract.sum() + emergency.sum()),
        emergency_kwh=float(day.e.sum()),
        emergency_events=_events(day),
        initial_soc=float(day.E_start.iloc[0]),
        final_soc=float(day.E_end.iloc[-1]),
    )


def _render_summary(summary):
    lines = [
        f"#### {summary['mode']}（{summary['settlement']}，实际交付价结算）",
        "",
        "| 时间段 | 0点计划合同 g（kWh） | 最终合同 q（kWh） | 实际接收购电量（kWh） |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['label']} | {_fmt(row['planned'])} | {_fmt(row['final'])} | "
        f"{_fmt(row['actual'])} |"
        for row in summary["slot_values"])
    lines.extend([
        f"| 全天合同量 | {_fmt(summary['planned_total'])} | {_fmt(summary['final_total'])} |",
        f"| 实际接收购电量 | - | {_fmt(summary['actual_grid_total'])} |",
        f"| 0点计划合同费（元） | {_fmt(summary['planned_cost'])} | - |",
        f"| 最终合同结算费（元） | - | {_fmt(summary['contract_cost'])} |",
        f"| 紧急购电费（元） | - | {_fmt(summary['emergency_cost'])} |",
        f"| 全天总费用（元） | - | {_fmt(summary['total_cost'])} |",
        "",
        "| 时间段 | 充电量（kWh） | 放电量（kWh） |",
        "|---|---:|---:|",
    ])
    lines.extend(
        f"| {row['label']} | {_fmt(row['charge'])} | {_fmt(row['discharge'])} |"
        for row in summary["blocks"])
    lines.extend([
        "",
        f"初始储能：{_fmt(summary['initial_soc'])} kWh；末端储能：{_fmt(summary['final_soc'])} kWh。",
        f"紧急购电量：{_fmt(summary['emergency_kwh'])} kWh。",
        "紧急购电事件：" + ("；".join(summary["emergency_events"])
                           if summary["emergency_events"] else "无"),
        "",
    ])
    return lines


def render_target_tables(q42_path, q43_path, settlement="N0"):
    q42 = pd.read_csv(Path(q42_path))
    q43 = pd.read_csv(Path(q43_path))
    lines = [
        "## 四个指定日期结果（修复后N0正式账本）",
        "",
        "表中 g 为每日0点计划合同，q 为交付前最后有效合同；费用按交付区间实际电价结算。Q4-2中 g=q。",
        "",
    ]
    for date in TARGET_DATES:
        lines.append(f"### {date}")
        lines.append("")
        lines.extend(_render_summary(date_summary(q42, date, "q4-2", settlement)))
        lines.extend(_render_summary(date_summary(q43, date, "q4-3", settlement)))
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--q42", required=True)
    parser.add_argument("--q43", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--settlement", default="N0")
    args = parser.parse_args()
    Path(args.output).write_text(
        render_target_tables(args.q42, args.q43, args.settlement),
        encoding="utf-8")
