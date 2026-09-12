"""Independent checks for frozen-Q4 consistency, attribution and stability outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage2_analysis import (
    accounting_attribution,
    compare_plan_execution,
    f0_step_independent,
    monthly_stability,
)


ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-6
PROTECTED_HASHES = {
    "src/q4_solver.py": "45144C1861D25D7967F892A7249DE6B6435D105D213F83FDA540F7B251EE22CC",
    "src/q4_experiments.py": "B1BE3B15B2FEFC06B8592A5D116B7C6B08E94913923AF4FF0FCF2E39BE56D2F2",
    "src/q4_forecast.py": "3C6C86714B2F911E2D2F09190A32D5BE541D77165F72383A3836AC3A23D0D23F",
    "src/dispatch_control.py": "46D1F176F4B7B6E065E4A9F38B38831ACD8D676CEB1C2108E33B6CB4C5FD8C9E",
    "outputs/q4_final/N0/q4-2_dispatch_10min.csv": "3343F44B2F265E69ACB5D0A41A1458A7F616D29DABA1D854B5CA4DCC1F812C75",
    "outputs/q4_final/N0/q4-3_dispatch_10min.csv": "0211A576FB6EE5E17CC54C026CD93E9CFB9BEE511E95FAC6B270BA49BCC0D7D1",
    "outputs/q4_final/result4-2.xlsx": "83B18F8AD095D81C0AE85F5FF326CCC274C56CA04BF5AA2E8100712123E0F669",
    "outputs/q4_final/result4-3.xlsx": "9D44EC88FD8DDC3BFD553AFFC59267551A07B966E6FBEB7AA434C97F41263B33",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def self_test() -> None:
    surplus = f0_step_independent(
        contract=100.0,
        load=80.0,
        pv=0.0,
        state=6000.0,
        remaining_slots=10,
        closing_target=None,
    )
    np.testing.assert_allclose(
        [surplus["c"], surplus["d"], surplus["e"], surplus["next_state"]],
        [20.0, 0.0, 0.0, 6018.0],
        atol=1e-10,
        rtol=0,
    )

    shortage = f0_step_independent(
        contract=0.0,
        load=100.0,
        pv=0.0,
        state=1200.0,
        remaining_slots=10,
        closing_target=None,
    )
    np.testing.assert_allclose(
        [shortage["c"], shortage["d"], shortage["e"], shortage["next_state"]],
        [0.0, 0.0, 100.0, 1200.0],
        atol=1e-10,
        rtol=0,
    )

    actual = pd.DataFrame(
        {"date": ["2025-02-01"], "interval_id": [0], "c": [2.0],
         "d": [1.0], "E_end": [6001.0]}
    )
    missing = compare_plan_execution(pd.DataFrame(), actual)
    assert missing["identifiable"] is False
    assert set(missing["missing_plan_columns"]) == {
        "date", "interval_id", "charge_plan", "discharge_plan", "SOC_plan"
    }

    plan = pd.DataFrame(
        {"date": ["2025-02-01"], "interval_id": [0],
         "charge_plan": [1.5], "discharge_plan": [0.5], "SOC_plan": [6000.0]}
    )
    compared = compare_plan_execution(plan, actual)
    assert compared["identifiable"] is True
    np.testing.assert_allclose(
        compared["rows"][["delta_charge", "delta_discharge", "delta_SOC"]],
        [[0.5, 0.5, 1.0]], atol=1e-12, rtol=0)

    q42 = dict(plan_cost=100.0, up_cost=0.0, down_fee=0.0,
               refund=0.0, emergency_cost=50.0, cost=150.0)
    q43 = dict(plan_cost=90.0, up_cost=20.0, down_fee=0.0,
               refund=0.0, emergency_cost=10.0, cost=120.0)
    attribution = accounting_attribution(q42, q43)
    assert abs(float(attribution.contribution_yuan.sum()) - 30.0) < 1e-12
    assert float(attribution.loc[
        attribution.component.eq("emergency_purchase"), "contribution_yuan"
    ].iloc[0]) == 40.0

    daily = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "cost_q4_2": [100.0, 100.0], "cost_q4_3": [90.0, 105.0],
        "delta_cost": [10.0, -5.0],
    })
    monthly = monthly_stability(daily)
    assert len(monthly) == 1
    assert float(monthly.iloc[0].savings_yuan) == 5.0
    assert int(monthly.iloc[0].positive_days) == 1
    assert int(monthly.iloc[0].loss_days) == 1
    print("SELF_TEST=PASS")


def verify_outputs() -> None:
    consistency = ROOT / "outputs/q4/consistency"
    ablation = ROOT / "outputs/q4/ablation"
    stability = ROOT / "outputs/q4/stability"

    residuals = pd.read_csv(consistency / "f0_replay_residuals.csv")
    if len(residuals) != 2 * 334 * 144:
        raise AssertionError("F0 replay does not cover two complete N0 ledgers")
    error_columns = [
        "charge_error", "discharge_error", "emergency_error",
        "unused_error", "spill_error", "soc_end_error",
    ]
    max_f0_error = float(np.abs(residuals[error_columns].to_numpy()).max())
    if max_f0_error > EPS:
        raise AssertionError(f"F0 replay error {max_f0_error} exceeds tolerance")

    availability = json.loads(
        (consistency / "plan_record_availability.json").read_text(encoding="utf-8")
    )
    if availability["plan_execution_deviation_identifiable"] is not False:
        raise AssertionError("missing plan trajectories were not marked unidentifiable")

    daily = pd.read_csv(ablation / "daily_savings.csv")
    if len(daily) != 334 or daily.duplicated("date").any():
        raise AssertionError("daily savings must contain 334 unique dates")
    annual = pd.read_csv(ablation / "annual_strategy_comparison.csv")
    q42_cost = float(annual.loc[annual.strategy.eq("Q4-2"), "annual_cost_yuan"].iloc[0])
    q43_cost = float(annual.loc[annual.strategy.eq("Q4-3"), "annual_cost_yuan"].iloc[0])
    if abs(float(daily.delta_cost.sum()) - (q42_cost - q43_cost)) > EPS:
        raise AssertionError("daily and annual savings do not reconcile")

    attribution = pd.read_csv(ablation / "annual_accounting_attribution.csv")
    attribution_gap = abs(float(attribution.contribution_yuan.sum()) - (q42_cost - q43_cost))
    if attribution_gap > EPS:
        raise AssertionError("accounting attribution does not reconcile")

    monthly = pd.read_csv(stability / "monthly_stability.csv")
    if len(monthly) != 12 or int(monthly.days.sum()) != 334:
        raise AssertionError("monthly stability coverage is incomplete")
    january = monthly.loc[monthly.month.eq("2025-01")]
    if (len(january) != 1
            or january.coverage_status.iloc[0] != "not_in_formal_evaluation"
            or int(january.days.iloc[0]) != 0
            or not pd.isna(january.savings_yuan.iloc[0])):
        raise AssertionError("January must be explicitly marked outside formal evaluation")
    if abs(float(monthly.savings_yuan.sum()) - (q42_cost - q43_cost)) > EPS:
        raise AssertionError("monthly savings do not reconcile")

    charts = [
        stability / "monthly_savings.svg",
        stability / "daily_benefit_histogram.svg",
        stability / "cumulative_savings.svg",
    ]
    for chart in charts:
        if not chart.exists() or chart.stat().st_size < 500:
            raise AssertionError(f"missing or empty chart: {chart}")

    required_reports = [
        ROOT / "docs/q4/plan_execution_consistency.md",
        ROOT / "docs/q4/revenue_attribution.md",
        ROOT / "docs/q4/stability_analysis.md",
    ]
    for report in required_reports:
        if not report.exists() or report.stat().st_size < 1000:
            raise AssertionError(f"missing report: {report}")

    protected = {}
    for relative, expected in PROTECTED_HASHES.items():
        actual = _sha256(ROOT / relative)
        protected[relative] = {"expected": expected, "actual": actual,
                               "unchanged": actual == expected}
        if actual != expected:
            raise AssertionError(f"protected file changed: {relative}")

    result = {
        "status": "PASS",
        "f0_rows": len(residuals),
        "max_f0_error": max_f0_error,
        "daily_rows": len(daily),
        "monthly_rows": len(monthly),
        "annual_savings_yuan": q42_cost - q43_cost,
        "accounting_attribution_gap_yuan": attribution_gap,
        "protected_files": protected,
    }
    output = ROOT / "outputs/q4/stage2_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VERIFY=PASS")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "protected_files"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        verify_outputs()
