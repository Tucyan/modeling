"""Validation for Q4 paper-ready report and frozen-ledger figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_materials import (
    annual_cost_comparison,
    select_typical_dates,
    svg_document,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "outputs/q4/paper_figures"
REPORT = ROOT / "docs/q4/Q4_final_review_report.md"
EXPECTED_FIGURES = [
    "figure_1_annual_cost_comparison.svg",
    "figure_2_monthly_savings.svg",
    "figure_3_daily_savings_distribution.svg",
    "figure_4_soc_trajectories.svg",
    "figure_5_typical_dates.svg",
]
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
    daily = pd.DataFrame({
        "date": ["2025-02-01", "2025-02-02", "2025-02-03"],
        "cost_q4_2": [100.0, 100.0, 100.0],
        "cost_q4_3": [90.0, 120.0, 95.0],
        "delta_cost": [10.0, -20.0, 5.0],
    })
    typical = select_typical_dates(daily)
    assert typical == {"maximum_gain": "2025-02-01", "maximum_loss": "2025-02-02"}

    annual = annual_cost_comparison(daily)
    q42 = annual.loc[annual.strategy.eq("Q4-2")].iloc[0]
    q43 = annual.loc[annual.strategy.eq("Q4-3")].iloc[0]
    assert float(q42.annual_cost_yuan) == 300.0
    assert float(q43.annual_cost_yuan) == 305.0
    assert float(q43.savings_vs_q4_2_yuan) == -5.0

    svg = svg_document(320, 180, "test", ["<circle cx='10' cy='10' r='2'/>"])
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "viewBox=\"0 0 320 180\"" in svg
    print("SELF_TEST=PASS")


def verify_outputs() -> None:
    if not REPORT.exists() or REPORT.stat().st_size < 6000:
        raise AssertionError("final review report is missing or unexpectedly short")

    for name in EXPECTED_FIGURES:
        path = FIGURES / name
        if not path.exists() or path.stat().st_size < 1200:
            raise AssertionError(f"missing or empty paper figure: {name}")
        text = path.read_text(encoding="utf-8")
        if not text.startswith("<svg") or not text.rstrip().endswith("</svg>"):
            raise AssertionError(f"invalid SVG document: {name}")

    manifest = json.loads((FIGURES / "figure_manifest.json").read_text(encoding="utf-8"))
    if [item["file"] for item in manifest["figures"]] != EXPECTED_FIGURES:
        raise AssertionError("figure manifest does not match required paper figures")
    if manifest["typical_dates"] != {
        "maximum_gain": "2025-06-01", "maximum_loss": "2025-08-25"
    }:
        raise AssertionError("typical dates must be selected from frozen daily extrema")

    annual = pd.read_csv(FIGURES / "annual_cost_comparison.csv")
    q42 = float(annual.loc[annual.strategy.eq("Q4-2"), "annual_cost_yuan"].iloc[0])
    q43 = float(annual.loc[annual.strategy.eq("Q4-3"), "annual_cost_yuan"].iloc[0])
    if abs(q42 - 14654856.782340743) > 1e-6 or abs(q43 - 14432049.316081068) > 1e-6:
        raise AssertionError("annual paper costs do not match frozen N0 results")

    typical = pd.read_csv(FIGURES / "typical_dates_summary.csv")
    if set(typical.date) != {"2025-06-01", "2025-08-25"}:
        raise AssertionError("typical-date summary does not cover both frozen extrema")
    if not np.isclose(float(typical.delta_cost.sum()), 34975.28543480601, atol=1e-6):
        raise AssertionError("typical-date cost deltas do not reconcile")

    protected = {}
    for relative, expected in PROTECTED_HASHES.items():
        actual = _sha256(ROOT / relative)
        protected[relative] = {"expected": expected, "actual": actual,
                               "unchanged": actual == expected}
        if actual != expected:
            raise AssertionError(f"protected file changed: {relative}")

    result = {
        "status": "PASS",
        "report": str(REPORT.relative_to(ROOT)),
        "figure_count": len(EXPECTED_FIGURES),
        "typical_dates": manifest["typical_dates"],
        "annual_savings_yuan": q42 - q43,
        "protected_files": protected,
    }
    (FIGURES / "paper_materials_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VERIFY=PASS")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "protected_files"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.verify:
        verify_outputs()
    else:
        parser.error("choose --self-test or --verify")
