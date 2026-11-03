"""A build's timing (G1 S1.4 T7; architecture §2): from its `build.json`, each quarter's days, seconds and ms a day, and
what a one-year build takes — setup, the warm-up's year and the first written year, the end — against G1's budget
(`plan.md` S1.4: a one-year CSV build ≤ 5 min on the executor). The loop's quarters are measured day by day
(`loop.day.Report`), so a longer build tells what a one-year build of the same code takes without running one."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BUDGET_S = 300.0  # a one-year CSV build


def one_year(report: Mapping[str, Any], year: int) -> float:
    """The seconds a build to the end of `year` takes, from a build to `year` or later: its setup and end, and the
    loop's quarters up to `year`'s last (the warm-up's included)."""
    timings = report["timings_s"]
    loop = sum(counts["seconds"] for quarter, counts in report["quarters"].items() if int(quarter[:4]) <= year)
    return float(timings["setup"] + loop + timings["end"])


def table(report: Mapping[str, Any], year: int) -> str:
    """The quarters' days, seconds and ms a day, and the one-year estimate against the budget, as a Markdown table."""
    rows = ["| quarter | days | s | ms a day |", "|---|---|---|---|"]
    for quarter, counts in report["quarters"].items():
        days, seconds = counts["days"], counts["seconds"]
        rows.append(f"| {quarter} | {days} | {seconds:.1f} | {1000 * seconds / days:.0f} |")
    estimate = one_year(report, year)
    verdict = "within" if estimate <= BUDGET_S else "OVER"
    timings = report["timings_s"]
    rows.append(
        f"\nsetup {timings['setup']:.1f} s · end {timings['end']:.1f} s · a build to {year}-12-31 ≈ {estimate:.0f} s"
        f" — {verdict} the {BUDGET_S:.0f} s budget"
    )
    return "\n".join(rows)
