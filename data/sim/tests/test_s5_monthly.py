"""Constructed design10 support and calendar-sensitivity cases for the twin and PostgreSQL."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest

from hartland_sim.checks import s5_decision, s5_monthly
from hartland_sim.checks import s5_development as dev
from hartland_sim.config import STORIES_DIR
from hartland_sim.evaluation import MONTHLY_FILE, Evaluation

HELD = Evaluation.load(MONTHLY_FILE)
REGIONS = tuple(s5_decision.SKILL)
MONTHS = s5_monthly.months(HELD)
SUPPORT_CASES = [
    ("complete", True),
    ("missing", False),
    ("duplicate", False),
    ("one chosen", False),
    ("one unchosen", False),
    ("99 unchosen", False),
]


def coverage_rows(case: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for month_index, month in enumerate(MONTHS):
        if case == "missing" and not month_index:
            continue
        chosen = 1 if case == "one chosen" and not month_index else 2
        unchosen = 2 if month_index == 0 else 6 if month_index == 1 else 4 + (month_index in {2, 3, 4, 5})
        if case == "one unchosen" and not month_index:
            unchosen = 1
        if month_index == 1 and case == "one unchosen":
            unchosen += 1
        if case == "99 unchosen" and month_index == 1:
            unchosen -= 1
        rows.append(
            {"region": "PHA", "decision_day": month - timedelta(days=1), "chosen": chosen, "unchosen": unchosen}
        )
    if case == "duplicate":
        rows.append(rows[0] | {"decision_day": rows[0]["decision_day"] + timedelta(days=1)})
    return rows


@pytest.mark.parametrize(("case", "expected"), SUPPORT_CASES)
def test_twin_monthly_support(case: str, expected: bool) -> None:
    rows = pd.DataFrame(coverage_rows(case))
    rows["decision_day"] = pd.to_datetime(rows["decision_day"])
    covered = s5_monthly.support(rows.set_index(["region", "decision_day"]), HELD)
    assert covered["PHA"] is expected
    assert not any(covered[region] for region in REGIONS[1:])


@pytest.mark.component
@pytest.mark.parametrize(("case", "expected"), SUPPORT_CASES)
def test_sql_monthly_support(case: str, expected: bool, pg_dsn: str) -> None:
    rows = coverage_rows(case)
    fields = ("decision_day", "chosen", "unchosen")
    data = ", ".join(["(1, %s::date, %s::integer, %s::integer)"] * len(rows))
    base = f"""WITH
        region(region_id) AS (VALUES (1)),
        evaluation(first_day, last_day) AS (VALUES (DATE '2024-01-01', DATE '2025-12-31')),
        round_quality(region_id, decision_day, chosen, unchosen) AS (VALUES {data}),
        region_quality(region_id, integrity) AS (VALUES (1, 'support')),
    """
    monthly = (STORIES_DIR / "pilot/s5_monthly.sql").read_text()
    support = monthly[: monthly.index("monthly_regions AS")].rstrip().removesuffix(",")
    params = [row[name] for row in rows for name in fields]
    with psycopg.connect(pg_dsn) as conn:
        row = conn.execute(base + support + " SELECT supported FROM monthly_support", params).fetchone()
    assert row == (expected,)


@pytest.mark.component
def test_audited_round_counts_cancellations_and_a_forecast_tie(
    hartland_db: psycopg.Connection,
    put: Callable[..., None],
    decision_rows: Callable[..., Any],
    tmp_path: Path,
) -> None:
    # Two funded decisions, including a cancellation, and five alternatives. One of the ten pairs ties.
    decisions = [(5, "selected"), (4, "cancelled")] + [(i, "no_budget") for i in (1, 2, 3, 4, 6)]
    case = decision_rows(lambda *_: decisions)
    case.write(tmp_path)
    judged = s5_monthly.judge(s5_decision.judge(tmp_path), HELD)
    assert judged.frame is not None and judged.rounds is not None
    assert set(judged.frame["integrity"]) == {"valid"}
    assert set(judged.frame["cancelled"]) == {24}
    assert set(judged.rounds["q"]) == {0.75}

    with hartland_db.transaction(force_rollback=True):
        hartland_db.execute("SET LOCAL search_path TO hartland")
        for table, rows in case.rows.items():
            for row in rows:
                put(hartland_db, table, **row)
        hartland_db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        result = hartland_db.execute(dev.monthly_sql()).fetchone()
    assert result is not None
    regions, rounds = result
    assert len(regions) == 6 and len(rounds) == 144
    assert {(r["integrity"], r["cancelled"]) for r in regions} == {("valid", 24)}
    assert {(r["chosen"], r["unchosen"], r["q"]) for r in rounds} == {(2, 5, 0.75)}
