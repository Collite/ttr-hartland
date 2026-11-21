"""Constructed design10 support and calendar-sensitivity cases for the twin and PostgreSQL."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest

from hartland_sim.checks import s5_decision, s5_monthly
from hartland_sim.checks import s5_development as dev
from hartland_sim.checks.s5_monthly import MONTHLY_FILE
from hartland_sim.config import STORIES_DIR
from hartland_sim.evaluation import Evaluation

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
        result = hartland_db.execute(dev.monthly_sql(), ("story",)).fetchone()
        parity = dev.compare(hartland_db, judged, s5_monthly.cases(judged, HELD))
        assert not parity.differences
    assert result is not None
    regions, rounds, cases, _ = result
    assert len(cases) == 301
    assert cases[0]["omitted"] == [] and cases[0]["rank"] == 0.0
    assert len(regions) == 6 and len(rounds) == 144
    assert {(r["integrity"], r["cancelled"]) for r in regions} == {("valid", 24)}
    assert {(r["chosen"], r["unchosen"], r["q"]) for r in rounds} == {(2, 5, 0.75)}


def sensitivity_fixture(scenario: str, mode: str = "story") -> s5_decision.Judged:
    if scenario == "equality":
        baseline = {"PHA": 0.75, "JMV": 0.65, "SJC": 0.625, "ZSC": 0.62, "VCH": 0.59, "SMS": 0.59}
    else:
        baseline = {"PHA": 0.74, "JMV": 0.65, "SJC": 0.60, "ZSC": 0.55, "VCH": 0.50, "SMS": 0.50}
    if scenario == "full failure":
        baseline["PHA"] = 0.76
    if mode != "story":
        baseline["SJC"] = 0.40
        baseline["PHA"], baseline["JMV"] = baseline["JMV"], baseline["PHA"]
    rows = []
    for region, q in baseline.items():
        for i, month in enumerate(MONTHS):
            score = q
            if region == "PHA":
                peaks = {"one decisive": (1,), "pair decisive": (1, 17)}
                if i in peaks.get(scenario, ()):
                    score = 1.0
                if scenario == "full failure" and i == 1:
                    score = 0.4
            rows.append((region, pd.Timestamp(month - timedelta(days=1)), 2, 250, score))
    rounds = pd.DataFrame(rows, columns=["region", "decision_day", "chosen", "unchosen", "q"])
    rounds = rounds.set_index(["region", "decision_day"])
    frame = pd.DataFrame({"integrity": "valid", "quality": rounds["q"].groupby("region").mean()})
    return s5_decision.Judged(Path("."), mode, "PASS", (), None, frame, rounds, {})


@pytest.mark.parametrize("scenario", ["equality", "one decisive", "pair decisive", "full failure"])
def test_common_omissions_and_original_verdict(scenario: str) -> None:
    found = s5_monthly.cases(sensitivity_fixture(scenario), HELD)
    assert tuple(case.omitted for case in found) == ((), *combinations(MONTHS, 1), *combinations(MONTHS, 2))
    full, *omitted = found
    outcome = s5_monthly.assessment(found)
    if scenario == "equality":
        assert outcome["failure_type"] is None
        assert full.verdict.q_high == 0.70 and full.verdict.difference == 0.10
        assert all(case.failure is None for case in omitted)
        control = s5_monthly.cases(sensitivity_fixture(scenario, "full-null"), HELD)[0]
        assert not control.verdict.rank_holds
        assert control.failure == "difference" and not control.verdict.control_rejects
        exact_rank = replace(HELD, acceptance=HELD.acceptance | {"rank": 1.0})
        assert s5_monthly.cases(sensitivity_fixture(scenario), exact_rank)[0].verdict.rank_holds
    elif scenario == "one decisive":
        assert outcome["failure_type"] == "sensitivity" and full.failure is None
        assert {case.omitted for case in omitted if case.failure} == {
            case.omitted for case in omitted if MONTHS[1] in case.omitted
        }
    elif scenario == "pair decisive":
        assert outcome["failure_type"] == "sensitivity" and full.failure is None
        assert outcome["failures"] == [{"omitted": (MONTHS[1], MONTHS[17]), "condition": "q_high"}]
    else:
        assert outcome["failure_type"] == "full_window" and full.failure == "q_high"
        assert any(case.failure is None for case in omitted)


def test_unsupported_control_cannot_count_as_a_rejection() -> None:
    built = sensitivity_fixture("equality", "skill-neutral")
    assert built.frame is not None
    built.frame.loc["VCH", "integrity"] = "support"
    assert not s5_monthly.cases(built, HELD)


@pytest.mark.component
@pytest.mark.parametrize(
    ("scenario", "mode"), [
        ("equality", "story"), ("equality", "full-null"), ("one decisive", "story"),
        ("pair decisive", "story"), ("full failure", "story"),
    ]
)
def test_sql_sensitivity_matches_the_twin_for_each_omission(scenario: str, mode: str, pg_dsn: str) -> None:
    twin = sensitivity_fixture(scenario, mode)
    assert twin.rounds is not None
    rows = [
        (REGIONS.index(region) + 1, region, pd.Timestamp(day).date(), chosen, unchosen, q)
        for region, day, chosen, unchosen, q in twin.rounds.reset_index().itertuples(index=False, name=None)
    ]
    units = [item for row in rows for item in row]
    data = ", ".join(["(%s::integer, %s::text, %s::date, %s::integer, %s::integer, %s::numeric)"] * len(rows))
    base = f"""WITH RECURSIVE
        skill(region, pos_skill) AS (
            VALUES ('PHA', 1.4), ('JMV', 1.3), ('SJC', 1.0), ('ZSC', 0.6), ('VCH', 0.5), ('SMS', 0.5)
        ),
        evaluation(first_day, last_day) AS (VALUES (DATE '2024-01-01', DATE '2025-12-31')),
        round_quality(region_id, region, decision_day, chosen, unchosen, q) AS (VALUES {data}),
        region AS (SELECT DISTINCT region_id, region AS code FROM round_quality),
        region_quality AS (SELECT region, 'support' AS integrity, avg(q) AS quality FROM round_quality GROUP BY region),
    """
    monthly = (STORIES_DIR / "pilot/s5_monthly.sql").read_text()
    with psycopg.connect(pg_dsn) as conn:
        result = conn.execute(base + monthly, [*units, mode]).fetchone()
    assert result is not None
    _, _, cases, _ = result
    expected = s5_monthly.cases(twin, HELD)
    assert len(cases) == len(expected) == 301
    for got, want in zip(cases, expected, strict=True):
        assert got["omitted"] == [month.isoformat() for month in want.omitted]
        assert got["regions"] == pytest.approx(want.regions, abs=1e-9)
        assert got["failure"] == want.failure
        for name in ("rank", "q_high", "q_low", "difference"):
            assert got[name] == pytest.approx(getattr(want.verdict, name), abs=1e-9)
