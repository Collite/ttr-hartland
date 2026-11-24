"""Constructed NH-D28 worlds through the ordinary-row audit and assembled SQL."""

from __future__ import annotations

import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import psycopg
import pytest

from hartland_sim.checks import s5_certify, s5_decision, s5_monthly
from hartland_sim.checks import s5_development as dev
from hartland_sim.checks.s5_monthly import GROUP_FILE, GROUP_REVISION, MONTHLY_FILE
from hartland_sim.cli import main
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.evaluation import Evaluation
from hartland_sim.protocol import Protocol

HELD = Evaluation.load(GROUP_FILE)
LEGACY = Evaluation.load(MONTHLY_FILE)
MONTHS = s5_monthly.months(HELD)
SCENARIOS = (
    "equal story bars",
    "control at gap with low high mean",
    "rank only story",
    "rank only control",
    "both story bars fail",
    "unsupported SJC",
    "decisive month",
    "decisive pair",
)


def _world(scenario: str, decision_rows: Callable[..., Any], run: Path) -> Any:
    # Round quality in twentieths, with different placement counts in the two high regions.
    values = {
        "equal story bars": (15, 13, 20, 12, 12, 12),
        "control at gap with low high mean": (13, 11, 10, 10, 10, 10),
        "rank only story": (14, 12, 10, 8, 6, 6),
        "rank only control": (14, 13, 12, 12, 12, 12),
        "both story bars fail": (10, 10, 10, 10, 10, 10),
        "unsupported SJC": (15, 13, 20, 12, 12, 12),
        "decisive month": (13, 15, 12, 11, 11, 11),
        "decisive pair": (13, 15, 12, 11, 11, 11),
    }
    by_region = dict(zip(s5_decision.SKILL, values[scenario], strict=True))

    def pool(region: str, month: int) -> list[tuple[int, str]]:
        credit = by_region[region]
        if region == "PHA":
            if scenario == "decisive month":
                credit = {0: 8, 1: 20}.get(month, credit)
            if scenario == "decisive pair":
                credit = {0: 11, 1: 18, 2: 11, 17: 18}.get(month, credit)
        if scenario == "unsupported SJC" and region == "SJC" and month == 0:
            return [(4, "selected")] + [(1, "no_budget")] * 5
        picked = [2, 4] * (2 if region == "JMV" else 1)
        wins, extra = divmod(credit, 4)
        # Each alternative at forecast 1, 4, 3, 2 or 5 earns respectively 4, 1, 2, 3 or 0 half-win units.
        alternatives = [1] * wins + ([(4, 3, 2)[extra - 1]] if extra else [])
        alternatives += [5] * (5 - len(alternatives))
        return [(score, "selected") for score in picked] + [(score, "no_budget") for score in alternatives]

    built = decision_rows(pool)
    if scenario == "control at gap with low high mean":
        built.meta["pilot_mode"] = "full-null"
    elif scenario == "rank only control":
        built.meta["pilot_mode"] = "skill-neutral"
    built.write(run)
    return built


def _check(
    scenario: str, base: s5_decision.Judged, judged: s5_decision.Judged, cases: tuple[s5_monthly.Case, ...]
) -> None:
    assert judged.frame is not None
    assert set(judged.frame.index) == set(s5_decision.SKILL)
    if scenario == "unsupported SJC":
        assert judged.status == "INCONCLUSIVE" and cases == ()
        assert judged.frame.loc["SJC", "integrity"] == "support"
        return
    assert len(cases) == 301
    full = cases[0]
    assert full.omitted == ()
    assert judged.status == full.verdict.status
    failed = {case.omitted: case.failure for case in cases[1:] if case.failure}
    if scenario == "equal story bars":
        assert full.verdict.status == "PASS" and not failed
        assert full.verdict.q_high == 0.70 and full.verdict.q_low == 0.60 and full.verdict.difference == 0.10
        counts = judged.frame["chosen"].to_dict()
        assert counts["PHA"] * 2 == counts["JMV"]
        weighted = (Fraction(15, 20) + 2 * Fraction(13, 20)) / 3
        assert weighted < Fraction(7, 10)
        legacy = s5_monthly.cases(s5_monthly.judge(base, LEGACY), LEGACY)
        assert not legacy[0].verdict.rank_holds
    elif scenario == "control at gap with low high mean":
        assert full.verdict.q_high == 0.60 and full.verdict.difference == 0.10
        assert not full.verdict.control_rejects and full.failure == ("difference",)
        assert len(failed) == 300
    elif scenario == "rank only story":
        legacy = s5_monthly.cases(s5_monthly.judge(base, LEGACY), LEGACY)
        assert legacy[0].verdict.rank_holds
        assert full.failure == ("q_high",)
    elif scenario == "rank only control":
        legacy = s5_monthly.cases(s5_monthly.judge(base, LEGACY), LEGACY)
        assert legacy[0].verdict.rank_holds
        assert full.verdict.control_rejects and not failed
    elif scenario == "both story bars fail":
        assert full.failure == ("q_high", "difference")
        assert len(failed) == 300 and set(failed.values()) == {full.failure}
    elif scenario == "decisive month":
        assert full.failure is None and failed[(MONTHS[1],)] == ("q_high",)
    else:
        assert full.failure is None and failed == {(MONTHS[1], MONTHS[17]): ("q_high",)}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_group_development_twin(scenario: str, decision_rows: Callable[..., Any], tmp_path: Path) -> None:
    _world(scenario, decision_rows, tmp_path)
    base = s5_decision.judge(tmp_path)
    judged, cases = s5_monthly.evaluate(base, HELD)
    _check(scenario, base, judged, cases)


@pytest.mark.parametrize(
    ("scenario", "full_status", "failed_count", "dates_and_conditions"),
    [
        ("decisive pair", "PASS", 1, "2024-02-01 a 2025-06-01 — Q_H ≥ 0,70"),
        ("both story bars fail", "REJECTED", 300, "2024-01-01 — Q_H ≥ 0,70, D ≥ 0,10"),
        ("control at gap with low high mean", "FAIL", 300, "2024-01-01 — D < 0,10"),
    ],
)
def test_stories_check_reports_failed_omissions(
    scenario: str,
    full_status: str,
    failed_count: int,
    dates_and_conditions: str,
    decision_rows: Callable[..., Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _world(scenario, decision_rows, tmp_path)
    seed = int(built.meta["master_seed"])
    world, stories = WorldConfig.load(), StoriesConfig.load()
    mode = built.meta["pilot_mode"]
    expected = dev.expected(seed, mode, world, stories, Protocol.load().window_until)
    built.meta.update(dict(HELD.identity()), stories_sha256=expected.stories_sha256, build_id=expected.build_id)
    built.write(tmp_path)
    head = "1" * 40
    path = tmp_path / "build.json"
    build_report = json.loads(path.read_bytes())
    build_report.update(git_sha=head, stories_sha256=expected.stories_sha256, build_id=expected.build_id)
    path.write_text(json.dumps(build_report), encoding="utf-8")
    monkeypatch.setattr(s5_certify, "git_sha", lambda: (head, False))

    flags = ["--null"] if mode == "full-null" else []
    assert main(["stories-check", "--csv", str(tmp_path), *flags]) == 1
    text = (tmp_path / "stories.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| S5 |"))
    assert "| FAIL (" in row and f"{failed_count} z 300" in row
    assert "Celkový výsledek: FAIL" in text
    assert f"Výsledek za celé období: {full_status}" in text
    assert f"- {dates_and_conditions}" in text
    report = text.split("## Nesplněné výpustky měsíců", 1)[1]
    assert sum(line.startswith("- ") for line in report.splitlines()) == failed_count


@pytest.mark.component
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_group_development_sql(
    scenario: str,
    decision_rows: Callable[..., Any],
    tmp_path: Path,
    hartland_db: psycopg.Connection,
    put: Callable[..., None],
) -> None:
    built = _world(scenario, decision_rows, tmp_path)
    base = s5_decision.judge(tmp_path)
    judged, cases = s5_monthly.evaluate(base, HELD)
    _check(scenario, base, judged, cases)
    with hartland_db.transaction(force_rollback=True):
        hartland_db.execute("SET LOCAL search_path TO hartland")
        for table, rows in built.rows.items():
            for row in rows:
                put(hartland_db, table, **row)
        hartland_db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        parity = dev.compare(hartland_db, judged, cases, revision=GROUP_REVISION, statement=s5_certify.sql())
        assert not parity.differences
