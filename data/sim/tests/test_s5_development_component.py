"""G2 S2.10 T4's parity — the development runner's SQL against its twin (`checks.s5_development.parity`, `compare`;
design09 §5) on PG16: conftest's hand-built decisions put in and dumped as a build's CSVs, then copied back by the
runner's own load (each table analyzed before `sales_pl` is refreshed, in a transaction that rolls back): PARITY with
the twin's judgement of the same decisions. A judgement off by a count, a give-away's two haléře, a q or a metric's
last digits, an integrity or its frames is named — the parity does not take the twin's word."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest

from hartland_sim.checks import s5_decision
from hartland_sim.checks import s5_development as dev

pytestmark = pytest.mark.component

Put = Callable[..., None]  # conftest's `put`
DecisionRows = Callable[..., Any]  # conftest's `decision_rows`
REGIONS = ("PHA", "JMV", "SJC", "ZSC", "VCH", "SMS")
Judged = s5_decision.Judged


def _graded(region: str, k: int) -> list[tuple[int, str]]:
    """Scores 1…10, the five ending at PHA's 10, JMV's 9, … SMS's 5 chosen, every fourth round's best cancelled."""
    best = 10 - REGIONS.index(region)
    out = [(s, "selected" if best - 4 <= s <= best else "no_budget") for s in range(1, 11)]
    return [(s, "cancelled" if s == best and k % 4 == 0 else o) for s, o in out]


@pytest.fixture(scope="module")
def built(
    hartland_db: psycopg.Connection,
    put: Put,
    decision_rows: DecisionRows,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Judged]:
    """The case's dump, as the runner loads a build, and the twin's judgement of the case's own files."""
    case = decision_rows(_graded)
    twin = tmp_path_factory.mktemp("twin")
    case.write(twin)
    judged = s5_decision.judge(twin)
    assert judged.status == "PASS"
    dump = tmp_path_factory.mktemp("dump")
    db = hartland_db
    with db.transaction(force_rollback=True):
        db.execute("SET LOCAL search_path TO hartland")
        for table, rows in case.rows.items():
            for row in rows:
                put(db, table, **row)
        for table in s5_decision.TABLES:
            with (
                (dump / f"{table}.csv").open("wb") as out,
                db.cursor().copy(f"COPY hartland.{table} TO STDOUT WITH (FORMAT csv, HEADER true)") as copy,
            ):
                for chunk in copy:
                    out.write(chunk)
    return dump, judged


def test_the_runner_s_load_is_parity_with_the_twin(built: tuple[Path, Judged], pg_dsn: str) -> None:
    dump, judged = built
    found = dev.parity(pg_dsn, dump, judged)
    assert found.differences == ()
    assert found.money <= 0.01 and found.quality <= 1e-9 and found.metric <= 1e-9
    with psycopg.connect(pg_dsn) as conn:  # the load rolled back
        assert conn.execute("SELECT COUNT(*) FROM hartland.pos_placement_decision").fetchone() == (0,)


def _frame(judged: Judged, region: str, name: str, value: Any) -> Judged:
    assert judged.frame is not None
    frame = judged.frame.copy()
    held: Any = frame.loc[region, name]
    frame.loc[region, name] = value if isinstance(value, str) else held + value
    return dataclasses.replace(judged, frame=frame)


def _q(judged: Judged) -> Judged:
    assert judged.rounds is not None
    rounds = judged.rounds.copy()
    region, day = rounds.index.get_level_values("region"), rounds.index.get_level_values("decision_day")
    moved = (region == "VCH") & (day == pd.Timestamp(2024, 5, 31))
    rounds["q"] = rounds["q"].where(~moved, rounds["q"] + 2e-9)
    return dataclasses.replace(judged, rounds=rounds)


def _verdict(judged: Judged, name: str, value: float) -> Judged:
    """The verdict's `name` moved by `value`, or set to it when NaN."""
    assert judged.verdict is not None
    held = getattr(judged.verdict, name)
    moved: Any = value if value != value else held + value
    return dataclasses.replace(judged, verdict=dataclasses.replace(judged.verdict, **{name: moved}))


@pytest.mark.parametrize(
    ("doctor", "named"),
    [
        (lambda j: _frame(j, "PHA", "chosen", 1), "PHA chosen"),
        (lambda j: _frame(j, "SMS", "cancelled", -1), "SMS cancelled"),
        (lambda j: _frame(j, "JMV", "handout_cost", 0.02), "JMV handout_cost"),
        (lambda j: _frame(j, "SJC", "quality", 2e-9), "SJC quality"),
        (lambda j: _frame(j, "ZSC", "integrity", "support"), "ZSC integrity"),
        (_q, "VCH 2024-05-31 q"),
        (lambda j: _verdict(j, "rank", 2e-9), "rank"),
        (lambda j: _verdict(j, "difference", float("nan")), "difference"),
        (lambda j: dataclasses.replace(j, frame=None, rounds=None, verdict=None), "no twin frames"),
    ],
    ids=["a count", "a cancellation", "two haléře", "a Q", "an integrity", "a round's q", "the rank", "a NaN", "none"],
)
def test_a_judgement_the_sql_does_not_give_is_named(
    built: tuple[Path, Judged], pg_dsn: str, doctor: Callable[[Judged], Judged], named: str
) -> None:
    dump, judged = built
    found = dev.parity(pg_dsn, dump, doctor(judged))
    assert len(found.differences) == 1 and found.differences[0].startswith(named), found.differences
