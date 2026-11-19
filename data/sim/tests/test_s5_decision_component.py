"""S5's decision-quality SQL against its twin (G2 S2.9 T5, `data/stories/pilot/s5_quality_rank.sql`,
`s5_quality_difference.sql`, `checks.s5_decision`; design09 §§3–5): on conftest's hand-built decisions, the SQL's
`round_quality` and `region_quality` are the twin's frames — keys, counts and integrity exact, money within 0.01 CZK,
q, Q and the metrics within 1e-9 — and its two metrics the twin's, NULL where the twin's are NaN: tied, empty and
reversed rounds, six equal Q, and builds that fail closed, a decision, an asset and a give-away's visit deliberately
missing among them. The rows go in inside a transaction that rolls back (conftest's `put`: the foreign keys are
deferred, so the rest is never missed), and are analyzed before `sales_pl` is refreshed (#22 Found: a rolled-back
load leaves the planner's statistics at none)."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest

from hartland_sim.checks import s5_decision
from hartland_sim.checks.stories import Build
from hartland_sim.config import STORIES_DIR

pytestmark = pytest.mark.component

Put = Callable[..., None]  # conftest's `put`
DecisionRows = Callable[..., Any]  # conftest's `decision_rows`
MONEY = (
    "observed_margin",
    "asset_use_cost",
    "handout_cost",
    "programme_cost",
    "no_sales_cost",
    "fleet_cost",
    "idle_cost",
)
COUNTS = tuple(name for name in s5_decision.COLUMNS if name not in (*MONEY, "quality", "integrity"))
DAYS = [date(2024 + m // 12, m % 12 + 1, 1) - timedelta(days=1) for m in range(24)]  # W's rounds' decision days


@pytest.fixture(scope="module")
def db(hartland_db: psycopg.Connection) -> psycopg.Connection:
    return hartland_db


def _sql(db: psycopg.Connection, name: str) -> tuple[Any, dict[str, dict[str, Any]], dict[tuple[str, date], Any]]:
    """The file's metric, its `region_quality` rows by region and its `round_quality` rows by (region, day)."""
    text = (STORIES_DIR / name).read_text(encoding="utf-8")
    final = re.search(r"^SELECT", text, re.M)  # the final SELECT starts its line; the CTEs' bodies are indented
    assert final, name
    ((metric,),) = db.execute(text).fetchall()
    out = []
    for relation in ("region_quality", "round_quality"):
        cursor = db.execute(text[: final.start()] + f"SELECT * FROM {relation}")
        assert cursor.description is not None
        names = [column.name for column in cursor.description]
        out.append([dict(zip(names, row, strict=True)) for row in cursor.fetchall()])
    regions = {row["region"]: row for row in out[0]}
    rounds = {(row["region"], row["decision_day"]): row for row in out[1]}
    return metric, regions, rounds


def _both(db: psycopg.Connection, put: Put, case: Any, run: Path) -> tuple[Any, ...]:
    """The twin's frames and metrics on the case's CSVs, and the SQL's on its rows."""
    case.write(run)
    build = Build(run)
    frame, rounds = s5_decision.regions(build), s5_decision.rounds(build)
    twins = s5_decision.rank(frame), s5_decision.quality(frame)[2]
    with db.transaction(force_rollback=True):
        db.execute("SET LOCAL search_path TO hartland")
        for table, rows in case.rows.items():
            for row in rows:
                put(db, table, **row)
        for table in case.rows:
            db.execute(f"ANALYZE hartland.{table}")
        db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        rank, regions, by_round = _sql(db, s5_decision.SQL[0])
        difference, again, by_round_again = _sql(db, s5_decision.SQL[1])
    assert (regions, by_round) == (again, by_round_again)  # one block, byte for byte
    return frame, rounds, (rank, difference), (regions, by_round), twins


def _number(value: Any) -> float:
    return float(Decimal(value)) if isinstance(value, Decimal) else float(value)


def _same(frame: Any, rounds: Any, regions: dict[str, dict[str, Any]], by_round: dict[Any, Any]) -> None:
    assert sorted(regions) == sorted(frame.index)
    for region, row in regions.items():
        mine = frame.loc[region]
        assert row["integrity"] == mine["integrity"], region
        for name in COUNTS:
            assert int(row[name]) == int(mine[name]), (region, name)
        for name in MONEY:
            assert _number(row[name]) == pytest.approx(mine[name], abs=0.01), (region, name)
        if row["quality"] is None:
            assert math.isnan(mine["quality"]), region
        else:
            assert _number(row["quality"]) == pytest.approx(mine["quality"], abs=1e-9), region
    keys = {(region, day.date()) for region, day in rounds.index}
    assert set(by_round) == keys and len(by_round) == 6 * 24
    for (region, day), row in by_round.items():
        mine = {name: rounds[name].loc[(region, pd.Timestamp(day))] for name in s5_decision.ROUND_COLUMNS}
        for name in ("decisions", "viable", "chosen", "unchosen"):
            assert int(row[name]) == int(mine[name]), (region, day, name)
        if row["q"] is None:
            assert math.isnan(mine["q"]), (region, day)
        else:
            assert _number(row["q"]) == pytest.approx(mine["q"], abs=1e-9), (region, day)


def _metric(sql: Any, twin: float) -> None:
    if sql is None:
        assert math.isnan(twin)
    else:
        assert _number(sql) == pytest.approx(twin, abs=1e-9)


def _story(region: str, k: int) -> list[tuple[int, str]]:
    """Each region a quality of its own (PHA chooses best, SMS worst), ties, a cancelled choice and drawn no_assets;
    round 3 of SJC all tied (q = 0.5), round 5 of ZSC empty, round 7 of VCH without an unchosen candidate."""
    if (region, k) == ("SJC", 3):
        return [(4, "selected")] * 5 + [(4, "no_budget")] * 5
    if (region, k) == ("ZSC", 5):
        return []
    if (region, k) == ("VCH", 7):
        return [(s, "selected") for s in range(1, 6)]
    best = {"PHA": 10, "JMV": 9, "SJC": 8, "ZSC": 7, "VCH": 6, "SMS": 5}[region]
    chosen = set(range(best - 4, best + 1))
    out = []
    for s in range(1, 11):
        outcome = "selected" if s in chosen else ("no_asset" if s == 1 else "no_budget")
        out.append((s if s != 3 else 2, "cancelled" if s == best and k % 4 == 0 else outcome))
    return out


@pytest.mark.parametrize("pool", [None, _story], ids=["six equal Q", "a story of six qualities"])
def test_the_sql_is_the_twin(
    db: psycopg.Connection, put: Put, decision_rows: DecisionRows, tmp_path: Path, pool: Any
) -> None:
    case = decision_rows() if pool is None else decision_rows(pool)
    frame, rounds, (rank, difference), (regions, by_round), (rank_twin, difference_twin) = _both(
        db, put, case, tmp_path
    )
    _same(frame, rounds, regions, by_round)
    _metric(rank, rank_twin)
    _metric(difference, difference_twin)
    assert rank is not None and difference is not None  # every region valid
    if pool is not None:
        assert regions["SJC"]["comparable_rounds"] == 24 and by_round[("ZSC", date(2024, 5, 31))]["q"] is None
        assert _number(by_round[("SJC", date(2024, 3, 31))]["q"]) == 0.5


def _short(case: Any) -> None:
    """The build ends a day before W does: no order on its last day."""
    case.rows["sales_order"] = [o for o in case.rows["sales_order"] if o["ordered_on"] != date(2025, 12, 31)]
    orders = {o["sales_order_id"] for o in case.rows["sales_order"]}
    case.rows["sales_order_line"] = [r for r in case.rows["sales_order_line"] if r["sales_order_id"] in orders]


def _selection_missing(case: Any) -> None:
    """A selected decision deleted: its loan has no selection."""
    held = case.decisions("JMV", date(2024, 5, 31))
    gone = next(d for d in held if d["outcome"] == "selected")
    case.rows["pos_placement_decision"].remove(gone)


def _key_twice(case: Any) -> None:
    held = case.decisions("SJC", date(2024, 5, 31))[0]
    case.rows["pos_placement_decision"].append(held | {"pos_placement_decision_id": 900_001, "loan_id": None})


def _score_changed(case: Any) -> None:
    held = case.decisions("ZSC", date(2025, 2, 28))[0]
    held["score"] = held["score"] + 0.5


def _cross_region(case: Any) -> None:
    held = case.decisions("VCH", date(2025, 2, 28))[9]
    held["region_id"] = 1


def _asset_missing(case: Any) -> None:
    gone = next(a for a in case.rows["pos_asset"] if a["region_id"] == 6)
    case.rows["pos_asset"].remove(gone)


def _visit_missing(case: Any) -> None:
    case.rows["visit"] = case.rows["visit"][1:]


def _holdout_display(case: Any) -> None:
    holdout = case.customer("PHA", 51)
    case.rows["pos_loan"].append(
        {"pos_loan_id": 900_002, "pos_asset_id": 1099, "customer_id": holdout, "store_id": holdout}
        | {"lent_on": date(2022, 6, 1), "returned_on": date(2024, 3, 1)}
    )


def _out_twice(case: Any) -> None:
    loan = next(row for row in case.rows["pos_loan"] if row["lent_on"] == date(2024, 6, 1))
    case.rows["pos_loan"].append(loan | {"pos_loan_id": 900_003, "lent_on": date(2022, 6, 1)})


def _assignment_twice(case: Any) -> None:
    """A programme customer's assignment copied under a new id: design09 §3's duplicate key, never counted twice."""
    held = next(a for a in case.rows["pos_pilot_assignment"] if a["customer_id"] == case.customer("PHA", 1))
    case.rows["pos_pilot_assignment"].append(held | {"pos_pilot_assignment_id": 900_101})


def _unsupported(case: Any) -> None:
    """SMS's last 13 rounds without an unchosen candidate: 11 comparable rounds."""
    for day in DAYS[11:]:
        for row in case.decisions("SMS", day):
            if row["outcome"] == "no_budget":
                case.rows["pos_placement_decision"].remove(row)


@pytest.mark.parametrize(
    ("spoil", "integrity"),
    [
        (_short, "short"),
        (_selection_missing, "loans"),
        (_key_twice, "duplicate"),
        (_score_changed, "fields"),
        (_cross_region, "links"),
        (_holdout_display, "ownership"),
        (_asset_missing, "asset"),
        (_visit_missing, "visit"),
        (_out_twice, "overlap"),
        (_unsupported, "support"),
        (_assignment_twice, "duplicate"),
    ],
)
def test_the_sql_fails_closed_as_the_twin_does(
    db: psycopg.Connection,
    put: Put,
    decision_rows: DecisionRows,
    tmp_path: Path,
    spoil: Callable[[Any], None],
    integrity: str,
) -> None:
    """A spoiled build: the same frames row for row, the failure named, and both metrics NULL in SQL, NaN in the
    twin."""
    case = decision_rows()
    spoil(case)
    frame, rounds, (rank, difference), (regions, by_round), _ = _both(db, put, case, tmp_path)
    _same(frame, rounds, regions, by_round)
    assert integrity in {row["integrity"] for row in regions.values()}
    assert rank is None and difference is None
