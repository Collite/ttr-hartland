"""S5's pilot SQL against its twin (G2 S2.7 T1(e)/T2, `data/stories/pilot/s5_rank.sql`, `s5_group.sql`,
`checks.s5_pilot`; design08 §8): on conftest's hand-built pilot, the SQL's `region_return` is the twin's frame — money
within 0.01 CZK, ratios within 1e-9, the same integrity — and its two metrics are the twin's, NULL where the twin's are
NaN: negative gains, tied and equal returns, a weak group not positive, and builds that fail closed. The rows go in
inside a transaction that rolls back (conftest's `put`: the foreign keys are deferred, so the rest is never missed)."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
import pytest

from hartland_sim.checks import s5_pilot
from hartland_sim.checks.stories import Build
from hartland_sim.config import STORIES_DIR

pytestmark = pytest.mark.component

Put = Callable[..., None]  # conftest's `put`
PilotRows = Callable[..., Any]  # conftest's `pilot_rows`
EFFECTS = {"PHA": 8000.0, "JMV": 6000.0, "SJC": 4000.0, "ZSC": 2000.0, "VCH": -1000.0, "SMS": 1000.0}
MONEY = ("programme_margin", "holdout_margin", "baseline_adjustment", "gain", "asset_cost", "handout_cost", "cost")
RATIOS = ("loan_coverage", "cost_coverage", "gain_per_cost", "net_return")
COUNTS = ("pairs", "unpaired", "cross_region", "potential_at_entry", "lost")


@pytest.fixture(scope="module")
def db(hartland_db: psycopg.Connection) -> psycopg.Connection:
    return hartland_db


def _sql(db: psycopg.Connection, name: str) -> tuple[Any, dict[str, dict[str, Any]]]:
    """The file's metric and its `region_return` rows by region (the CTEs, then `SELECT * FROM region_return`)."""
    text = (STORIES_DIR / name).read_text(encoding="utf-8")
    final = re.search(r"^SELECT", text, re.M)  # the final SELECT starts its line; the CTEs' bodies are indented
    assert final, name
    db.execute("SET LOCAL search_path TO hartland")
    ((metric,),) = db.execute(text).fetchall()
    cursor = db.execute(text[: final.start()] + "SELECT * FROM region_return")
    assert cursor.description is not None
    names = [column.name for column in cursor.description]
    return metric, {row[0]: dict(zip(names, row, strict=True)) for row in cursor.fetchall()}


def _both(db: psycopg.Connection, put: Put, case: Any, run: Path) -> tuple[Any, Any, Any]:
    """The twin's frame and metrics on the case's CSVs, and the SQL's on its rows."""
    case.write(run)
    frame = s5_pilot.regions(Build(run))
    twins = s5_pilot.rank(frame), s5_pilot.claim(frame)[2]
    with db.transaction(force_rollback=True):
        for table, rows in case.rows.items():
            for row in rows:
                put(db, table, **row)
        db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        rank, rows = _sql(db, s5_pilot.SQL[0])
        group, again = _sql(db, s5_pilot.SQL[1])
    assert rows == again  # one block, byte for byte
    return frame, (rank, group), (rows, twins)


def _same(frame: Any, rows: dict[str, dict[str, Any]]) -> None:
    assert sorted(rows) == sorted(frame.index)
    for region, row in rows.items():
        mine = frame.loc[region]
        assert row["integrity"] == mine["integrity"], region
        for name in COUNTS:
            assert row[name] == mine[name], (region, name)
        for names, tolerance in ((MONEY, 0.01), (RATIOS, 1e-9)):
            for name in names:
                theirs = row[name]
                if theirs is None:
                    assert math.isnan(mine[name]), (region, name)
                else:
                    assert float(Decimal(theirs)) == pytest.approx(mine[name], abs=tolerance), (region, name)


def _metric(sql: Any, twin: float) -> None:
    if sql is None:
        assert math.isnan(twin)
    else:
        assert float(sql) == pytest.approx(twin, abs=1e-9)


@pytest.mark.parametrize(
    "effects",
    [
        EFFECTS,
        {"PHA": -1000.0, "JMV": -2000.0, "SJC": 1000.0, "ZSC": 3000.0, "VCH": 4000.0, "SMS": 4000.0},
        dict.fromkeys(EFFECTS, 2000.0),
        {"PHA": 9000.0, "JMV": 8000.0, "SJC": 4000.0, "ZSC": 0.0, "VCH": -2000.0, "SMS": -1000.0},
    ],
    ids=["story", "negative gains, VCH and SMS tied", "six equal returns", "weak group not positive"],
)
def test_the_sql_is_the_twin(
    db: psycopg.Connection, put: Put, pilot_rows: PilotRows, tmp_path: Path, effects: dict[str, float]
) -> None:
    frame, (rank, group), (rows, (rank_twin, group_twin)) = _both(db, put, pilot_rows(effects), tmp_path)
    _same(frame, rows)
    _metric(rank, rank_twin)
    _metric(group, group_twin)
    assert rank is not None  # every region valid


def _short(case: Any) -> None:
    """The build ends a day before W does: no order on its last day."""
    gone = {o["sales_order_id"] for o in case.rows["sales_order"] if o["ordered_on"] == date(2025, 12, 31)}
    case.rows["sales_order"] = [o for o in case.rows["sales_order"] if o["sales_order_id"] not in gone]
    case.rows["sales_order_line"] = [r for r in case.rows["sales_order_line"] if r["sales_order_id"] not in gone]
    lines = {r["sales_order_line_id"] for r in case.rows["sales_order_line"]}
    discounts = case.rows["sales_order_line_discount"]
    case.rows["sales_order_line_discount"] = [r for r in discounts if r["sales_order_line_id"] in lines]


def _overlap(case: Any) -> None:
    loan = case.rows["pos_loan"][0]
    case.rows["pos_loan"].append(loan | {"pos_loan_id": 9001, "lent_on": date(2024, 12, 31), "returned_on": None})


def _contaminated(case: Any) -> None:
    holdout = case.customer("JMV", 2)
    case.rows["visit"].append({"visit_id": 9002, "customer_id": holdout, "store_id": holdout, "day": date(2025, 2, 3)})
    case.rows["pos_handout"].append({"pos_handout_id": 9002, "visit_id": 9002, "cost": 340.0})


def _broken_pair(case: Any) -> None:
    gone = case.customer("SJC", 2)
    case.rows["pos_pilot_assignment"] = [r for r in case.rows["pos_pilot_assignment"] if r["customer_id"] != gone]


def _uncovered(case: Any) -> None:
    case.rows["customer"].append({"customer_id": 9003, "channel": "gas_ka", "since": None, "until": None})
    case.rows["pos_asset"].append({"pos_asset_id": 9003, "region_id": 4, "cost": 50000.0})
    case.rows["pos_loan"].append(
        {"pos_loan_id": 9003, "pos_asset_id": 9003, "customer_id": 9003, "store_id": None}
        | {"lent_on": date(2024, 1, 1), "returned_on": None}
    )


@pytest.mark.parametrize(
    ("spoil", "integrity"),
    [
        (_short, "short"),
        (_overlap, "overlap"),
        (_contaminated, "contamination"),
        (_broken_pair, "pairs"),
        (_uncovered, "coverage"),
    ],
)
def test_the_sql_fails_closed_as_the_twin_does(
    db: psycopg.Connection,
    put: Put,
    pilot_rows: PilotRows,
    tmp_path: Path,
    spoil: Callable[[Any], None],
    integrity: str,
) -> None:
    """A spoiled build: the same integrity row for row, and both metrics NULL in SQL, NaN in the twin."""
    case = pilot_rows(EFFECTS)
    spoil(case)
    frame, (rank, group), (rows, _) = _both(db, put, case, tmp_path)
    _same(frame, rows)
    assert integrity in {row["integrity"] for row in rows.values()}
    assert rank is None and group is None
