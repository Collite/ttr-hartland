"""G2 S2.4 T3's cohort-support diagnosis (`checks.s5_support`, NH-D20), with no effect anywhere: a control acquired
in the window pulls NH-D17's gain down and the rule (a control active on the lending day) removes it; a control lost
in the window pushes it up and the rule cannot see it; every control kept is `s5_score`'s own; the strata; the oracle
and the paper read the display's lines."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hartland_sim.checks.s5_support import STRATA, eligible, oracle, pairs, paper, score, strata
from hartland_sim.checks.stories import Build, s5_held, s5_score
from hartland_sim.loop.pos import Grid

LENT, BACK = date(2025, 3, 3), date(2025, 6, 1)  # 13 weekly orders during, 8 in the 56 days before
CELL = {"store_class": "convenience", "street_type": "high_street", "size_class": "S"}
LOAN = 10**12
OPEN = date(2023, 1, 1)
GRID = Grid(0.2, {"high_street": 1.5}, {"S": 1.0})  # u = 0.3
DISPLAYS = {"D-L1": ("L1",)}


def _world(run: Path, others: str = "", lift: float = 0.0) -> Build:
    """One cell of one region: the display lent `LENT` to `BACK` at store 100 (its orders × 1 + `lift` during);
    stores 101–106 active throughout; 107–109 `others`: "acquired" (potential customers until 14 days into the loan,
    then buying) or "lost" (lost 14 days into it) — or absent; store 110 with a display of its own during the loan.
    A store buying orders 1 000 of line L1 a week from 8 weeks before `LENT` to `BACK`."""
    rows: dict[str, list[dict[str, Any]]] = {
        "region": [{"region_id": 1, "code": "PHA"}],
        "area": [{"area_id": 1, "region_id": 1}],
        "product_line": [{"product_line_id": 1, "code": "L1"}],
        "product": [{"product_id": 1, "product_line_id": 1}],
        "sku": [{"sku_id": 1, "product_id": 1}],
        "pos_material": [{"pos_material_id": 1, "code": "D-L1"}],
        "pos_asset": [{"pos_asset_id": 1, "pos_material_id": 1, "region_id": 1, "cost": 36_000.0}],
        "pos_loan": [],
        "store": [],
        "customer": [],
        "delivery_point": [],
        "sales_order": [],
        "sales_order_line": [],
    }
    stores = [*range(100, 107), *(range(107, 110) if others else []), 110]
    turn = LENT + timedelta(days=14)
    for store in stores:
        customer: dict[str, Any] = {"since": OPEN, "until": None, "status": "active"}
        buys = (LENT - timedelta(weeks=8), BACK)
        if store in (107, 108, 109) and others == "acquired":
            customer, buys = {"since": turn, "until": None, "status": "active"}, (turn, BACK)
        elif store in (107, 108, 109) and others == "lost":
            customer, buys = {"since": OPEN, "until": turn, "status": "lost"}, (buys[0], turn)
        rows["store"].append({"store_id": store, "area_id": 1} | CELL)
        rows["customer"].append({"customer_id": store, "channel": "small_retail"} | customer)
        rows["delivery_point"].append({"delivery_point_id": store, "customer_id": store, "store_id": store})
        for week in range(-8, 13):
            day = LENT + timedelta(weeks=week)
            if not buys[0] <= day <= buys[1]:
                continue
            k = len(rows["sales_order"]) + 1
            gross = 1000.0 * (1 + lift) if store == 100 and day >= LENT else 1000.0
            rows["sales_order"].append(
                {"sales_order_id": k, "customer_id": store, "delivery_point_id": store, "ordered_on": day}
            )
            rows["sales_order_line"].append(
                {"sales_order_line_id": k, "sales_order_id": k, "sku_id": 1, "qty_cases": 1.0}
                | {"list_price_case": gross, "delivery_cost_case": None, "warehouse_cost_case": None}
            )
    for k, (store, lent, back) in enumerate(((100, LENT, BACK), (110, LENT + timedelta(30), BACK))):
        rows["pos_loan"].append(
            {"pos_loan_id": LOAN + k, "pos_asset_id": 1, "customer_id": store, "store_id": store}
            | {"lent_on": lent, "returned_on": back}
        )
    rows["sales_order_line_discount"] = []
    run.mkdir(parents=True, exist_ok=True)
    headers = {"sales_order_line_discount": ["sales_order_line_id", "seq", "kind", "amount_line"]}
    for table, found in rows.items():
        with (run / f"{table}.csv").open("w", encoding="utf-8", newline="") as out:
            columns = list(dict.fromkeys(k for row in found for k in row)) if found else headers[table]
            writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in found:
                writer.writerow({k: "" if row.get(k) is None else str(row[k]) for k in columns})
    return Build(run)


def _scored(build: Build) -> tuple[pd.DataFrame, pd.DataFrame]:
    held = s5_held(build)
    columns = ["pos_loan_id", "region", "customer_id", "store_id", "lent_on", "until", "cost"]
    loans = held.loc[held["pos_loan_id"] == LOAN, columns]
    return s5_score(build, loans, held), pairs(build, loans, held)


@pytest.mark.parametrize(
    ("others", "lift", "nh_d17", "rule"),
    [
        ("", 0.0, 0.0, 0.0),  # common drift alone: both 0
        ("acquired", 0.0, -13_000 * 33 / 78, 0.0),  # 3 × 11 orders during, nothing before: q × 111 ÷ 78
        ("acquired", 0.2, 2_600 - 13_000 * 33 / 78, 2_600.0),  # the lift survives the rule only
        ("lost", 0.0, 13_000 - 8_000 * 87 / 72, 13_000 - 8_000 * 87 / 72),  # 3 × 3 orders during: the rule keeps them
    ],
)
def test_a_control_acquired_or_lost_in_the_window_moves_a_gain_of_nothing(
    tmp_path: Path, others: str, lift: float, nh_d17: float, rule: float
) -> None:
    """Store 100 gains nothing (or `lift`); the controls are 101–106 and `others` (110 has a display out)."""
    loans, cohort = _scored(_world(tmp_path / "run", others, lift))
    assert set(cohort["store_id"]) == set(range(101, 107)) | ({107, 108, 109} if others else set())
    assert loans.loc[LOAN, "gain"] == pytest.approx(nh_d17)
    kept = score(loans, cohort, eligible(cohort))
    assert kept.loc[LOAN, "gain"] == pytest.approx(rule)
    assert kept.loc[LOAN, "controls"] == (6 if others == "acquired" else 6 + 3 * (others == "lost"))


@pytest.mark.parametrize("others", ["", "acquired", "lost"])
def test_every_control_kept_is_s5_score_s_own(tmp_path: Path, others: str) -> None:
    loans, cohort = _scored(_world(tmp_path / "run", others))
    again = score(loans, cohort)
    pd.testing.assert_frame_equal(again, loans)


def test_a_store_s_stratum_is_its_customer_and_its_orders_before(tmp_path: Path) -> None:
    """Placebo-like loans on the lending day: 101 bought, 107 not yet a customer, and 101 again a year later — still
    active, nothing bought in the 56 days before."""
    build = _world(tmp_path / "run", "acquired")
    held = s5_held(build)
    later = pd.Timestamp(BACK + timedelta(days=200))
    loans = pd.DataFrame(
        {
            "pos_loan_id": [1, 2, 3],
            "region": "PHA",
            "customer_id": [101, 107, 101],
            "store_id": [101, 107, 101],
            "lent_on": [pd.Timestamp(LENT), pd.Timestamp(LENT), later],
            "until": [pd.Timestamp(BACK), pd.Timestamp(BACK), later + pd.Timedelta(days=30)],
            "cost": 0.0,
        }
    )
    found = strata(build, s5_score(build, loans, held))
    assert found.tolist() == [STRATA[2], STRATA[0], STRATA[1]]


def test_the_oracle_and_the_paper_read_the_display_s_lines(tmp_path: Path) -> None:
    """u = 0.3; store 100's L1 × 1.3 during: the oracle 13 × 1 300 × 0.3 ÷ 1.3, the paper 8 000 ÷ 56 × 0.3 × 85 (the
    loan's days end at the last order day, `LENT` + 84)."""
    build = _world(tmp_path / "run", lift=0.3)
    loans, _ = _scored(build)
    assert oracle(build, loans, DISPLAYS, GRID).loc[LOAN] == pytest.approx(13 * 1_300 * 0.3 / 1.3)
    assert paper(build, loans, DISPLAYS, GRID).loc[LOAN] == pytest.approx(8_000 / 56 * 0.3 * 85)
    flat = Grid(0.2, {}, {}, flat=0.05)
    assert oracle(build, loans, DISPLAYS, flat).loc[LOAN] == pytest.approx(13 * 1_300 * 0.05 / 1.05)
    assert np.isfinite(paper(build, loans, DISPLAYS, flat).loc[LOAN])
