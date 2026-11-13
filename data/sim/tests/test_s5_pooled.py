"""G2 S2.4 T3's pooled candidate (`checks.s5_pooled`, NH-D19), a prototype: the placebo loans are untreated stores of
the loan's cell and no one's display; the variance is v = σ²_c · X^κ with its fallbacks; the candidate recovers a
lift common to a region and misses one that is not as its weights say; the resampling weights are a store's and a
month's."""

from __future__ import annotations

import csv
import math
from collections.abc import Hashable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hartland_sim.checks.s5_pooled import (
    LIFT,
    Variance,
    counterfactual,
    lifts,
    placebos,
    regions,
    resampling,
    rho,
    spread,
)
from hartland_sim.checks.stories import Build, s5_loans
from hartland_sim.config import StoriesConfig
from hartland_sim.loop.pos import Grid

LENT, BACK = date(2025, 3, 3), date(2025, 6, 1)
CELL = {"store_class": "convenience", "street_type": "high_street", "size_class": "S"}
REAL = 10**12  # a real loan's id: far from the placebos' 1, 2, …


def _world(run: Path, real: int = REAL) -> Build:
    """One region: a display lent `LENT` to `BACK` at store 100 (its orders × 1.2 during); stores 101–108 of its cell,
    107 with a display of its own out until 10 days before `LENT`; 109 of another cell. Every store orders 1 000 a week
    from 8 weeks before `LENT` to `BACK`."""
    rows: dict[str, list[dict[str, Any]]] = {
        "region": [{"region_id": 1, "code": "PHA"}],
        "area": [{"area_id": 1, "region_id": 1}],
        "product": [{"product_id": 1, "product_line_id": 1}],
        "sku": [{"sku_id": 1, "product_id": 1}],
        "pos_asset": [{"pos_asset_id": 1, "region_id": 1, "cost": 36_000.0}],
        "pos_loan": [],
        "store": [],
        "customer": [],
        "delivery_point": [],
        "sales_order": [],
        "sales_order_line": [],
    }
    for store in range(100, 110):
        cell = CELL | ({"size_class": "L"} if store == 109 else {})
        rows["store"].append({"store_id": store, "area_id": 1} | cell)
        rows["customer"].append({"customer_id": store, "channel": "small_retail"})
        rows["delivery_point"].append({"delivery_point_id": store, "customer_id": store, "store_id": store})
        for week in range(-8, 13):
            day = LENT + timedelta(weeks=week)
            k = len(rows["sales_order"]) + 1
            gross = 1200.0 if store == 100 and day >= LENT else 1000.0
            rows["sales_order"].append(
                {"sales_order_id": k, "customer_id": store, "delivery_point_id": store, "ordered_on": day}
            )
            rows["sales_order_line"].append(
                {"sales_order_line_id": k, "sales_order_id": k, "sku_id": 1, "qty_cases": 1.0}
                | {"list_price_case": gross, "delivery_cost_case": None, "warehouse_cost_case": None}
            )
    for k, (store, lent, back) in enumerate(((100, LENT, BACK), (107, LENT - timedelta(100), LENT - timedelta(10)))):
        rows["pos_loan"].append(
            {"pos_loan_id": real + k, "pos_asset_id": 1, "customer_id": store, "store_id": store}
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


def test_a_loan_s_placebos_are_untreated_stores_of_its_cell(tmp_path: Path) -> None:
    """Five of the six free stores (101–106, 108): never its own, nor 107 (a display out in the 56 days before), nor
    109 (another cell); each is scored against the real displays only — its controls are its cell's but itself, 100
    and 107: six — and gains nothing, its store's orders flat. The real loan's gain is the lift, its controls seven."""
    build = _world(tmp_path / "run")
    found = placebos(build)
    mine = found[found["mirrors"] == REAL]
    assert sorted(mine["draw"]) == [0, 1, 2, 3, 4]
    assert mine["store_id"].is_unique and set(mine["store_id"]) <= {101, 102, 103, 104, 105, 106, 108}
    assert set(mine["controls"]) == {6}
    assert mine["gain"].to_numpy() == pytest.approx(np.zeros(5), abs=1e-9)
    real = s5_loans(build)
    assert real.loc[REAL, "controls"] == 7 and real.loc[REAL, "gain"] == pytest.approx(13 * 200.0)
    assert set(found.index) == set(range(1, len(found) + 1))
    assert (found[list(CELL)] == pd.Series(CELL)).all().all() and set(found["channel"]) == {"small_retail"}


def test_the_placebos_are_one_draw_a_seed(tmp_path: Path) -> None:
    build = _world(tmp_path / "run")
    first, again = placebos(build, seed=1), placebos(build, seed=1)
    assert first["store_id"].tolist() == again["store_id"].tolist()
    other = [placebos(build, seed=s).query(f"mirrors == {REAL}")["store_id"].tolist() for s in range(2, 8)]
    assert any(o != first.query(f"mirrors == {REAL}")["store_id"].tolist() for o in other)


def test_a_placebo_id_never_is_a_real_loan_s(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="a placebo's id is a real loan's"):
        placebos(_world(tmp_path / "run", real=3))


def _loans(x: list[float], g: list[float | None], **columns: Any) -> pd.DataFrame:
    """Known loans of one store each (`gain` None: not known) with q = 1 and one day, so X is `before`."""
    n = len(x)
    frame = pd.DataFrame(
        {
            "region": "PHA",
            "gain": pd.array(g, dtype="Float64").to_numpy(dtype=float, na_value=np.nan),
            "before": x,
            "control_before": 1.0,
            "control_during": 1.0,
            "days": 1,
            "new_store": [v == 0 for v in x],
            "cost": 100.0,
            "store_id": pd.array(range(1, n + 1), dtype="Int64"),
            "lent_on": pd.Timestamp(LENT),
            "region_id": 1,
            "channel": "small_retail",
        }
        | CELL
    )
    return frame.assign(**columns)


def _flat(kappa: float = 1.0, scale: float = 1.0, **cells: float) -> Variance:
    by_cell: dict[tuple[Hashable, ...], float] = {
        (1, "small_retail", "convenience", "high_street", size): v for size, v in cells.items()
    }
    return Variance(kappa, by_cell, {"PHA": scale}, 0, 0)


def test_a_lift_common_to_the_region_is_the_sum_whatever_the_weights() -> None:
    """g = 0.1 · X exactly: θ̂ = 0.1 and the candidate is the baseline's Σ g, under any weights or variances."""
    x = [100.0, 400.0, 900.0, 1600.0]
    loans = _loans(x, [0.1 * v for v in x], size_class=["S", "S", "M", "M"])
    found = regions(loans, _flat(S=1.0, M=9.0), np.array([1.0, 2.0, 0.5, 3.0])).loc["PHA"]
    assert found["theta"] == pytest.approx(0.1)
    assert found["candidate"] == pytest.approx(found["baseline"]) == pytest.approx(0.1 * (100 + 800 + 450 + 4800))


def test_a_lift_that_is_not_common_moves_with_the_weights() -> None:
    """The lift is 0.2 in the low-variance cell (S), none in the other: θ̂ leans to S's, so the candidate overstates
    the region's Σ g — as Σ w X g ÷ Σ w X² · Σ X says."""
    x = [100.0, 100.0]
    loans = _loans(x, [20.0, 0.0], size_class=["S", "M"])
    found = regions(loans, _flat(S=1.0, M=4.0)).loc["PHA"]
    theta = (100 * 20 / 100) / (100**2 / 100 + 100**2 / 400)
    assert found["theta"] == pytest.approx(theta) and found["candidate"] == pytest.approx(theta * 200)
    assert found["baseline"] == pytest.approx(20.0) and found["candidate"] > 20


def test_new_stores_unknown_and_negative_bases_are_not_fitted() -> None:
    """A new store (X = 0) keeps its gain apart; an unknown loan counts nowhere but its cost; a store whose margin
    before was negative is in neither (`other`)."""
    loans = _loans([100.0, 0.0, 100.0, -50.0], [10.0, 500.0, None, 7.0])
    found = regions(loans, _flat()).loc["PHA"]
    assert found["theta"] == pytest.approx(0.1) and found["x"] == pytest.approx(100.0)
    assert found["candidate"] == pytest.approx(10.0 + 500.0) and found["new_store"] == pytest.approx(500.0)
    assert (found["other"], found["others"], found["fitted"]) == (7.0, 1, 1)
    assert found["baseline"] == pytest.approx(517.0)
    assert (found["loan_cost"], found["known_cost"]) == (400.0, 300.0)


def test_a_region_without_a_variance_has_no_candidate() -> None:
    loans = _loans([100.0, 200.0], [10.0, 20.0])
    found = regions(loans, Variance(1.0, {}, {}, 0, 0))
    assert found["candidate"].isna().all() and found.at["PHA", "baseline"] == pytest.approx(30.0)
    assert found.at["PHA", "unestimable"] == 2


def test_the_variance_finds_kappa_and_falls_back_to_the_region() -> None:
    """g ~ N(0, 4 · X^1.5): κ's slope and σ²'s mean recover them; a cell under 20 placebos takes its region's."""
    rng = np.random.default_rng(0)
    x = rng.uniform(100, 10_000, 4000)
    g = rng.normal(0, np.sqrt(4 * x**1.5))
    size = np.where(np.arange(4000) < 3990, "S", "M")
    found = Variance.of(_loans(x.tolist(), g.tolist(), size_class=size))
    assert found.kappa == pytest.approx(1.5, abs=0.1) and found.fitted == 4000
    assert list(found.cells) == [(1, "small_retail", "convenience", "high_street", "S")]
    scale = found.scale(_loans([1.0, 1.0], [0.0, 0.0], size_class=["S", "M"]))
    assert scale[1] == pytest.approx(found.regions["PHA"]) and scale[0] == found.cells[next(iter(found.cells))]


def test_the_resampling_weights_are_a_store_s_and_a_month_s() -> None:
    """Two loans at one store in one month weigh alike; a loan at no store is a store of its own; the weights are the
    seed's."""
    loans = pd.DataFrame(
        {
            "store_id": pd.array([1, 1, 2, None, 1], dtype="Int64"),
            "lent_on": pd.to_datetime(["2025-03-03", "2025-03-20", "2025-03-03", "2025-03-03", "2025-04-01"]),
        }
    )
    w = resampling(loans, n=50, seed=7)
    assert w.shape == (50, 5) and (w[:, 0] == w[:, 1]).all() and not (w[:, 0] == w[:, 2]).all()
    assert not (w[:, 3] == w[:, 2]).all() and not (w[:, 0] == w[:, 4]).all()
    assert (resampling(loans, n=50, seed=7) == w).all()
    assert w.mean() == pytest.approx(1.0, abs=0.3)


def test_the_known_lifts() -> None:
    """H0 is 0.1 · X; H1 follows the store's uplift, 0.1 on average in the region; H2 is 0.2 · X in the low-variance
    half."""
    loans = _loans([100.0, 100.0, 100.0, 100.0], [0.0] * 4, street_type=["high_street"] * 2 + ["village"] * 2)
    loans["size_class"] = ["S", "L", "S", "L"]
    grid = Grid.of(StoriesConfig.load())
    assert lifts(loans, _flat(), grid, "H0") == pytest.approx([10.0] * 4)
    h1 = lifts(loans, _flat(), grid, "H1")
    assert (h1 / counterfactual(loans).to_numpy()).mean() == pytest.approx(LIFT)
    assert h1[1] / h1[2] == pytest.approx((1.0 * 1.4) / (0.4 * 0.8))
    h2 = lifts(_loans([100.0] * 4, [0.0] * 4, size_class=["S", "L", "S", "L"]), _flat(S=1.0, L=4.0), grid, "H2")
    assert h2.tolist() == [20.0, 0.0, 20.0, 0.0]
    with pytest.raises(ValueError, match="no design H3"):
        lifts(loans, _flat(), grid, "H3")


def test_spread_and_rho_read_as_the_sql() -> None:
    cost = {"PHA": 10.0, "VCH": 10.0, "JMV": 10.0}
    assert spread({"PHA": 30.0, "VCH": 15.0}, cost) == pytest.approx(2.0)
    assert math.isnan(spread({"PHA": 30.0, "VCH": 0.0}, cost)) and math.isnan(spread({"PHA": 1.0}, {"PHA": 1.0}))
    skill = {"PHA": 1.4, "JMV": 1.3, "VCH": 0.5}
    assert rho({"PHA": 30.0, "JMV": 20.0, "VCH": 15.0}, cost, skill) == pytest.approx(1.0)
