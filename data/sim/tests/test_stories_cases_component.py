"""G0 S0.4 T3, the component tier: the detectors' and probes' definitions on hand-built rows. Each case writes only the
rows its SQL reads (conftest's `put`), inside a transaction that rolls back; the foreign keys are deferred, so the
rest is never missed. G5 S5.2 tunes the stories' parameters on the build; these cases pin what the SQL means."""

from __future__ import annotations

import csv
import itertools
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import pytest

from hartland_sim.checks.stories import Build, a6, s5, s5_best_worst, s5_loans, s5_regions, s6
from hartland_sim.config import STORIES_DIR, StoriesConfig

pytestmark = pytest.mark.component

Put = Callable[..., None]  # conftest's `put`

_ids = itertools.count(10_000)


@pytest.fixture(scope="module")
def db(hartland_db: psycopg.Connection) -> psycopg.Connection:
    return hartland_db


def _one(db: psycopg.Connection, name: str) -> Any:
    """The file's one number."""
    db.execute("SET LOCAL search_path TO hartland")
    ((value,),) = db.execute((STORIES_DIR / name).read_text(encoding="utf-8")).fetchall()
    return value


def _cte(db: psycopg.Connection, name: str, cte: str) -> dict[Any, dict[str, Any]]:
    """The rows of the CTE `cte` of the file, by their first column, each as a mapping of its columns."""
    sql = (STORIES_DIR / name).read_text(encoding="utf-8")
    final = re.search(r"^SELECT", sql, re.M)  # the final SELECT starts its line; the CTEs' bodies are indented
    assert final, name
    db.execute("SET LOCAL search_path TO hartland")
    cursor = db.execute(sql[: final.start()] + f"SELECT * FROM {cte}")
    assert cursor.description is not None
    names = [column.name for column in cursor.description]
    return {row[0]: dict(zip(names, row, strict=True)) for row in cursor.fetchall()}


def _catalogue(db: psycopg.Connection, put: Put, lines: int = 1, code: str | None = None) -> None:
    """Lines 1…n, each with one product and one SKU of the same id."""
    for n in range(1, lines + 1):
        put(db, "product_line", product_line_id=n, code=code or f"L{n}")
        put(db, "product", product_id=n, product_line_id=n)
        put(db, "sku", sku_id=n, product_id=n)


def _sale(
    db: psycopg.Connection,
    put: Put,
    *,
    customer: int,
    day: date,
    gross: float = 100,
    line: int = 1,
    store: int | None = None,
    agreed: float = 0,
    channel: str = "small_retail",
) -> None:
    """One order of one case: `gross` CZK, of which `agreed` the agreed discount (sales_pl's customer margin is the
    difference). Okres 1 must exist."""
    n = next(_ids)
    put(db, "delivery_point", delivery_point_id=n, customer_id=customer, store_id=store, okres_id=1)
    put(db, "sales_order", sales_order_id=n, customer_id=customer, delivery_point_id=n, channel=channel, ordered_on=day)
    put(
        db, "sales_order_line", sales_order_line_id=n, sales_order_id=n, sku_id=line, qty_cases=1, list_price_case=gross
    )
    if agreed:
        put(db, "sales_order_line_discount", sales_order_line_id=n, seq=1, kind="agreed", amount_line=agreed)


def _last_order_day(db: psycopg.Connection, put: Put, day: date) -> None:
    """The world's last order day, which the detectors read as `as_of`."""
    put(db, "sales_order", sales_order_id=next(_ids), ordered_on=day)


TENURE = {"valid_from": date(2023, 1, 1), "valid_to": date(2026, 12, 31)}


# ── S2: like against like ────────────────────────────────────────────────────


def test_s2_compares_mechanic_with_mechanic(db: psycopg.Connection, put: Put) -> None:
    """Premium and basic gain alike per mechanic, but the premium side runs two 1+1s to one price-off: the metric is
    1.0 (a pooled mean gave 1.13)."""
    with db.transaction(force_rollback=True):
        put(db, "campaign", campaign_id=1, has_tv=True)
        for line, premium in ((1, True), (2, False)):
            put(db, "product_line", product_line_id=line, is_premium=premium)
        runs = [(1, "price_off", 20, 40), (1, "free_goods_same", None, 90), (1, "free_goods_same", None, 90)]
        runs += [(2, "price_off", 20, 40), (2, "free_goods_same", None, 90)]
        for line, mechanic, discount, uplift in runs:
            promotion = next(_ids)
            put(
                db,
                "promotion",
                promotion_id=promotion,
                campaign_id=1,
                target_product_line_id=line,
                mechanic=mechanic,
                discount_pct=discount,
            )
            put(db, "promotion_result", promotion_id=promotion, week=date(2025, 3, 3), uplift_pct=uplift)
        metric = _one(db, "detectors/s2.sql")
    assert float(metric) == pytest.approx(1.0)


def _tv_promotion(db: psycopg.Connection, put: Put) -> None:
    """Line 1's TV promotion, 2025-11-10…12-21 (42 days), and its 1+1 in the 8 weeks before, 09-08…10-05 (35 clean
    days before are left). Four okresy, one a quarter: 1 the poorest, 4 the richest."""
    _catalogue(db, put)
    for okres, affluence in ((1, 0.8), (2, 0.9), (3, 1.0), (4, 1.3)):
        put(db, "okres", okres_id=okres, affluence_index=affluence)
    put(db, "campaign", campaign_id=1, has_tv=True)
    put(
        db,
        "promotion",
        promotion_id=1,
        campaign_id=1,
        target_product_line_id=1,
        valid_from=date(2025, 11, 10),
        valid_to=date(2025, 12, 21),
    )
    put(
        db,
        "promotion",
        promotion_id=2,
        target_product_line_id=1,
        valid_from=date(2025, 9, 8),
        valid_to=date(2025, 10, 5),
    )
    _last_order_day(db, put, date(2026, 10, 31))


def _cases(
    db: psycopg.Connection,
    put: Put,
    okres: int,
    day: date,
    qty: float,
    *,
    store: bool = True,
    channel: str = "small_retail",
) -> None:
    """One order of `qty` cases of line 1, delivered in `okres`: to a store, or to a place without one (a DC, a depot,
    an own shop, a web consumer's town)."""
    n = next(_ids)
    put(db, "delivery_point", delivery_point_id=n, okres_id=okres, store_id=n if store else None)
    put(db, "sales_order", sales_order_id=n, delivery_point_id=n, channel=channel, ordered_on=day)
    put(db, "sales_order_line", sales_order_line_id=n, sales_order_id=n, sku_id=1, qty_cases=qty)


def test_s2_affluence_reads_the_clean_days_only(db: psycopg.Connection, put: Put) -> None:
    """Gold's 1+1 surges in the TV promotion's 8 weeks before; the days it ran are left out on both sides. The rich
    okres doubles its daily cases, the poor one keeps them: lift 2 over lift 1."""
    with db.transaction(force_rollback=True):
        _tv_promotion(db, put)
        _cases(db, put, 4, date(2025, 9, 20), 350)  # the 1+1's surge
        _cases(db, put, 4, date(2025, 10, 20), 35)
        _cases(db, put, 4, date(2025, 11, 20), 84)
        _cases(db, put, 1, date(2025, 10, 20), 35)
        _cases(db, put, 1, date(2025, 11, 20), 42)
        metric = _one(db, "detectors/s2_affluence.sql")
    assert float(metric) == pytest.approx(2.0)


def test_s2_affluence_reads_the_stores_and_the_web_not_the_dcs(db: psycopg.Connection, put: Put) -> None:
    """A chain that orders to its own DC puts its national volume in the DC's okres, a rich one here: the order leaves
    the metric at 2. A web consumer's order counts where the consumer lives: doubling the poor okres's cases during
    the promotion, it brings the metric to 1."""
    with db.transaction(force_rollback=True):
        _tv_promotion(db, put)
        _cases(db, put, 4, date(2025, 10, 20), 35)
        _cases(db, put, 4, date(2025, 11, 20), 84)
        _cases(db, put, 1, date(2025, 10, 20), 35)
        _cases(db, put, 1, date(2025, 11, 20), 42)
        _cases(db, put, 4, date(2025, 11, 21), 5000, store=False, channel="big_retail")  # to the chain's DC
        assert float(_one(db, "detectors/s2_affluence.sql")) == pytest.approx(2.0)
        _cases(db, put, 1, date(2025, 11, 21), 42, store=False, channel="web")
        metric = _one(db, "detectors/s2_affluence.sql")
    assert float(metric) == pytest.approx(1.0)


def test_s2_affluence_reads_the_own_shops_not_the_depots(db: psycopg.Connection, put: Put) -> None:
    """An own shop is a site, not a store, and sells where it stands: its order doubles the poor okres's cases during
    the promotion and brings the metric to 1, as a web order does. A vending operator's depot is not where its
    machines are: its order in the rich okres leaves the metric at 1."""
    with db.transaction(force_rollback=True):
        _tv_promotion(db, put)
        _cases(db, put, 4, date(2025, 10, 20), 35)
        _cases(db, put, 4, date(2025, 11, 20), 84)
        _cases(db, put, 1, date(2025, 10, 20), 35)
        _cases(db, put, 1, date(2025, 11, 20), 42)
        _cases(db, put, 1, date(2025, 11, 21), 42, store=False, channel="own_shops")
        assert float(_one(db, "detectors/s2_affluence.sql")) == pytest.approx(1.0)
        _cases(db, put, 4, date(2025, 11, 21), 5000, store=False, channel="vending")  # to the operator's depot
        metric = _one(db, "detectors/s2_affluence.sql")
    assert float(metric) == pytest.approx(1.0)


# ── S4: every street type judged ─────────────────────────────────────────────


def test_s4_is_null_when_a_street_type_has_no_ordering_store(db: psycopg.Connection, put: Put) -> None:
    """The high street's one store ordered in 2024 only: the detector cannot judge the high street, so the metric is
    NULL, not the other two types' smallest share. Its 2025 order of Gold brings the metric to 1."""

    def order(store: int, line: int, day: date) -> None:
        n = next(_ids)
        put(db, "delivery_point", delivery_point_id=n, okres_id=1, store_id=store)
        put(db, "sales_order", sales_order_id=n, delivery_point_id=n, ordered_on=day)
        put(db, "sales_order_line", sales_order_line_id=n, sales_order_id=n, sku_id=line, qty_cases=5)

    with db.transaction(force_rollback=True):
        for n, code in enumerate(("SMESI", "POLEVKY", "GOLD"), start=1):
            put(db, "product_line", product_line_id=n, code=code)
            put(db, "product", product_id=n, product_line_id=n)
            put(db, "sku", sku_id=n, product_id=n)
        for store, street in ((1, "transport_hub"), (2, "village"), (3, "high_street")):
            put(db, "store", store_id=store, street_type=street)
        order(1, 1, date(2025, 3, 3))
        order(2, 2, date(2025, 3, 3))
        order(3, 3, date(2024, 3, 4))  # 2024: outside S4's window
        assert _one(db, "detectors/s4.sql") is None
        order(3, 3, date(2025, 3, 3))
        metric = _one(db, "detectors/s4.sql")
    assert float(metric) == pytest.approx(1.0)


# ── S1: the region's weeks ───────────────────────────────────────────────────


def test_s1_weather_reads_the_stores_not_the_dcs(db: psycopg.Connection, put: Put) -> None:
    """A chain that orders to its own DC puts its national volume in the DC's okres, and so in the DC's region: the
    region's week counts the store's 10 cases alone."""
    with db.transaction(force_rollback=True):
        _catalogue(db, put, code="POLEVKY")
        put(db, "okres", okres_id=1)
        put(db, "area", area_id=1, region_id=1)
        put(db, "area_okres", area_id=1, okres_id=1)
        _cases(db, put, 1, date(2025, 1, 15), 10)
        _cases(db, put, 1, date(2025, 1, 16), 5000, store=False, channel="big_retail")  # to the chain's DC
        weeks = _cte(db, "detectors/s1_weather.sql", "weekly_cases")
    assert float(weeks[1]["cases"]) == pytest.approx(10)


# ── S3: rates over their bases ───────────────────────────────────────────────


def test_s3_acquisition_is_per_potential_store_not_per_day(db: psycopg.Connection, put: Put) -> None:
    """Two reps win one store in fifty: one in an area of 50 stores, one in an area of 200. Their acquisition rates
    are equal; per duty day they were 4× apart."""
    with db.transaction(force_rollback=True):
        for rep, area, stores, won in ((1, 10, 50, 1), (2, 20, 200, 4)):
            put(db, "area", area_id=area)
            put(db, "assignment", assignment_id=rep, employee_id=rep, kind="rep_area", area_id=area, **TENURE)
            for k in range(stores):
                store = area * 1000 + k
                put(db, "store", store_id=store, area_id=area, store_class="minimarket")
                if k < won:
                    since = date(2024, 6, 1)
                    put(db, "customer", customer_id=store, channel="small_retail", store_id=store, since=since)
        _last_order_day(db, put, date(2024, 12, 31))
        rates = _cte(db, "detectors/s3.sql", "rates")
    acquisition = {rep: float(rates[rep]["acquisition"]) for rep in (1, 2)}
    assert acquisition[1] == pytest.approx(1 / (50 * 366 - 214))
    assert acquisition[2] == pytest.approx(acquisition[1])


def test_s3_the_historys_first_orders_add_no_line(db: psycopg.Connection, put: Put) -> None:
    """A customer since 2022 buying its four usual lines in January 2024 adds nothing; a fifth line in July does."""
    with db.transaction(force_rollback=True):
        _catalogue(db, put, lines=5)
        put(db, "okres", okres_id=1)
        put(db, "area", area_id=10)
        put(db, "assignment", assignment_id=1, employee_id=1, kind="rep_area", area_id=10, **TENURE)
        put(db, "store", store_id=1, area_id=10, store_class="minimarket")
        put(db, "customer", customer_id=1, channel="small_retail", store_id=1, since=date(2022, 5, 1))
        for line in range(1, 5):
            _sale(db, put, customer=1, store=1, line=line, day=date(2024, 1, 15))
        assert _cte(db, "detectors/s3.sql", "rates")[1]["cross_sell"] == 0
        _sale(db, put, customer=1, store=1, line=5, day=date(2024, 7, 1))
        rates = _cte(db, "detectors/s3.sql", "rates")
    assert float(rates[1]["cross_sell"]) == pytest.approx(1 / 183)  # one line ÷ 2024-01-01…07-01


def test_s3_counts_small_retail_only(db: psycopg.Connection, put: Put) -> None:
    """A HoReCa customer won in the area (the KAMs') is not the rep's win."""
    with db.transaction(force_rollback=True):
        put(db, "area", area_id=10)
        put(db, "assignment", assignment_id=1, employee_id=1, kind="rep_area", area_id=10, **TENURE)
        put(db, "store", store_id=1, area_id=10, store_class="minimarket")
        put(db, "store", store_id=2, area_id=10, store_class="horeca")
        put(db, "customer", customer_id=2, channel="horeca", store_id=2, since=date(2024, 6, 1))
        _last_order_day(db, put, date(2024, 12, 31))
        rates = _cte(db, "detectors/s3.sql", "rates")
    assert rates[1]["acquisition"] == 0


def test_s3_conversion_counts_the_market_only(db: psycopg.Connection, put: Put) -> None:
    """The rep converts one visit in two at an independent. Two visits without an order at a key-account chain's
    station, whose orders come through the key account, leave the rate at a half (counted, they gave a quarter)."""
    with db.transaction(force_rollback=True):
        put(db, "area", area_id=10)
        put(db, "assignment", assignment_id=1, employee_id=1, kind="rep_area", area_id=10, **TENURE)
        put(db, "chain", chain_id=1, is_key_account=True)
        put(db, "store", store_id=1, area_id=10, store_class="minimarket")
        put(db, "store", store_id=2, area_id=10, store_class="petrol", chain_id=1)
        put(db, "customer", customer_id=1, channel="small_retail", store_id=1)
        put(db, "customer", customer_id=2, channel="gas_ka", store_id=2)
        for k, (store, outcome) in enumerate(((1, "order"), (1, "no_order"), (2, "no_order"), (2, "no_order"))):
            day = date(2024, 3, 4) + timedelta(days=k)
            put(
                db,
                "visit",
                visit_id=k,
                employee_id=1,
                store_id=store,
                customer_id=store,
                day=day,
                kind="sales",
                outcome=outcome,
            )
        _last_order_day(db, put, date(2024, 12, 31))
        rates = _cte(db, "detectors/s3.sql", "rates")
    assert float(rates[1]["conversion"]) == pytest.approx(0.5)


# ── S5: the gain of a display (NH-D17: against the untreated stores of its cell) ──


def _pos(db: psycopg.Connection, put: Put) -> None:
    _catalogue(db, put)
    put(db, "okres", okres_id=1)
    put(db, "region", region_id=1, code="PHA")
    put(db, "pos_asset", pos_asset_id=1, region_id=1, cost=36_000)


def test_s5_a_display_at_a_chain_store_reads_that_store(db: psycopg.Connection, put: Put) -> None:
    """A loan at one store of a chain customer reads that store's sales, not the chain's; with no store of its cell to
    compare, its gain is not known — S2.3's raw gain still reads the store."""
    with db.transaction(force_rollback=True):
        _pos(db, put)
        put(db, "customer", customer_id=1, channel="big_retail")
        for store in (1, 2):
            put(db, "store", store_id=store)
        put(db, "pos_loan", pos_loan_id=1, pos_asset_id=1, customer_id=1, store_id=1, lent_on=date(2025, 3, 3))
        _sale(db, put, customer=1, store=1, day=date(2025, 4, 1), gross=1000, channel="big_retail")
        _sale(db, put, customer=1, store=2, day=date(2025, 4, 1), gross=5000, channel="big_retail")
        db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        margin = _cte(db, "detectors/s5.sql", "margin")
    assert float(margin[1]["raw_gain"]) == pytest.approx(1000) and margin[1]["gain"] is None


def test_s5_a_loan_outside_the_history_is_ignored(db: psycopg.Connection, put: Put) -> None:
    """Lent after the world's last order day (no negative days), or before 2024-02-26 (no 8 weeks before)."""
    with db.transaction(force_rollback=True):
        _pos(db, put)
        put(db, "customer", customer_id=1)
        for loan, lent_on in ((1, date(2025, 3, 3)), (2, date(2025, 6, 1)), (3, date(2024, 1, 15))):
            put(db, "pos_loan", pos_loan_id=loan, pos_asset_id=1, customer_id=1, lent_on=lent_on)
        _last_order_day(db, put, date(2025, 5, 31))
        db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        margin = _cte(db, "detectors/s5.sql", "margin")
    assert set(margin) == {1}


def test_s5_counts_the_giveaways_from_the_loans_first_day(db: psycopg.Connection, put: Put) -> None:
    """The loans count from 2024-02-26, so the give-aways do too: one handed out in January is not charged."""
    with db.transaction(force_rollback=True):
        _pos(db, put)
        put(db, "area", area_id=1, region_id=1)
        put(db, "store", store_id=1, area_id=1)
        for k, day in enumerate((date(2024, 1, 15), date(2024, 3, 4))):
            put(db, "visit", visit_id=k, store_id=1, day=day)
            put(db, "pos_handout", pos_handout_id=k, visit_id=k, cost=100 * (k + 1))
        _last_order_day(db, put, date(2025, 5, 31))
        db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")  # the block's plan reads it, whichever CTE is asked
        giveaway = _cte(db, "detectors/s5.sql", "giveaway")
    assert float(giveaway["PHA"]["cost"]) == pytest.approx(200)


CELL = {"store_class": "convenience", "street_type": "high_street", "size_class": "S"}
LENT, BACK = date(2025, 3, 3), date(2025, 6, 1)  # 91 days; an order a week from 8 weeks before: 8 before, 13 during


@dataclass
class Case:
    """Six regions (or those of `lifts`), each with one display lent `LENT` to `BACK` at its store and `controls` stores
    of the same cell: every store orders once a week from 8 weeks before, 1 000 CZK before; during the loan the
    controls' orders × `drift`, the display's store's × `drift` × 1 + the region's lift; and one give-away of the
    region's cost. `rows` are the tables; `loan`, `store` add to them."""

    lifts: dict[str, float]
    giveaways: dict[str, float]
    drift: float = 1.0
    controls: int = 5
    before: float = 1000.0  # the controls' order before
    rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rows = {name: [] for name in TABLES}
        self.rows |= {
            "okres": [{"okres_id": 1}],
            "product_line": [{"product_line_id": 1, "code": "L1"}],
            "product": [{"product_id": 1, "product_line_id": 1}],
            "sku": [{"sku_id": 1, "product_id": 1}],
        }
        for r, code in enumerate(self.lifts, start=1):
            self.rows["region"].append({"region_id": r, "code": code})
            self.rows["area"].append({"area_id": r, "region_id": r})
            self.rows["pos_asset"].append({"pos_asset_id": r, "region_id": r, "cost": 36_000.0})
            self.store(100 * r, r, 1000.0, 1000.0 * self.drift * (1 + self.lifts[code]))
            for c in range(1, self.controls + 1):
                self.store(100 * r + c, r, self.before, self.before * self.drift)
            self.loan(r, r, 100 * r)
            k = next(_ids)
            self.rows["visit"].append({"visit_id": k, "store_id": 100 * r, "day": date(2025, 4, 7)})
            self.rows["pos_handout"].append({"pos_handout_id": k, "visit_id": k, "cost": self.giveaways[code]})
        self.rows["delivery_point"].append({"delivery_point_id": 999, "customer_id": 999, "okres_id": 1})
        self.order(999, BACK, 1000.0)  # the world's last order day, at no store

    def store(self, store: int, area: int, before: float, during: float, **cell: str) -> None:
        """A store of `area`, its customer and delivery point (all of id `store`), ordering weekly `before`, then
        `during` from `LENT`."""
        self.rows["store"].append({"store_id": store, "area_id": area} | CELL | cell)
        self.rows["customer"].append({"customer_id": store, "store_id": store, "channel": "small_retail"})
        self.rows["delivery_point"].append(
            {"delivery_point_id": store, "customer_id": store, "store_id": store, "okres_id": 1}
        )
        for week in range(-8, 13):
            day = LENT + timedelta(weeks=week)
            self.order(store, day, during if day >= LENT else before)

    def order(self, point: int, day: date, gross: float) -> None:
        """One order of one case at the delivery point `point` (its customer's id is the point's)."""
        k = next(_ids)
        self.rows["sales_order"].append(
            {"sales_order_id": k, "customer_id": point, "delivery_point_id": point, "channel": "small_retail"}
            | {"ordered_on": day}
        )
        self.rows["sales_order_line"].append(
            {"sales_order_line_id": k, "sales_order_id": k, "sku_id": 1, "qty_cases": 1.0}
            | {"list_price_case": round(gross, 2), "delivery_cost_case": None, "warehouse_cost_case": None}
        )

    def loan(self, loan: int, asset: int, store: int, lent: date = LENT, back: date | None = BACK) -> None:
        self.rows["pos_loan"].append(
            {"pos_loan_id": loan, "pos_asset_id": asset, "customer_id": store, "store_id": store}
            | {"lent_on": lent, "returned_on": back}
        )


TABLES = (
    "region",
    "area",
    "store",
    "customer",
    "delivery_point",
    "pos_asset",
    "pos_loan",
    "visit",
    "pos_handout",
    "sales_order",
    "sales_order_line",
    "sales_order_line_discount",
)


@dataclass(frozen=True)
class Both:
    """S5 on a `Case`, in SQL and in the twins: the loans (`margin`, by loan), the regions (`region_return`, by region)
    and the two metrics."""

    margin: dict[Any, dict[str, Any]]
    regions: dict[Any, dict[str, Any]]
    rho: Any
    spread: Any
    loans: pd.DataFrame
    found: pd.DataFrame
    rho_twin: float
    spread_twin: float


def _csv_cell(value: Any) -> str:
    return "" if value is None else value.isoformat() if isinstance(value, date) else str(value)


def _s5_both(db: psycopg.Connection, put: Put, case: Case, run: Path) -> Both:
    run.mkdir(parents=True, exist_ok=True)
    headers = {"sales_order_line_discount": ["sales_order_line_id", "seq", "kind", "amount_line"]}
    for table, found in case.rows.items():
        with (run / f"{table}.csv").open("w", encoding="utf-8", newline="") as out:
            columns = list(dict.fromkeys(k for row in found for k in row)) if found else headers[table]
            writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in found:
                writer.writerow({k: _csv_cell(row.get(k)) for k in columns})
    build, stories = Build(run), StoriesConfig.load()
    twins = s5_loans(build), s5_regions(build).set_index("region"), s5(build, stories), s5_best_worst(build, stories)
    with db.transaction(force_rollback=True):
        for table, found in case.rows.items():
            for row in found:
                put(db, table, **row)
        db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        margin = _cte(db, "detectors/s5.sql", "margin")
        regions = _cte(db, "detectors/s5.sql", "region_return")
        rho, spread = _one(db, "detectors/s5.sql"), _one(db, "detectors/s5_best_worst.sql")
    return Both(margin, regions, rho, spread, *twins)


def _gains(both: Both) -> tuple[dict[int, float | None], dict[int, float | None]]:
    """Each loan's gain in SQL and in the twin, None where it is not known."""
    sql_gains = {k: None if v["gain"] is None else float(v["gain"]) for k, v in both.margin.items()}
    twin = {int(str(k)): None if math.isnan(v) else float(v) for k, v in both.loans["gain"].items()}
    return sql_gains, twin


SIX = {"PHA": 0.8, "JMV": 0.6, "SJC": 0.4, "ZSC": 0.2, "VCH": 0.15, "SMS": 0.1}


def test_s5_a_common_drift_alone_gains_nothing(db: psycopg.Connection, put: Put, tmp_path: Path) -> None:
    """T1(a): every store sells 20 % more during the loan and no display lifts it: S2.3's raw gain is the drift,
    13 000 × 0.2 at each display; the controls' q is 1.2, so the gain is 0 — in SQL and in the twin."""
    both = _s5_both(db, put, Case(dict.fromkeys(SIX, 0.0), dict.fromkeys(SIX, 500.0), drift=1.2), tmp_path)
    sql_gains, twin = _gains(both)
    assert sql_gains == pytest.approx(dict.fromkeys(range(1, 7), 0.0), abs=1e-6)
    assert twin == pytest.approx(sql_gains, abs=1e-6)
    assert [float(both.margin[k]["raw_gain"]) for k in range(1, 7)] == pytest.approx([2600.0] * 6)


def test_s5_a_known_lift_survives_a_common_drift(db: psycopg.Connection, put: Put, tmp_path: Path) -> None:
    """T1(b): with the same drift, each display's store sells × 1 + its lift on top: the gain is the lift on the drifted
    sales, 13 000 × 1.2 × lift."""
    both = _s5_both(db, put, Case(SIX, dict.fromkeys(SIX, 500.0), drift=1.2), tmp_path)
    sql_gains, twin = _gains(both)
    assert sql_gains == pytest.approx({r: 13_000 * 1.2 * lift for r, lift in enumerate(SIX.values(), 1)})
    assert twin == pytest.approx(sql_gains, rel=1e-9)
    assert {float(r["coverage"]) for r in both.regions.values()} == {1.0}


def test_s5_a_loan_without_five_controls_is_not_known(db: psycopg.Connection, put: Put, tmp_path: Path) -> None:
    """T1(c): four stores of its cell are not enough: no loan's gain is known, every region's coverage is 0 and both
    metrics are NULL (NaN in the twins), the costs still counted."""
    both = _s5_both(db, put, Case(SIX, dict.fromkeys(SIX, 500.0), controls=4), tmp_path)
    sql_gains, twin = _gains(both)
    assert set(sql_gains.values()) == {None} and set(twin.values()) == {None}
    assert {float(r["coverage"]) for r in both.regions.values()} == {0.0}
    assert all(float(r["cost"]) > 500 for r in both.regions.values())
    assert (both.rho, both.spread) == (None, None)
    assert math.isnan(both.rho_twin) and math.isnan(both.spread_twin)


def test_s5_controls_that_sold_nothing_before_are_no_base(db: psycopg.Connection, put: Put, tmp_path: Path) -> None:
    """T1(c): five controls whose margin before is 0 give no q: the gain is not known."""
    both = _s5_both(db, put, Case(SIX, dict.fromkeys(SIX, 500.0), before=0.0), tmp_path)
    sql_gains, twin = _gains(both)
    assert set(sql_gains.values()) == {None} and set(twin.values()) == {None}


def test_s5_a_store_with_a_display_out_is_no_control(db: psycopg.Connection, put: Put, tmp_path: Path) -> None:
    """A control with a display of its own out in the 56 days before PHA's loan is left out (four remain: not known);
    one whose display came back the day before those 56 days stays a control (JMV keeps five)."""
    case = Case(SIX, dict.fromkeys(SIX, 500.0))
    case.rows["pos_asset"].append({"pos_asset_id": 7, "region_id": 1, "cost": 36_000.0})
    case.loan(7, 7, 101, lent=LENT - timedelta(days=120), back=LENT - timedelta(days=56))
    case.loan(8, 7, 201, lent=LENT - timedelta(days=120), back=LENT - timedelta(days=57))
    both = _s5_both(db, put, case, tmp_path)
    sql_gains, twin = _gains(both)
    assert (sql_gains[1], twin[1]) == (None, None)
    assert sql_gains[2] == pytest.approx(13_000 * 0.6) and twin[2] == pytest.approx(13_000 * 0.6)


def test_s5_a_new_store_gains_all_it_buys_and_is_reported_apart(
    db: psycopg.Connection, put: Put, tmp_path: Path
) -> None:
    """A display at a store that bought nothing in the 56 days before: its gain is all it buys during (13 000), kept in
    the region's gain and reported apart (`new_store_gain`)."""
    case = Case(SIX, dict.fromkeys(SIX, 500.0))
    case.rows["pos_asset"].append({"pos_asset_id": 7, "region_id": 1, "cost": 36_000.0})
    case.store(150, 1, 0.0, 1000.0)
    case.loan(7, 7, 150)
    both = _s5_both(db, put, case, tmp_path)
    sql_gains, twin = _gains(both)
    assert sql_gains[7] == pytest.approx(13_000) and twin[7] == pytest.approx(13_000)
    assert float(both.regions["PHA"]["new_store_gain"]) == pytest.approx(13_000)
    assert both.found.loc["PHA", "new_store_gain"] == pytest.approx(13_000)
    assert float(both.regions["PHA"]["gain"]) == pytest.approx(13_000 * 1.8)
    assert float(both.regions["PHA"]["coverage"]) == pytest.approx(1.0)
    assert float(both.regions["PHA"]["identified"]) == pytest.approx(0.5) == both.found.loc["PHA", "identified"]


@pytest.mark.parametrize(("other_cost", "known"), [(3_000.0, True), (4_800.0, False)])
def test_s5_under_nine_tenths_of_a_region_s_loan_cost_known_both_metrics_are_null(
    db: psycopg.Connection, put: Put, tmp_path: Path, other_cost: float, known: bool
) -> None:
    """A second display in PHA, at a store of another cell (no controls): at 3 000 of PHA's 39 000 of asset cost,
    0.92 is known and the metrics stand; at 4 800 of 40 800 (0.88) both are NULL, in SQL and in the twins."""
    case = Case(SIX, dict.fromkeys(SIX, 500.0))
    case.rows["pos_asset"].append({"pos_asset_id": 7, "region_id": 1, "cost": other_cost})
    case.store(150, 1, 1000.0, 1000.0, size_class="L")
    case.loan(7, 7, 150)
    both = _s5_both(db, put, case, tmp_path)
    coverage = float(both.regions["PHA"]["coverage"])
    assert coverage == pytest.approx(both.found.loc["PHA", "coverage"]) and (coverage >= 0.9) == known
    assert (both.rho is not None, both.spread is not None) == (known, known)
    assert (math.isnan(both.rho_twin), math.isnan(both.spread_twin)) == (not known, not known)


def test_s5_the_twins_are_the_sql_s_values(db: psycopg.Connection, put: Put, tmp_path: Path) -> None:
    """T1(d): the returns fall with the managers' skill: rho is the SQL's on average ranks (VCH and SMS share theirs),
    and the spread PHA's gain ÷ cost over VCH's, in SQL as in the twins; the regions' columns agree."""
    both = _s5_both(db, put, Case(SIX, dict.fromkeys(SIX, 500.0), drift=1.1), tmp_path)
    assert float(both.rho) == pytest.approx(both.rho_twin, rel=1e-9) and both.rho_twin > 0.98
    assert float(both.spread) == pytest.approx(both.spread_twin, rel=1e-9) and both.spread_twin > 3
    for region, row in both.regions.items():
        columns = ("gain", "cost", "raw_gain", "new_store_gain", "loan_cost", "coverage", "identified", "loans")
        for column in (*columns, "unknown", "new_stores"):
            assert float(row[column]) == pytest.approx(both.found.loc[region, column], rel=1e-9), (region, column)


def test_s5_tied_returns_share_their_rank_in_both(db: psycopg.Connection, put: Put, tmp_path: Path) -> None:
    """T1(d): ZSC and SJC lift alike and pay alike, so their returns tie and share a rank, as VCH's and SMS's skills do:
    the SQL's rho and the twin's are one number."""
    lifts = SIX | {"ZSC": 0.4}
    both = _s5_both(db, put, Case(lifts, dict.fromkeys(lifts, 500.0)), tmp_path)
    assert float(both.rho) == pytest.approx(both.rho_twin, rel=1e-9) and 0.8 < both.rho_twin < 1


def test_s5_a_worst_region_destroying_margin_is_null_in_both(db: psycopg.Connection, put: Put, tmp_path: Path) -> None:
    """T1(e): VCH's displays lose margin: its gain ÷ cost is ≤ 0, so the spread is NULL — NaN in the twin."""
    lifts = SIX | {"VCH": -0.3}
    both = _s5_both(db, put, Case(lifts, dict.fromkeys(lifts, 500.0)), tmp_path)
    assert both.spread is None and math.isnan(both.spread_twin)


def test_s5_a_region_without_cost_has_no_spread(db: psycopg.Connection, put: Put, tmp_path: Path) -> None:
    """T1(e), NH-D19: VCH's display cost nothing and it handed out nothing: gain ÷ cost has no denominator and its
    coverage none — VCH is not covered, so both metrics are NULL (NaN in the twins)."""
    case = Case(SIX, dict.fromkeys(SIX, 500.0) | {"VCH": 0.0})
    (vch,) = [a for a in case.rows["pos_asset"] if a["region_id"] == 5]
    vch["cost"] = 0.0
    both = _s5_both(db, put, case, tmp_path)
    assert (both.rho, both.spread) == (None, None)
    assert math.isnan(both.rho_twin) and math.isnan(both.spread_twin)


@pytest.mark.parametrize("lent", [None, date(2023, 12, 4)])
def test_s5_a_region_without_a_loan_is_not_covered(
    db: psycopg.Connection, put: Put, tmp_path: Path, lent: date | None
) -> None:
    """NH-D19: all six regions of `region` must be covered. Five lift and pay as in the twins' case; SMS, the sixth, has
    no loan, or only one lent before the history (not counted): it is in no region return, and both metrics are NULL
    — in SQL and in the twins, which judge the five present no more than the SQL does."""
    five = {r: lift for r, lift in SIX.items() if r != "SMS"}
    case = Case(five, dict.fromkeys(five, 500.0), drift=1.1)
    case.rows["region"].append({"region_id": 6, "code": "SMS"})
    case.rows["area"].append({"area_id": 6, "region_id": 6})
    if lent:
        case.rows["pos_asset"].append({"pos_asset_id": 6, "region_id": 6, "cost": 36_000.0})
        case.store(600, 6, 1000.0, 1000.0)
        case.loan(6, 6, 600, lent=lent, back=lent + timedelta(days=60))
    both = _s5_both(db, put, case, tmp_path)
    assert "SMS" not in both.regions and "SMS" not in both.found.index
    assert (both.rho, both.spread) == (None, None)
    assert math.isnan(both.rho_twin) and math.isnan(both.spread_twin)


# ── S6: the twin's measure in SQL (G1 S1.5 T1(d)) ─────────────────────────────


def _s6_in_sql(db: psycopg.Connection, put: Put, s6_case: Any, run: Path) -> tuple[Any, float]:
    """s6.sql's metric on conftest's `s6_case` and the pandas twin's on its CSV build. `sales_pl`'s Customer Margin is
    the net revenue where the costs are NULL, never NULL itself."""
    s6_case.write(run)
    twin = s6(Build(run), StoriesConfig.load())
    with db.transaction(force_rollback=True):
        for table, rows in s6_case.tables.items():
            for row in rows:
                put(db, table, **row)
        db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        ((nulls,),) = db.execute("SELECT count(*) FROM hartland.sales_pl WHERE customer_margin IS NULL").fetchall()
        metric = _one(db, "detectors/s6.sql")
    assert nulls == 0
    return metric, twin


def test_s6_is_the_twin_s_value_on_the_hand_built_case(
    db: psycopg.Connection, put: Put, s6_case: Any, tmp_path: Path
) -> None:
    """The disciplined customer's gain a crown less the undisciplined one's: a difference, so the negative side
    holds."""
    metric, twin = _s6_in_sql(db, put, s6_case, tmp_path)
    assert twin > 0.5 and float(metric) == pytest.approx(twin, rel=1e-6)


def test_s6_leaves_out_what_the_twin_leaves_out(
    db: psycopg.Connection, put: Put, s6_case: Any, tmp_path: Path, s6_left_out: str
) -> None:
    """Each way of `S6Case.LEFT_OUT` leaves the disciplined side with no agreement counted in SQL as in the twin —
    those on the sums too, where the twin's `< 0.0005` and the SQL's `> 0` are different code (review-225 R3)."""
    s6_case.leave_out(s6_left_out)
    metric, twin = _s6_in_sql(db, put, s6_case, tmp_path)
    assert metric is None and math.isnan(twin)


# ── the probes ───────────────────────────────────────────────────────────────


def test_a13_a_stop_after_the_third_rejection(db: psycopg.Connection, put: Put) -> None:
    """G2 S2.2: only A-13's reason counts — a rep's own rejections stop a line too (the rule), but are not the case."""
    reason = str(StoriesConfig.load().anomaly("A-13").parameters["reason_cs"])

    def prompts(customer: int, count: int, why: str = reason) -> None:
        for k in range(count):
            put(
                db,
                "rep_prompt",
                rep_prompt_id=next(_ids),
                customer_id=customer,
                kind="unsupplied_items",
                generated_on=date(2025, 1, 6) + timedelta(weeks=2 * k),
                payload='{"line": "NUDLE", "product": "NUDLE-KURECI", "skus": ["2931000000944"]}',
                outcome="rejected",
                outcome_reason=why,
            )

    with db.transaction(force_rollback=True):
        prompts(1, 5)  # rejected five times: no stop
        assert _one(db, "probes/a13.sql") == 0
        prompts(3, 3, why="podnět se pro prodejnu nehodí")  # three of the rep's own: the rule, not the case
        assert _one(db, "probes/a13.sql") == 0
        prompts(2, 3)  # rejected three times, then none
        assert _one(db, "probes/a13.sql") == 1


def test_a14_finds_the_planted_signature_not_any_hold(db: psycopg.Connection, put: Put) -> None:
    def order(customer: int, day: date, reason: str | None = "credit_limit", status: str = "released") -> None:
        put(
            db,
            "sales_order",
            sales_order_id=next(_ids),
            customer_id=customer,
            ordered_on=day,
            hold_reason=reason,
            status=status,
        )

    with db.transaction(force_rollback=True):
        order(1, date(2026, 6, 10))  # one held order: the policy, not the case
        assert _one(db, "probes/a14.sql") == 0
        order(2, date(2026, 6, 3))
        order(2, date(2026, 6, 17))  # two held, both released, none before: the case
        assert _one(db, "probes/a14.sql") == 1
        order(3, date(2026, 6, 3))
        order(3, date(2026, 6, 17))
        order(3, date(2025, 9, 8), reason="overdue")  # held before: a customer with a history of holds
        assert _one(db, "probes/a14.sql") == 1


def test_a6_is_the_twin_s_count_on_the_hand_built_arc(
    db: psycopg.Connection, put: Put, tmp_path: Path, a6_rows: Any
) -> None:
    """G2 S2.3 T7: a6.sql on conftest's `a6_case` gives the twin's count — the sightings with the arc whole, none with a
    leg missing (the facings too flat, the dip too shallow, no sighting); on a build that ends inside the weeks the
    planted dip and no other, on one that ends before the dip's `from` none (review-248 R9)."""
    a6_case, write_rows = a6_rows
    cut = date(2025, 11, 30)  # review-248 R9: a build that ends inside the weeks reads its days, as the twin
    arcs: list[tuple[int, float, float, int, date | None]] = [
        (5, 1.5, 0.88, 5, None),
        (5, 1.2, 0.88, 0, None),
        (5, 1.5, 0.96, 0, None),
        (0, 1.5, 0.88, 0, None),
        (3, 1.5, 0.88, 3, cut),
        (3, 1.5, 1.0, 0, cut),
        (1, 1.5, 0.88, 0, date(2025, 11, 7)),
    ]
    for k, (seen, lift, dip, found, ends) in enumerate(arcs):
        rows = a6_case(seen, lift, dip, cut=ends)
        write_rows(tmp_path / str(k), rows)
        twin = a6(Build(tmp_path / str(k)), StoriesConfig.load())
        with db.transaction(force_rollback=True):
            for table, held in rows.items():
                for row in held:
                    put(db, table, **row)
            db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
            metric = _one(db, "probes/a6.sql")
        assert (metric, twin) == (found, found)


def test_a12_sees_more_buyers_not_only_bigger_baskets(db: psycopg.Connection, put: Put) -> None:
    """Liberec sells one Premium soup in 5 of its 10 receipts, the other shop in 2 of 10: 2.5× more buyers, one unit
    each. Per buying basket both were 1."""
    with db.transaction(force_rollback=True):
        _catalogue(db, put, code="PREMIUM")
        for site, code, buyers in ((1, "SH-LIBEREC", 5), (2, "SH-JINDE", 2)):
            put(db, "site", site_id=site, code=code, kind="shop")
            for k in range(10):
                receipt = next(_ids)
                put(db, "shop_receipt", shop_receipt_id=receipt, site_id=site)
                if k < buyers:
                    put(db, "shop_receipt_line", shop_receipt_line_id=receipt, shop_receipt_id=receipt, sku_id=1, qty=1)
        assert _one(db, "probes/a12.sql") == 1
