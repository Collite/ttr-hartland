"""G0 S0.4 T3, the component tier: the detectors' and probes' definitions on hand-built rows. Each case writes only the
rows its SQL reads (conftest's `put`), inside a transaction that rolls back; the foreign keys are deferred, so the
rest is never missed. G5 S5.2 tunes the stories' parameters on the build; these cases pin what the SQL means."""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import psycopg
import pytest

from hartland_sim.config import STORIES_DIR

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


# ── S5: the gain of a display ────────────────────────────────────────────────


def _pos(db: psycopg.Connection, put: Put) -> None:
    _catalogue(db, put)
    put(db, "okres", okres_id=1)
    put(db, "region", region_id=1, code="PHA")
    put(db, "pos_asset", pos_asset_id=1, region_id=1, cost=36_000)


def test_s5_a_display_that_wins_a_new_store_gains(db: psycopg.Connection, put: Put) -> None:
    """No orders in the 8 weeks before and ten during: the gain is all ten. A loan without any order keeps its cost."""
    with db.transaction(force_rollback=True):
        _pos(db, put)
        for customer in (1, 2):
            put(db, "store", store_id=customer)
            put(db, "customer", customer_id=customer, store_id=customer)
            put(
                db,
                "pos_loan",
                pos_loan_id=customer,
                pos_asset_id=1,
                customer_id=customer,
                store_id=customer,
                lent_on=date(2025, 3, 3),
                returned_on=date(2025, 6, 1),
            )
        for week in range(10):
            _sale(db, put, customer=1, store=1, day=date(2025, 3, 10) + timedelta(weeks=week), gross=1000)
        db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        margin = _cte(db, "detectors/s5.sql", "margin")
    assert float(margin[1]["gain"]) == pytest.approx(10_000)
    assert margin[1]["cost"] > 0
    assert (margin[2]["gain"], margin[2]["cost"] > 0) == (0, True)


def test_s5_a_display_at_a_chain_store_reads_that_store(db: psycopg.Connection, put: Put) -> None:
    """A loan at one store of a chain customer reads that store's sales, not the chain's."""
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
    assert float(margin[1]["gain"]) == pytest.approx(1000)


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
        giveaway = _cte(db, "detectors/s5.sql", "giveaway")
    assert float(giveaway["PHA"]["cost"]) == pytest.approx(200)


# ── S6: a difference, not a ratio ────────────────────────────────────────────


def test_s6_holds_when_the_undisciplined_gain_is_negative(db: psycopg.Connection, put: Put) -> None:
    """The disciplined rep gains 1.2 CZK of margin per discount crown, the undisciplined one loses 0.6: the metric is
    1.8 (the ratio of ratios was −2, a fail). The disciplined customer's order before the agreement carries its
    predecessor's discount, which is not this agreement's."""
    with db.transaction(force_rollback=True):
        _catalogue(db, put)
        put(db, "okres", okres_id=1)
        for rep, slot in ((1, "SJC-02"), (2, "SMS-02")):
            put(db, "area", area_id=rep, code=slot)
            put(db, "assignment", assignment_id=rep, employee_id=rep, kind="rep_area", area_id=rep, **TENURE)
            put(db, "customer", customer_id=rep)
            put(
                db,
                "price_agreement",
                price_agreement_id=rep,
                customer_id=rep,
                product_line_id=1,
                kind="agreed_discount",
                valid_from=date(2025, 1, 1),
                valid_to=date(2025, 3, 31),
                agreed_by_employee_id=rep,
            )
        # 90 days; the 8 weeks before give 5 600 ÷ 56 = 100 CZK a day
        _sale(db, put, customer=1, day=date(2024, 12, 1), gross=7600, agreed=2000)
        _sale(db, put, customer=1, day=date(2025, 2, 1), gross=31_000, agreed=10_000)  # (21 000 − 9 000) ÷ 10 000
        _sale(db, put, customer=2, day=date(2024, 12, 1), gross=5600)
        _sale(db, put, customer=2, day=date(2025, 2, 1), gross=11_000, agreed=5000)  # (6 000 − 9 000) ÷ 5 000
        _last_order_day(db, put, date(2026, 10, 31))
        db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        metric = _one(db, "detectors/s6.sql")
    assert float(metric) == pytest.approx(1.2 - (-0.6))


# ── the probes ───────────────────────────────────────────────────────────────


def test_a13_a_stop_after_the_third_rejection(db: psycopg.Connection, put: Put) -> None:
    def prompts(customer: int, count: int) -> None:
        for k in range(count):
            put(
                db,
                "rep_prompt",
                rep_prompt_id=next(_ids),
                customer_id=customer,
                kind="unsupplied_items",
                generated_on=date(2025, 1, 6) + timedelta(weeks=2 * k),
                payload='{"lines": ["NUDLE"]}',
                outcome="rejected",
            )

    with db.transaction(force_rollback=True):
        prompts(1, 5)  # rejected five times: no stop
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


def test_a6_counts_promotion_sightings_only(db: psycopg.Connection, put: Put) -> None:
    with db.transaction(force_rollback=True):
        put(db, "competitor", competitor_id=1, code="KAVALIR")
        put(db, "region", region_id=1, code="VCH")
        put(db, "area", area_id=1, region_id=1)
        put(db, "store", store_id=1, area_id=1)

        def sightings(count: int, kind: str) -> None:
            for k in range(count):
                visit = next(_ids)
                put(db, "visit", visit_id=visit, store_id=1, day=date(2025, 11, 3) + timedelta(days=3 * k))
                put(
                    db,
                    "competitor_sighting",
                    competitor_sighting_id=visit,
                    visit_id=visit,
                    competitor_id=1,
                    activity_kind=kind,
                )

        sightings(12, "price_cut")
        assert _one(db, "probes/a6.sql") == 0
        sightings(10, "promotion")
        assert _one(db, "probes/a6.sql") == 10


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
