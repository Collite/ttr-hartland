"""The order of application and the sums (G1 S1.3 T6, contracts §4): a golden line through every step, a KAM's line,
the free lines, the list in force — and 10 000 generated lines on which the steps apply in the contracts' order each
on the price the one before left, Σ `amount_case` × cases = `discount_total` to the haléř, promotions never touch the
sales budget and no order holds two promotions that exclude each other."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.loop.demand import BasketReady
from hartland_sim.loop.pricing import (
    Draft,
    Priced,
    Rules,
    Term,
    admits,
    channel_list,
    condition,
    halere,
    price_line,
    price_order,
    standard_list,
)
from hartland_sim.setup.run import Setup
from hartland_sim.world import Agreement, Checkpoint, Customer, PriceList, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
ORDER = ("volume", "cross_sell", "agreed", "individual", "promotion", "free_goods")
MARKETING_KINDS = ("promotion", "free_goods")
GOLD_TV = row_id("promotion", code="PR-2025-03-GOLD-TV")


def _lists(opening: Setup, day: date) -> dict[str, PriceList]:
    skus = opening.portfolio.skus
    standard = standard_list(skus, day)
    return {channel: channel_list(standard, skus, WORLD, channel, day) for channel in opening.world.channel_lists}


@pytest.fixture(scope="module")
def rules(opening: Setup) -> Rules:
    return Rules.of(opening.world, WORLD, STORIES)


# ── goldens ──────────────────────────────────────────────────────────────────


def test_a_retail_line_through_every_step() -> None:
    terms = [
        Term("volume", "VOL-T2", "sales", pct=4),
        Term("cross_sell", "XS-T2", "sales", pct=3),
        Term("agreed", "AG-2025-00001", "sales", pct=5, price_agreement_id=11),
    ]
    promotion = Term("promotion", "PR-2025-03-GOLD-TV", "marketing", pct=20, promotion_id=GOLD_TV)
    priced = price_line(612.0, Draft(1, 3, 18, promotions=(promotion,)), terms)
    rows = [(d.seq, d.kind, d.amount_case, d.amount_line, d.budget) for d in priced.discounts]
    assert rows == [
        (1, "volume", 24.48, 73.44, "sales"),  # 4 % of 612.00
        (2, "cross_sell", 17.63, 52.89, "sales"),  # 3 % of 587.52
        (3, "agreed", 28.49, 85.47, "sales"),  # 5 % of 569.89
        (4, "promotion", 108.28, 324.84, "marketing"),  # 20 % of 541.40
    ]
    assert (priced.list_price_case, priced.discount_total_case, priced.invoice_price_case) == (612.0, 178.88, 433.12)
    assert (priced.discount_total, priced.cases, priced.is_free_goods) == (536.64, 3, False)


def test_a_kam_s_line_takes_its_individual_price_then_the_promotion() -> None:
    individual = Term("individual", "AG-2025-00002", "sales", price_case=480.0, price_agreement_id=12)
    promotion = Term("promotion", "PR-2025-03-GOLD-TV", "marketing", pct=20, promotion_id=GOLD_TV)
    priced = price_line(526.32, Draft(1, 10, 60, promotions=(promotion,)), [individual])
    assert [(d.kind, d.amount_case) for d in priced.discounts] == [("individual", 46.32), ("promotion", 96.0)]
    assert (priced.invoice_price_case, priced.discount_total) == (384.0, 1423.2)


def test_a_free_line_is_its_list_price_taken_off_whole() -> None:
    gift = Term("free_goods", "PR-2025-09-GOLD-11", "marketing", pct=100, promotion_id=7)
    tier = Term("volume", "VOL-T2", "sales", pct=4)
    priced = price_line(612.0, Draft(1, 2, 12, True, (gift,)), [tier])
    assert [(d.kind, d.amount_case, d.budget) for d in priced.discounts] == [("free_goods", 612.0, "marketing")]
    assert (priced.invoice_price_case, priced.discount_total, priced.is_free_goods) == (0.0, 1224.0, True)


def test_an_order_is_priced_on_the_list_in_force(opening: Setup, rules: Rules) -> None:
    world = Checkpoint.of(opening.world).restore()
    shop = next(c for c in world.customers.values() if c.channel == "small_retail" and c.status == "active")
    assert world.demand is not None
    point = next(s.delivery_point_id for s in world.demand.spots if s.customer_id == shop.customer_id)
    gold = next(sku for sku in opening.portfolio.skus if sku.line == "GOLD" and sku.launched_on < date(2023, 1, 1))
    basket = BasketReady(date(2025, 9, 15), shop.customer_id, point, "small_retail", ((gold.sku_id, 5.0, 5.0 * 6),))
    with pytest.raises(ValueError, match="roll it"):
        price_order(world, rules, basket)
    world.channel_lists = _lists(opening, basket.day)
    paid, free = price_order(world, rules, basket)
    listed = world.channel_lists["small_retail"].prices[gold.sku_id].case
    assert (paid.cases, paid.list_price_case, paid.discounts) == (3, listed, ())  # GOLD-11's 1 + 1
    assert (free.cases, free.is_free_goods, free.invoice_price_case) == (2, True, 0.0)
    assert free.discounts[0].code == "PR-2025-09-GOLD-11"


# ── 10 000 generated lines ───────────────────────────────────────────────────


def _generated(opening: Setup, rules: Rules) -> list[tuple[BasketReady, list[Priced], World]]:
    """Orders through 2025 until they hold 10 000 lines: a random channel, a random active customer of it at one of
    its points, random SKUs of the month's list (in cases, or pieces at the web and the own shops), the running
    promotions' condition and reward SKUs mixed in; small retail with the month's random conditions; half the
    customers with agreed discounts or individual prices, some over the line's maximum or above the list."""
    world = Checkpoint.of(opening.world).restore()
    draw = np.random.default_rng(20251102)
    customers = [c for c in world.customers.values() if c.status == "active"]
    points: dict[int, list[int]] = {}
    assert world.demand is not None
    for spot in world.demand.spots:
        points.setdefault(spot.customer_id, []).append(spot.delivery_point_id)
    lists = {month: _lists(opening, date(2025, month, 1)) for month in range(1, 13)}
    year = (date(2025, 1, 1), date(2025, 12, 31))
    employee = next(iter(world.actors))
    channels: dict[str, list[Customer]] = {}
    for customer in customers:
        channels.setdefault(customer.channel, []).append(customer)
    for n, customer in enumerate(customers[::2]):
        kind = "agreed_discount" if customer.channel == "small_retail" else "individual_price"
        for line in draw.choice(list(rules.lines), size=3, replace=False):
            code = f"AG-2025-{n:03d}{line}"
            pct = float(round(draw.uniform(2, 14), 1))
            world.add_agreement(
                row_id("price_agreement", code=code),
                Agreement(customer.customer_id, str(line), pct, *year, code, kind, employee),
            )
        if kind == "individual_price":
            sku = int(draw.choice(list(lists[1]["big_retail"].prices)))
            code = f"AG-2025-{n:03d}SKU"
            price = float(round(lists[1]["big_retail"].prices[sku].case * draw.uniform(0.8, 1.05), 2))
            world.add_agreement(
                row_id("price_agreement", code=code),
                Agreement(customer.customer_id, rules.sku_line[sku], 5.0, *year, code, kind, employee, sku, price),
            )
    out: list[tuple[BasketReady, list[Priced], World]] = []
    lines = 0
    while lines < 10_000:
        day = date(2025, 1, 1) + timedelta(days=int(draw.integers(0, 365)))
        among = channels[str(draw.choice(list(channels)))]
        customer = among[int(draw.integers(len(among)))]
        point = int(draw.choice(points[customer.customer_id]))
        world.channel_lists = lists[day.month]
        prices = world.channel_lists[customer.channel].prices
        if customer.channel == "small_retail":
            customer.condition = condition(
                WORLD.pricing.tiers,
                day.replace(day=1),
                float(draw.uniform(0, 150_000)),
                int(draw.integers(0, 9)),
                bool(draw.random() < 0.5),
            )
        skus = [int(sku) for sku in draw.choice(list(prices), size=int(draw.integers(1, 7)), replace=False)]
        if draw.random() < 0.4:
            for promotion in world.promotions.values():
                if promotion.valid_from <= day <= promotion.valid_to:
                    for named in (promotion.condition_sku, promotion.reward_sku):
                        if named and rules.sku_ids[named] in prices and rules.sku_ids[named] not in skus:
                            skus.append(rules.sku_ids[named])
        grain = bool(world.demand.pieces[rules.points[point]])
        basket_lines = []
        for sku in skus:
            if grain:
                pieces = float(draw.integers(1, 25))
                basket_lines.append((sku, pieces / rules.units[sku], pieces))
            else:
                cases = float(draw.integers(1, 13))
                basket_lines.append((sku, cases, cases * rules.units[sku]))
        basket = BasketReady(day, customer.customer_id, point, customer.channel, tuple(basket_lines))
        priced = price_order(world, rules, basket)
        out.append((basket, priced, world))
        lines += len(priced)
    return out


@pytest.fixture(scope="module")
def generated(opening: Setup, rules: Rules) -> list[tuple[BasketReady, list[Priced], World]]:
    return _generated(opening, rules)


def test_on_10_000_lines_the_rows_sum_to_the_line_s_total(
    generated: list[tuple[BasketReady, list[Priced], World]],
) -> None:
    count = 0
    for _, priced, _ in generated:
        for line in priced:
            per_case = sum((Decimal(str(d.amount_case)) for d in line.discounts), Decimal(0))
            total = (per_case * Decimal(str(line.cases))).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
            assert total == Decimal(str(line.discount_total)), line
            assert Decimal(str(line.discount_total_case)) == per_case
            assert Decimal(str(line.invoice_price_case)) == Decimal(str(line.list_price_case)) - per_case >= 0
            count += 1
    assert count >= 10_000


def test_on_10_000_lines_the_steps_apply_in_the_contracts_order_each_on_the_price_before(
    generated: list[tuple[BasketReady, list[Priced], World]],
) -> None:
    kinds: Counter[str] = Counter()
    for _, priced, _ in generated:
        for line in priced:
            assert [d.seq for d in line.discounts] == list(range(1, len(line.discounts) + 1))
            ranks = [ORDER.index(d.kind) for d in line.discounts]
            assert ranks == sorted(ranks), line
            price = halere(line.list_price_case)
            for d in line.discounts:
                amount = halere(d.amount_case)
                if d.pct is not None:
                    assert amount == round(price * d.pct / 100), (line, d)
                else:
                    assert d.kind == "individual" and 0 < amount <= price
                price -= amount
                kinds[d.kind] += 1
            if line.is_free_goods:
                assert [(d.kind, d.pct) for d in line.discounts] == [("free_goods", 100)]
    assert all(kinds[kind] >= 50 for kind in ORDER), kinds


def test_on_10_000_lines_promotions_never_touch_the_sales_budget(
    generated: list[tuple[BasketReady, list[Priced], World]],
) -> None:
    for _, priced, _ in generated:
        for line in priced:
            for d in line.discounts:
                assert d.budget == ("marketing" if d.kind in MARKETING_KINDS else "sales")
                assert (d.promotion_id is not None) == (d.kind in MARKETING_KINDS)
                assert (d.price_agreement_id is not None) == (d.kind in ("agreed", "individual"))


def test_on_10_000_lines_each_promotion_runs_admits_the_point_and_excludes_none_other_on_the_order(
    generated: list[tuple[BasketReady, list[Priced], World]], rules: Rules
) -> None:
    for basket, priced, world in generated:
        customer = world.customers[basket.customer_id]
        assert world.demand is not None
        spot = world.demand.spots[rules.points[basket.delivery_point_id]]
        applied = {d.promotion_id for line in priced for d in line.discounts if d.promotion_id is not None}
        held = [world.promotions[promotion_id] for promotion_id in applied]
        for promotion in held:
            assert admits(promotion, spot, customer, basket.day)
            assert not {other.code for other in held} & set(promotion.excludes), promotion.code
