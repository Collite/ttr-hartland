"""The tier steps (G1 S1.3 T2, `01` §5.2, ⚑NH-7): the conditions a small-retail customer's previous month sets, read
from `world.yaml pricing.tiers`; the volume and cross-sell terms on every line of the month, applied each on the
price the one before left."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal

import pytest

from hartland_sim.config import CrossSellTier, VolumeTier, WorldConfig
from hartland_sim.loop.pricing import (
    Discount,
    Term,
    apply,
    condition,
    condition_row,
    discount_row,
    tiers,
)
from hartland_sim.setup.run import Setup
from hartland_sim.world import Condition, Customer

TIERS = WorldConfig.load().pricing.tiers
MAY = date(2025, 5, 1)


def _retail(opening: Setup, held: Condition | None = None) -> Customer:
    first = next(c for c in opening.world.customers.values() if c.channel == "small_retail" and c.status == "active")
    return replace(first, condition=held)


def _total(discounts: tuple[Discount, ...], cases: float) -> Decimal:
    per_case = sum((Decimal(str(d.amount_case)) for d in discounts), Decimal(0))
    return (per_case * Decimal(str(cases))).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)


def test_51_000_crowns_and_five_lines_of_both_brands_give_4_and_3_percent() -> None:
    held = condition(TIERS, MAY, 51_000, 5, True)
    assert held == Condition(MAY, "VOL-T2", 4, "XS-T2", 3, 51_000, 5, True)


@pytest.mark.parametrize("tier", TIERS.volume, ids=lambda tier: tier.code)
def test_a_volume_tier_starts_at_its_threshold(tier: VolumeTier) -> None:
    below = [t for t in TIERS.volume if t.min_net_revenue < tier.min_net_revenue]
    assert condition(TIERS, MAY, tier.min_net_revenue, 0, False).volume_tier == tier.code
    under = condition(TIERS, MAY, tier.min_net_revenue - 0.01, 0, False)
    assert under.volume_tier == (below[-1].code if below else None)
    assert under.volume_pct == (below[-1].pct if below else 0)


@pytest.mark.parametrize("tier", TIERS.cross_sell, ids=lambda tier: tier.code)
def test_a_cross_sell_tier_starts_at_its_lines_and_both_brands_add_their_bonus(tier: CrossSellTier) -> None:
    alone = condition(TIERS, MAY, 0, tier.min_lines, False)
    both = condition(TIERS, MAY, 0, tier.min_lines, True)
    assert (alone.cross_sell_tier, alone.cross_sell_pct) == (tier.code, tier.pct)
    bonus = min(tier.pct + TIERS.cross_sell_both_brands_pct, TIERS.cross_sell_max_pct)
    assert (both.cross_sell_tier, both.cross_sell_pct) == (tier.code, bonus)


def test_both_brands_without_a_tier_give_nothing_and_the_bonus_is_capped() -> None:
    lowest = min(tier.min_lines for tier in TIERS.cross_sell)
    short = condition(TIERS, MAY, 0, lowest - 1, True)
    assert (short.cross_sell_tier, short.cross_sell_pct) == (None, 0)
    capped = replace(TIERS, cross_sell_max_pct=2.5)
    assert condition(capped, MAY, 0, 5, True).cross_sell_pct == 2.5


def test_the_thresholds_are_world_yaml_s() -> None:
    doubled = replace(TIERS, volume=tuple(replace(t, min_net_revenue=2 * t.min_net_revenue) for t in TIERS.volume))
    assert condition(doubled, MAY, 51_000, 0, False).volume_tier == TIERS.volume[0].code
    fewer = replace(TIERS, cross_sell=tuple(replace(t, min_lines=t.min_lines - 2) for t in TIERS.cross_sell))
    assert condition(fewer, MAY, 0, 3, False).cross_sell_tier == TIERS.cross_sell[-1].code


def test_the_month_s_condition_is_a_volume_and_a_cross_sell_term(opening: Setup) -> None:
    customer = _retail(opening, condition(TIERS, MAY, 51_000, 5, True))
    assert tiers(customer, MAY) == (
        Term("volume", "VOL-T2", "sales", pct=4),
        Term("cross_sell", "XS-T2", "sales", pct=3),
    )


def test_no_tiers_without_the_month_s_condition_or_outside_small_retail(opening: Setup) -> None:
    assert tiers(_retail(opening), MAY) == ()
    assert tiers(_retail(opening, condition(TIERS, date(2025, 4, 1), 51_000, 5, True)), MAY) == ()
    assert tiers(_retail(opening, condition(TIERS, MAY, 10_000, 1, False)), MAY) == ()
    chain = next(c for c in opening.world.customers.values() if c.channel == "big_retail")
    assert tiers(replace(chain, condition=condition(TIERS, MAY, 9e6, 11, True)), MAY) == ()


@pytest.mark.parametrize(("list_price", "cases"), [(612.0, 3), (1234.56, 1), (99.99, 7), (450.1, 0.167)])
def test_every_line_gets_both_rows_each_on_the_price_the_one_before_left(
    opening: Setup, list_price: float, cases: float
) -> None:
    terms = tiers(_retail(opening, condition(TIERS, MAY, 51_000, 5, True)), MAY)
    volume, cross = apply(list_price, cases, terms)
    assert (volume.seq, volume.kind, volume.code, volume.budget, volume.pct) == (1, "volume", "VOL-T2", "sales", 4)
    assert (cross.seq, cross.kind, cross.code, cross.budget, cross.pct) == (2, "cross_sell", "XS-T2", "sales", 3)
    assert volume.amount_case == round(round(list_price * 100) * 4 / 100) / 100
    after = round(list_price * 100) - round(volume.amount_case * 100)
    assert cross.amount_case == round(after * 3 / 100) / 100
    assert volume.promotion_id is cross.promotion_id is None
    rows = Decimal(str(volume.amount_line)) + Decimal(str(cross.amount_line))
    slack = Decimal(0) if cases == int(cases) else Decimal("0.01")  # each row's amount_line rounds on its own
    assert abs(rows - _total((volume, cross), cases)) <= slack


def test_an_individual_price_takes_the_rest_down_to_it_and_none_above() -> None:
    (row,) = apply(500.0, 2, [Term("individual", "AG-1", "sales", price_case=430.0)])
    assert (row.amount_case, row.amount_line, row.pct) == (70.0, 140.0, None)
    assert apply(500.0, 2, [Term("individual", "AG-1", "sales", price_case=520.0)]) == ()
    with pytest.raises(ValueError, match="pct or a price"):
        apply(500.0, 2, [Term("agreed", "AG-2", "sales")])


def test_the_rows_render_with_the_inventory_s_columns(opening: Setup) -> None:
    held = condition(TIERS, MAY, 51_000, 5, True)
    customer = _retail(opening, held)
    assert condition_row(customer.customer_id, held) == {
        "customer_id": customer.customer_id,
        "month": MAY,
        "prev_month_net_revenue": 51_000,
        "volume_tier": "VOL-T2",
        "volume_pct": 4,
        "prev_month_lines_bought": 5,
        "both_brands": True,
        "cross_sell_tier": "XS-T2",
        "cross_sell_pct": 3,
    }
    (volume, _) = apply(612.0, 3, tiers(customer, MAY))
    assert discount_row(77, volume) == {
        "sales_order_line_id": 77,
        "seq": 1,
        "kind": "volume",
        "code": "VOL-T2",
        "promotion_id": None,
        "price_agreement_id": None,
        "pct": 4,
        "amount_case": 24.48,
        "amount_line": 73.44,
        "budget": "sales",
    }
