"""The pricing engine (`01` §5, contracts §4, `05` §3 step 4): what an order line costs and why.

**The price lists** (`01` §5.1). The standard list is a quarterly version: each SKU sold in the quarter at the
catalogue's `base_price_case`, per piece ÷ `units_per_case`, with the factories' standard price (`cogs_case`) as its
purchase price. A channel's list is a monthly version: the standard × `world.yaml pricing.channel_factor[channel]`,
or × the factor of Marketing's hero override for that SKU in that channel. Both carry their validity; the `World`
holds the versions in force (`World.standard_list`, `World.channel_lists`), which the period ends roll (S1.4 the
channel lists monthly, S4.3 the standard list quarterly with its price drift).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

from hartland_sim.config import WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import Row
from hartland_sim.setup.portfolio import Sku
from hartland_sim.setup.rows import row
from hartland_sim.world import Price, PriceList

STANDARD_CODE = "SC-{year}-Q{quarter}"  # `standard_price_list.code`: SC = standardní ceník


def quarter_start(day: date) -> date:
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


def month_end(day: date) -> date:
    following = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    return following - timedelta(days=1)


def quarter_end(day: date) -> date:
    first = quarter_start(day)
    return month_end(date(first.year, first.month + 2, 1))


def _sold(sku: Sku, first: date, last: date) -> bool:
    return sku.launched_on <= last and sku.last_day >= first


def standard_list(skus: Sequence[Sku], day: date) -> PriceList:
    """The standard list of the quarter holding `day`: each SKU sold in the quarter at its base price."""
    first, last = quarter_start(day), quarter_end(day)
    prices = {
        sku.sku_id: Price(sku.base_price_case, round(sku.base_price_case / sku.units_per_case, 2), sku.cogs_case)
        for sku in skus
        if _sold(sku, first, last)
    }
    code = STANDARD_CODE.format(year=first.year, quarter=(first.month - 1) // 3 + 1)
    return PriceList(None, code, first, last, 1.0, prices)


def channel_list(standard: PriceList, skus: Sequence[Sku], world: WorldConfig, channel: str, day: date) -> PriceList:
    """`channel`'s list of the month holding `day`: the standard's SKUs sold in the month × the channel's factor, a
    hero override's factor for its SKU."""
    first, last = day.replace(day=1), month_end(day)
    factor = {str(name): value for name, value in world.pricing.channel_factor.items()}[channel]
    heroes = {hero.sku: hero.factor for hero in world.pricing.hero_overrides if hero.channel == channel}
    prices = {}
    for sku in skus:
        standard_price = standard.prices.get(sku.sku_id)
        if standard_price is None or not _sold(sku, first, last):
            continue
        own = heroes.get(sku.code, factor)
        case = round(standard_price.case * own, 2)
        prices[sku.sku_id] = Price(case, round(case / sku.units_per_case, 2), is_override=sku.code in heroes)
    return PriceList(channel, None, first, last, factor, prices)


def standard_rows(standard: PriceList, issuer: int) -> tuple[Row, list[Row]]:
    """The `standard_price_list` row of `standard` and its `standard_price` rows; `issuer` is Marketing's director."""
    list_id = row_id("standard_price_list", code=standard.code)
    head = row(
        "standard_price_list",
        standard_price_list_id=list_id,
        code=standard.code,
        valid_from=standard.valid_from,
        valid_to=standard.valid_to,
        issued_by_employee_id=issuer,
        document_id=None,
    )
    items = [
        row(
            "standard_price",
            standard_price_list_id=list_id,
            sku_id=sku_id,
            price_case=price.case,
            price_piece=price.piece,
            purchase_price_case=price.purchase_case,
        )
        for sku_id, price in standard.prices.items()
    ]
    return head, items


def channel_rows(channel_list_: PriceList) -> tuple[Row, list[Row]]:
    """The `channel_price_list` row of a channel's list and its `channel_price` rows."""
    list_id = row_id("channel_price_list", channel=channel_list_.channel, valid_from=channel_list_.valid_from)
    head = row(
        "channel_price_list",
        channel_price_list_id=list_id,
        channel=channel_list_.channel,
        valid_from=channel_list_.valid_from,
        valid_to=channel_list_.valid_to,
        factor=channel_list_.factor,
        document_id=None,
    )
    items = [
        row(
            "channel_price",
            channel_price_list_id=list_id,
            sku_id=sku_id,
            price_case=price.case,
            price_piece=price.piece,
            is_override=price.is_override,
        )
        for sku_id, price in channel_list_.prices.items()
    ]
    return head, items
