"""The pricing engine (`01` §5, contracts §4, `05` §3 step 4): what an order line costs and why.

**The price lists** (`01` §5.1). The standard list is a quarterly version: each SKU sold in the quarter at the
catalogue's `base_price_case`, per piece ÷ `units_per_case`, with the factories' standard price (`cogs_case`) as its
purchase price. A channel's list is a monthly version: the standard × `world.yaml pricing.channel_factor[channel]`,
or × the factor of Marketing's hero override for that SKU in that channel. Both carry their validity; the `World`
holds the versions in force (`World.standard_list`, `World.channel_lists`), which the period ends roll (S1.4 the
channel lists monthly, S4.3 the standard list quarterly with its price drift).

**The discounts** (contracts §4). A line qualifies for `Term`s — a tier, an agreement, a promotion — and `apply` takes
them in the pricing order, each on the price the earlier ones left: a percentage off it, or an individual price down
to it. Each applied term is a `Discount`, a `sales_order_line_discount` row without its line id: `seq` is its place
in the order, `amount_case` what it took off a case. Money is counted in whole haléře, so the invariant holds exactly:
Σ `amount_case` × `cases` = the line's `discount_total`, rounded to the haléř half to even; the cases are the line's
`qty_cases`, to three decimals.

**The tiers** (`01` §5.2, ⚑NH-7): a small-retail customer's previous month sets its conditions for the month
(`condition`, S1.4's month-end): the volume tier its net revenue reached (`world.yaml pricing.tiers.volume`), the
cross-sell tier its product lines reached (`tiers.cross_sell`) with `cross_sell_both_brands_pct` more when it bought
both brands, up to `cross_sell_max_pct`. The bonus rides on a reached tier: two lines of two brands are no breadth.
`tiers` reads the month's `Condition` off the customer: a volume and a cross-sell term on every line, from the sales
budget, coded with the tier's code (the cross-sell tier's code with the bonus in its percentage).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from hartland_sim.config import Tiers, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import Row
from hartland_sim.setup.portfolio import Sku
from hartland_sim.setup.rows import row
from hartland_sim.world import Condition, Customer, Price, PriceList

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


# ── the discounts (T2–T6) ────────────────────────────────────────────────────

SALES, MARKETING = "sales", "marketing"  # `budget`: whose budget bears a discount
TIERED = ("small_retail",)  # the channel whose customers have monthly conditions (`customer_condition`)


@dataclass(frozen=True)
class Term:
    """A discount a line qualifies for, before its amount: `pct` off the price it meets, or the individual
    `price_case` it sets."""

    kind: str  # volume · cross_sell · agreed · individual · promotion · free_goods
    code: str
    budget: str
    pct: float | None = None
    price_case: float | None = None
    promotion_id: int | None = None
    price_agreement_id: int | None = None


@dataclass(frozen=True)
class Discount:
    """An applied term: a `sales_order_line_discount` row without its line id (`discount_row`)."""

    seq: int
    kind: str
    code: str
    promotion_id: int | None
    price_agreement_id: int | None
    pct: float | None
    amount_case: float
    amount_line: float
    budget: str


def halere(czk: float) -> int:
    return round(czk * 100)


def milli(cases: float) -> int:
    """`cases` in thousandths, as `qty_cases` (numeric(12,3)) holds them."""
    return round(cases * 1000)


def times(amount: int, cases: int) -> int:
    """`amount` haléře a case × `cases` thousandths, to the haléř, half to even."""
    whole, rest = divmod(amount * cases, 1000)
    return whole + (rest > 500 or (rest == 500 and whole % 2 == 1))


def apply(list_price_case: float, cases: float, terms: Sequence[Term]) -> tuple[Discount, ...]:
    """`terms` in order, each on the price the earlier ones left (module docstring). A term that takes nothing (an
    individual price at or above the price already reached) is no row."""
    price = halere(list_price_case)
    count = milli(cases)
    out: list[Discount] = []
    for term in terms:
        if term.price_case is not None:
            amount = max(0, price - halere(term.price_case))
        elif term.pct is not None:
            amount = round(price * term.pct / 100)
        else:
            raise ValueError(f"{term.code}: a term needs a pct or a price")
        if amount <= 0:
            continue
        price -= amount
        out.append(
            Discount(
                len(out) + 1,
                term.kind,
                term.code,
                term.promotion_id,
                term.price_agreement_id,
                term.pct,
                amount / 100,
                times(amount, count) / 100,
                term.budget,
            )
        )
    return tuple(out)


def condition(tiers: Tiers, month: date, net_revenue: float, lines_bought: int, both_brands: bool) -> Condition:
    """A small-retail customer's conditions for `month` from its previous month (module docstring)."""
    volume = [tier for tier in tiers.volume if net_revenue >= tier.min_net_revenue]
    cross = [tier for tier in tiers.cross_sell if lines_bought >= tier.min_lines]
    top_volume = max(volume, key=lambda tier: tier.min_net_revenue) if volume else None
    top_cross = max(cross, key=lambda tier: tier.min_lines) if cross else None
    cross_pct = 0.0
    if top_cross is not None:
        bonus = tiers.cross_sell_both_brands_pct if both_brands else 0.0
        cross_pct = min(top_cross.pct + bonus, tiers.cross_sell_max_pct)
    return Condition(
        month,
        top_volume.code if top_volume else None,
        top_volume.pct if top_volume else 0.0,
        top_cross.code if top_cross else None,
        cross_pct,
        net_revenue,
        lines_bought,
        both_brands,
    )


def condition_row(customer_id: int, held: Condition) -> Row:
    """The `customer_condition` row of a customer's month."""
    return row(
        "customer_condition",
        customer_id=customer_id,
        month=held.month,
        prev_month_net_revenue=held.net_revenue,
        volume_tier=held.volume_tier,
        volume_pct=held.volume_pct,
        prev_month_lines_bought=held.lines_bought,
        both_brands=held.both_brands,
        cross_sell_tier=held.cross_sell_tier,
        cross_sell_pct=held.cross_sell_pct,
    )


def tiers(customer: Customer, month: date) -> tuple[Term, ...]:
    """The tier terms of `customer`'s lines in `month` (its first day): its `Condition` of that month, none without
    one (a customer gained this month) or outside small retail."""
    held = customer.condition
    if customer.channel not in TIERED or held is None or held.month != month:
        return ()
    out = []
    if held.volume_tier and held.volume_pct > 0:
        out.append(Term("volume", held.volume_tier, SALES, pct=held.volume_pct))
    if held.cross_sell_tier and held.cross_sell_pct > 0:
        out.append(Term("cross_sell", held.cross_sell_tier, SALES, pct=held.cross_sell_pct))
    return tuple(out)


def discount_row(line_id: int, discount: Discount) -> Row:
    """The `sales_order_line_discount` row of `discount` on the line `line_id`."""
    return row(
        "sales_order_line_discount",
        sales_order_line_id=line_id,
        seq=discount.seq,
        kind=discount.kind,
        code=discount.code,
        promotion_id=discount.promotion_id,
        price_agreement_id=discount.price_agreement_id,
        pct=discount.pct,
        amount_case=discount.amount_case,
        amount_line=discount.amount_line,
        budget=discount.budget,
    )
