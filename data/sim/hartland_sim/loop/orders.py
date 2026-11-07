"""The orders (`05` §3 step 4, `03` §6): the day's baskets as `sales_order`s with their lines priced, and the
customers' asks for a discount.

**An order** (`order`): each `BasketReady` is one `sales_order` of its customer's delivery point.
- `order_no` is its business key — the day, the customer's code and its number among the customer's orders of the day
  (`20240115-MO-00042-1`; a chain's stores order the same day) — so its id and its lines' (`order_no`, `line_no`)
  never follow the generation order.
- `source` is drawn by the channel's mix (`world.yaml channels.<channel>.order_sources`; a station shop takes special
  retail's), stream `order` by the order's id. A visit or a phone order is taken by who holds the customer that day —
  its area's rep, its KAM (`pricing.granter`) — the others by no one. When no one holds it that day (an area or an
  account unheld), the same draw, rescaled within the source it picked, picks again among the mix's other sources,
  their shares renormalised (review-221 N3, RR4): no visit nobody made. A basket a visit took (G2 S2.1,
  `loop.field`) is a `visit` order with its `visit_id`, taken by the visit's rep; no channel's mix holds a visit, so
  a `visit` order is always the field's, what the visits' conversion makes (HoReCa's KAMs make no visits yet).
- `requested_on` is the next working day (`setup.calendar`'s Czech days off); `dc_site_id` the DC of the point's
  region (`world.yaml sites.dcs`); `status` the credit check's — `released`, or `held` with its reason (A-14).
Its lines are `pricing.price_order`'s, numbered in that order (the free lines last), each with the purchase price of
the standard list in force (the factories' standard price), its SKU's VAT rate, an inbound cost of 0 until G3 S3.1
costs the inbound transports, and no warehouse, delivery or marketing cost yet (G3, G4 S4.3). Each applied discount
is a `sales_order_line_discount` row.

**The asks** (`asks`, S1.3's `grant`): once a quarter each customer a rep or a KAM answers for — small retail, a
station shop, a key account, a vending operator, a HoReCa venue — asks for a discount on one of the lines it buys, on
a day of the quarter and a line drawn from stream `pricing_agree` by its id (one ask a quarter, so about 2 500
agreements a quarter, `03`'s ~20 k over the history). Only a customer active that day asks. The day's asks come
before its orders, so an order of that day carries what was agreed; the demand lifts the line from the next day
(`demand.Terms.today` reads the agreements before the asks). An agreement granted on a written day is a
`price_agreement` row; one of the warm-up runs to its quarter's end and is never cited after 2024-01-01.

**One day** (`step`): the asks, then each basket's order in the events' order, written through `Output` (nothing
before `history.first_day`, though the warm-up's orders are priced and their agreements kept as on any day; what only
the rows need — the order ids, the source draws, the agreement rows — is made on a written day only). A
tiered customer's order is booked on its month (`book`, `World.bought`): its net revenue, its product lines — what
the month-end reads the next month's tiers and the phases from.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, get_args

import numpy as np

from hartland_sim.config import Channel, WorldConfig
from hartland_sim.dates import next_working_day, quarter_end, quarter_start
from hartland_sim.ids import SEPARATOR, business_id, row_id
from hartland_sim.io.reference import Row
from hartland_sim.loop.demand import BasketReady
from hartland_sim.loop.pricing import (
    KAM_CHANNELS,
    TIERED,
    Priced,
    Refusal,
    Rules,
    agreement_row,
    credit_check,
    discount_row,
    grant,
    granter,
    order_value,
    price_order,
    running,
)
from hartland_sim.loop.warmup import Output
from hartland_sim.rng import uniforms
from hartland_sim.setup.customers import terms
from hartland_sim.setup.rows import row
from hartland_sim.world import Agreement, Bought, Customer, Promotion, World

ORDER_NO = "{day:%Y%m%d}-{code}-{n}"
TAKEN = ("visit", "phone")  # the sources a person takes: the customer's rep or KAM
ASKING = (*TIERED, *KAM_CHANNELS)  # the channels whose customers a rep or a KAM answers (`pricing.granter`)
Array = np.ndarray[Any, Any]


@dataclass(frozen=True)
class Desk:
    """What the orders read beside the `World`, for a run (never pickled): the pricing rules, `world.yaml`, each
    region's DC, each channel's source mix, and the asks' plan of each quarter (made when first asked for)."""

    rules: Rules
    config: WorldConfig
    dcs: Mapping[str, int]  # region code → its DC's site_id
    sources: Mapping[str, tuple[tuple[str, ...], Array]]  # a terms channel → its sources, their cumulative shares
    untaken: Mapping[str, tuple[tuple[str, ...], Array]]  # the same without the taken ones, for an order with no taker
    asking: Array  # the indices (in the World's order) of the customers a rep or a KAM answers
    _plans: dict[date, tuple[dict[int, Array], Array]] = field(default_factory=dict, repr=False)

    @classmethod
    def of(cls, world: World, rules: Rules, config: WorldConfig) -> Desk:
        dcs = {region: row_id("site", code=dc) for dc, regions in config.sites.dcs.items() for region in regions}
        sources, untaken = {}, {}
        for name in get_args(Channel):
            mix = getattr(config.channels, name).order_sources
            shares = np.cumsum([float(share) for share in mix.values()])
            if not np.isclose(shares[-1], 1.0):
                raise ValueError(f"world.yaml channels.{name}.order_sources sum to {shares[-1]}, not 1")
            sources[name] = (tuple(mix), shares)
            other = {kind: float(share) for kind, share in mix.items() if kind not in TAKEN}
            if len(other) < len(mix):
                if not sum(other.values()):
                    raise ValueError(f"world.yaml channels.{name}.order_sources: an order no one takes has no source")
                untaken[name] = (tuple(other), np.cumsum(list(other.values())) / sum(other.values()))
        customers = list(world.customers.values())
        asking = np.array([c for c, customer in enumerate(customers) if customer.channel in ASKING], dtype=np.int64)
        return cls(rules, config, dcs, sources, untaken, asking)

    def plan(self, world: World, day: date) -> tuple[dict[int, Array], Array]:
        """The asks of `day`'s quarter: the asking customers by the day they ask (its ordinal), and each one's draw of
        the line (by its index in `asking`)."""
        first = quarter_start(day)
        found = self._plans.get(first)
        if found is None:
            demand = world.demand
            assert demand is not None
            ids = demand.customer_ids[self.asking]
            days = (quarter_end(day) - first).days + 1
            offset = np.floor(uniforms("pricing_agree", "ask", first, ids=ids) * days).astype(np.int64)
            by_day: dict[int, Array] = {}
            for k in np.unique(offset):
                by_day[first.toordinal() + int(k)] = np.flatnonzero(offset == k)
            found = self._plans[first] = (by_day, uniforms("pricing_agree", "ask_line", first, ids=ids))
        return found


@dataclass(frozen=True)
class Order:
    """An order as written: its `sales_order` row, its lines' rows and their discounts'."""

    head: Row | None  # None for an order of a day that writes nothing (the warm-up's)
    lines: list[Row]
    discounts: list[Row]
    priced: list[Priced]
    value: float  # `pricing.order_value`: net, without VAT
    status: str
    hold_reason: str | None


def order_no(code: str, day: date, n: int) -> str:
    return ORDER_NO.format(day=day, code=code, n=n)


def _pick(shares: Array, draw: float) -> int:
    return min(int(np.searchsorted(shares, draw, side="right")), len(shares) - 1)


def source(desk: Desk, channel: str, draw: float, taken: bool = True) -> str:
    """The source a uniform `draw` picks from `channel`'s mix (a terms channel: a station shop's is special retail);
    from its sources no one takes when `taken` is False (module docstring)."""
    names, shares = (desk.sources if taken else desk.untaken)[channel]
    return names[_pick(shares, draw)]


def rescaled(desk: Desk, channel: str, draw: float) -> float:
    """`draw` within the interval of `channel`'s mix it falls in, as a uniform of [0, 1) again: one draw, read twice."""
    _, shares = desk.sources[channel]
    k = _pick(shares, draw)
    low = float(shares[k - 1]) if k else 0.0
    return min(max((draw - low) / (float(shares[k]) - low), 0.0), float(np.nextafter(1.0, 0.0)))


def order(
    world: World,
    desk: Desk,
    basket: BasketReady,
    n: int,
    draw: float,
    write: bool = True,
    today: Sequence[tuple[int, Promotion]] | None = None,
) -> Order:
    """`basket` as the customer's `n`-th order of its day, its source picked by `draw` (module docstring); its rows
    when it is `write`n. `today` is `pricing.running`'s promotions of the day, when the caller has them."""
    rules, day = desk.rules, basket.day
    customer = world.customers[basket.customer_id]
    standard = world.standard_list
    if standard is None or not standard.valid_from <= day <= standard.valid_to:
        held = f"{standard.valid_from}–{standard.valid_to}" if standard else "none"
        raise ValueError(f"the standard list in force on {day} is not in the World (it holds {held}): roll it first")
    priced = price_order(world, rules, basket, today)
    value = order_value(priced)
    status, reason = credit_check(world, desk.config, customer, day, value)
    if not write:
        return Order(None, [], [], priced, value, status, reason)
    number = order_no(customer.code, day, n)
    order_id = business_id("sales_order", number)  # `row_id`'s: its id_key is its `order_no`
    visit_id: int | None = None
    taker: int | None
    if basket.visit is not None:  # a visit took the basket (G2 S2.1)
        kind, (visit_id, taker) = "visit", basket.visit
    else:
        kind = source(desk, terms(customer), draw)
        taker = granter(world, customer, day) if kind in TAKEN else None
        if kind in TAKEN and taker is None:
            kind = source(desk, terms(customer), rescaled(desk, terms(customer), draw), taken=False)
    demand = world.demand
    assert demand is not None
    spot = demand.spots[rules.points[basket.delivery_point_id]]
    head = row(
        "sales_order",
        sales_order_id=order_id,
        order_no=number,
        customer_id=customer.customer_id,
        delivery_point_id=basket.delivery_point_id,
        channel=basket.channel,
        ordered_on=day,
        requested_on=next_working_day(day),
        source=kind,
        taken_by_employee_id=taker,
        visit_id=visit_id,
        status=status,
        hold_reason=reason,
        dc_site_id=desk.dcs[spot.region],
    )
    lines, discounts = [], []
    for line_no, line in enumerate(priced, start=1):
        line_id = business_id("sales_order_line", f"{number}{SEPARATOR}{line_no}")  # `row_id`'s (order_no, line_no)
        lines.append(
            row(
                "sales_order_line",
                sales_order_line_id=line_id,
                sales_order_id=order_id,
                line_no=line_no,
                sku_id=line.sku_id,
                qty_cases=line.cases,
                qty_pieces=line.pieces,
                list_price_case=line.list_price_case,
                discount_total_case=line.discount_total_case,
                discount_total=line.discount_total,
                invoice_price_case=line.invoice_price_case,
                is_free_goods=line.is_free_goods,
                purchase_price_case=standard.prices[line.sku_id].purchase_case,
                inbound_cost_case=0.0,
                warehouse_cost_case=None,
                delivery_cost_case=None,
                marketing_cost_case=None,
                vat_rate=rules.vat[line.sku_id],
            )
        )
        discounts += [discount_row(line_id, discount) for discount in line.discounts]
    return Order(head, lines, discounts, priced, value, status, reason)


def book(world: World, rules: Rules, customer: Customer, day: date, made: Order) -> None:
    """`made` on its customer's month (module docstring): only a tiered customer's, the one with monthly
    conditions and phase rows."""
    if customer.channel not in TIERED:
        return
    held = world.bought.setdefault(day.replace(day=1), {}).setdefault(customer.customer_id, Bought())
    held.net_revenue = round(held.net_revenue + made.value, 2)
    held.orders += 1
    for line in made.priced:
        if not line.is_free_goods:
            held.lines |= 1 << rules.lines[rules.sku_line[line.sku_id]]


def asks(world: World, desk: Desk, day: date) -> list[Agreement | Refusal]:
    """The day's asks (module docstring), each answered by `pricing.grant`, in the World's order of the customers."""
    by_day, line_draws = desk.plan(world, day)
    today = by_day.get(day.toordinal())
    if today is None:
        return []
    demand = world.demand
    assert demand is not None
    customers = list(world.customers.values())
    out: list[Agreement | Refusal] = []
    for k in today:
        c = int(desk.asking[k])
        customer = customers[c]
        if not customer.active_on(day):
            continue
        bought = np.flatnonzero(demand.carried[c])
        if not len(bought):
            continue
        line = demand.lines[int(bought[min(int(line_draws[k] * len(bought)), len(bought) - 1)])].code
        out.append(grant(world, desk.rules, customer.customer_id, line, day))
    return out


@dataclass
class Day:
    """What a day's orders did: the orders and lines made, the asks' answers."""

    orders: int = 0
    lines: int = 0
    granted: int = 0
    refused: Counter[str] = field(default_factory=Counter)  # by `Refusal.reason`
    held: Counter[str] = field(default_factory=Counter)  # the held orders by `hold_reason`


def step(world: World, desk: Desk, day: date, events: Sequence[BasketReady], output: Output) -> Day:
    """The day's asks, then its baskets' orders (module docstring), the rows written through `output`."""
    done = Day()
    write = output.writes(day)
    agreements = []
    for answer in asks(world, desk, day):
        if isinstance(answer, Refusal):
            done.refused[answer.reason] += 1
        else:
            done.granted += 1
            if write:
                agreements.append(agreement_row(row_id("price_agreement", code=answer.code), answer))
    output.append(day, "price_agreement", agreements)
    numbers: Counter[int] = Counter()
    counts = []
    for event in events:
        numbers[event.customer_id] += 1
        counts.append(numbers[event.customer_id])
    draws = [0.0] * len(events)  # a warm-up order is not written: its source is never drawn (each id draws its own)
    if write and events:
        codes = [world.customers[event.customer_id].code for event in events]
        ids = np.array([business_id("sales_order", order_no(c, day, n)) for c, n in zip(codes, counts, strict=True)])
        draws = uniforms("order", "source", ids=ids).tolist()
    today = running(world, day)
    heads, lines, discounts = [], [], []
    for event, n, draw in zip(events, counts, draws, strict=True):
        made = order(world, desk, event, n, draw, write, today)
        book(world, desk.rules, world.customers[event.customer_id], day, made)
        done.lines += len(made.priced)
        if made.head is not None:
            heads.append(made.head)
        lines += made.lines
        discounts += made.discounts
        if made.status == "held" and made.hold_reason:
            done.held[made.hold_reason] += 1
    done.orders = len(events)
    output.append(day, "sales_order", heads)
    output.append(day, "sales_order_line", lines)
    output.append(day, "sales_order_line_discount", discounts)
    return done
