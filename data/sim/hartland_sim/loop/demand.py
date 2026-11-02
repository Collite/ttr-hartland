"""The demand model (`04` §8 and S1–S6, `05` §3 step 1): what each delivery point would buy each day, by product line.

**The grain.** Demand is made per delivery point × product line, in cases a day: a chain's stores each sell for
themselves (S4 judges the store), and a customer with one place has one point. The arrays are indexed by the points in
setup's order (each customer's places in turn) and by the lines in `product_line.csv`'s order; the SKUs of a line
share its demand when a basket becomes an order.

**The base** (S4, `base`): `size_base × street_mix × hours_factor × affluence(okres, line)` for a store, `affluence`
alone for a point without one (a chain's DC, a depot, an own shop, a web consumer's town), each × the line's
`line_base` — the cases a line sells at a residential store, relative to the other lines. `hours_factor` =
`weekly_hours ^ elasticity × (1 + sunday_open_uplift)`; `affluence(okres, line)` = the okres's affluence index raised
to the line's exponent (`affluence_exponent`: premium or other). The null world (`s4.yaml enabled: false`) flattens
the street mixes to 1.0 and keeps the rest: the clusters are the mixes.

**The scale.** The base is in units; each point's `scale` turns it into cases a day so that, at the opening's prices
and listings, a customer's expected gross a quarter is its channel's plan (`world.yaml pricing.budgets.
opening_quarter_gross`, and `demand.quarter_gross` for the web and the own shops, which hold no budget). A key-account
chain's plan is spread over its points by their base; any other customer's point gets its channel's plan × its base ÷
the mean base of the channel's points active at the opening, so a bigger store sells more and a store acquired later
sells what its base says. The base and the scale are cached at setup (`Demand.base`) and recomputed for a point
whose store changes (`Demand.rebase`).

**The listing.** A point buys what its listing group lists on the day (`channel_listing`, setup's rows): its
customer's channel, a big-retail chain its own rows, a station shop small retail's (the listing has no sub-channel,
S1.1). Within a line a SKU's share of the cases is its weight (`importance + 1`) ÷ its price at the group's channel
factor, among the SKUs listed that day.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import Row, read
from hartland_sim.rng import pairs, uniforms
from hartland_sim.setup.affluence import load as affluence_index
from hartland_sim.setup.customers import Customers
from hartland_sim.setup.portfolio import Portfolio
from hartland_sim.setup.reference import Geography
from hartland_sim.world import Customer, Promotion, World

QUARTER_DAYS = 365.25 / 4
KEY_ACCOUNTS = ("big_retail", "gas_ka")  # one customer per chain, its plan spread over its points
CHANNELS = ("big_retail", "small_retail", "gas_ka", "own_shops", "web", "vending", "horeca")
Array = np.ndarray[Any, Any]


@dataclass(frozen=True)
class Line:
    code: str
    is_premium: bool
    is_professional: bool


def lines() -> tuple[Line, ...]:
    """The product lines, in `product_line.csv`'s order — the demand arrays' second axis."""
    return tuple(Line(row["code"], row["is_premium"], row["is_professional"]) for row in read("product_line"))


@dataclass(frozen=True)
class Store:
    """The store attributes the base reads (S4)."""

    size_class: str
    street_type: str
    weekly_hours: float
    sunday_open: bool
    store_class: str  # a promotion's eligibility may name it


@dataclass(frozen=True)
class Spot:
    """A delivery point as the demand model reads it."""

    delivery_point_id: int
    customer_id: int
    channel: str  # the customer's
    group: tuple[str, str | None]  # the listing it buys from: (channel, chain)
    okres: str
    region: str
    store: Store | None
    chain: str | None = None  # the customer's chain (a key account, a franchise)


@dataclass(frozen=True)
class StoreTerms:
    """S4's parameters (`s4.yaml`); `of` flattens the street mixes in the null world."""

    size_base: Mapping[str, float]
    street_mix: Mapping[str, Mapping[str, float]]
    sunday_open_uplift: float
    weekly_hours_elasticity: float
    line_base: Mapping[str, float]
    affluence_exponent: Mapping[str, float]  # premium · other

    @classmethod
    def of(cls, stories: StoriesConfig) -> StoreTerms:
        story = stories.story("S4")
        parameters = story.parameters
        mixes = {street: dict(mix) if story.enabled else {} for street, mix in parameters["street_mix"].items()}
        return cls(  # plain dicts: the config's read-only mappings do not pickle into the checkpoint
            dict(parameters["size_base"]),
            mixes,
            parameters["sunday_open_uplift"],
            parameters["weekly_hours_elasticity"],
            dict(parameters["line_base"]),
            dict(parameters["affluence_exponent"]),
        )


def base(spots: Sequence[Spot], line_set: Sequence[Line], terms: StoreTerms, affluence: Mapping[str, float]) -> Array:
    """The base of each spot × line, in units (module docstring)."""
    line_base = np.array([terms.line_base.get(line.code, 1.0) for line in line_set])
    exponent = np.array([terms.affluence_exponent["premium" if line.is_premium else "other"] for line in line_set])
    streets = sorted({spot.store.street_type for spot in spots if spot.store is not None})
    mixes = np.array(
        [[terms.street_mix.get(street, {}).get(line.code, 1.0) for line in line_set] for street in streets]
    )
    street = np.array([streets.index(spot.store.street_type) if spot.store else -1 for spot in spots], dtype=np.int64)
    store = np.ones(len(spots))
    for i, spot in enumerate(spots):
        if spot.store is not None:
            sunday = 1 + terms.sunday_open_uplift if spot.store.sunday_open else 1.0
            hours = spot.store.weekly_hours**terms.weekly_hours_elasticity
            store[i] = terms.size_base[spot.store.size_class] * hours * sunday
    rich = np.array([affluence[spot.okres] for spot in spots])
    out = line_base * rich[:, None] ** exponent * store[:, None]
    if streets:
        out = np.where((street >= 0)[:, None], out * mixes[np.maximum(street, 0)], out)
    return np.asarray(out, dtype=float)


def same(first: object, second: object) -> bool:
    """Two states equal field by field, numpy arrays by their dtype and values (a dataclass's `==` cannot compare
    arrays): what a checkpoint's round trip must keep."""
    if type(first) is not type(second):
        return False
    for name in (f.name for f in fields(first)):  # type: ignore[arg-type]
        mine, theirs = getattr(first, name), getattr(second, name)
        if isinstance(mine, np.ndarray):
            if not (isinstance(theirs, np.ndarray) and mine.dtype == theirs.dtype and np.array_equal(mine, theirs)):
                return False
        elif is_dataclass(mine) and not isinstance(mine, type):
            if not same(mine, theirs):
                return False
        elif mine != theirs:
            return False
    return True


@dataclass(eq=False)
class Catalogue:
    """The SKUs and what each listing group lists when (`channel_listing`), as arrays: SKUs × groups."""

    sku_ids: Array  # (S,) int64
    sku_codes: tuple[str, ...]
    line: Array  # (S,) the line's index
    weight: Array  # (S,) importance + 1
    units: Array  # (S,) units per case
    price: Array  # (G, S) the case price at the group's channel factor
    groups: tuple[tuple[str, str | None], ...]
    listed_from: Array  # (G, S) the first day listed, as an ordinal (0: never)
    listed_to: Array  # (G, S) the last day listed, as an ordinal

    def __eq__(self, other: object) -> bool:
        return same(self, other)

    __hash__ = None  # type: ignore[assignment]

    def live(self, day: date) -> Array:
        """(G, S): listed on `day`."""
        today = day.toordinal()
        return (self.listed_from > 0) & (self.listed_from <= today) & (today <= self.listed_to)

    def shares(self, day: date, n_lines: int) -> tuple[Array, Array, Array]:
        """On `day`: each SKU's share of its line's cases in each group (G, S); whether a group lists the line
        (G, L); the line's mean case price in each group (G, L)."""
        live = self.live(day)
        raw = np.where(live, self.weight / np.where(self.price > 0, self.price, 1.0), 0.0)
        totals = np.zeros((len(self.groups), n_lines))
        for g in range(len(self.groups)):
            totals[g] = np.bincount(self.line, weights=raw[g], minlength=n_lines)
        per_line = totals[:, self.line]
        share = np.divide(raw, per_line, out=np.zeros_like(raw), where=per_line > 0)
        listed = totals > 0
        price_line = np.zeros_like(totals)
        for g in range(len(self.groups)):
            price_line[g] = np.bincount(self.line, weights=share[g] * self.price[g], minlength=n_lines)
        return share, listed, price_line


def catalogue(
    portfolio: Portfolio, line_set: Sequence[Line], listing_rows: Sequence[Row], world: WorldConfig
) -> Catalogue:
    """The catalogue from setup's `channel_listing` rows (each group in the rows' order)."""
    index = {line.code: i for i, line in enumerate(line_set)}
    skus = portfolio.skus
    position = {sku.sku_id: s for s, sku in enumerate(skus)}
    chains = {row_id("chain", code=code): code for code in world.channels.big_retail.accounts}
    groups: dict[tuple[str, str | None], int] = {}
    for listing in listing_rows:
        groups.setdefault((listing["channel"], chains.get(listing["chain_id"])), len(groups))
    listed_from = np.zeros((len(groups), len(skus)), dtype=np.int64)
    listed_to = np.zeros((len(groups), len(skus)), dtype=np.int64)
    for listing in listing_rows:
        g = groups[listing["channel"], chains.get(listing["chain_id"])]
        s = position[listing["sku_id"]]
        listed_from[g, s] = listing["valid_from"].toordinal()
        listed_to[g, s] = listing["valid_to"].toordinal()
    factors = {str(channel): factor for channel, factor in world.pricing.channel_factor.items()}
    base_price = np.array([sku.base_price_case for sku in skus])
    return Catalogue(
        sku_ids=np.array([sku.sku_id for sku in skus], dtype=np.int64),
        sku_codes=tuple(sku.code for sku in skus),
        line=np.array([index[sku.line] for sku in skus], dtype=np.int64),
        weight=np.array([sku.importance + 1 for sku in skus], dtype=float),
        units=np.array([sku.units_per_case for sku in skus], dtype=np.int64),
        price=np.array([base_price * factors[channel] for channel, _ in groups]),
        groups=tuple(groups),
        listed_from=listed_from,
        listed_to=listed_to,
    )


@dataclass(eq=False)
class Demand:
    """The demand model's state in the `World` (`World.demand`): the points and their cached base, the customers'
    side of the model, the catalogue. Arrays only, filled in a fixed order, so the checkpoint pickles stably."""

    lines: tuple[Line, ...]
    catalogue: Catalogue
    spots: tuple[Spot, ...]  # the points, in setup's order
    point_ids: Array  # (P,) delivery_point_id
    point_customer: Array  # (P,) the customer's index in `customer_ids`
    point_channel: Array  # (P,) index into CHANNELS
    point_group: Array  # (P,) index into `catalogue.groups`
    point_region: Array  # (P,) index into `regions`
    point_affluence: Array  # (P,) the okres's affluence index
    regions: tuple[str, ...]
    customer_ids: Array  # (C,) in the World's order
    customer_start: Array  # (C + 1,) customer c's points are customer_start[c]:customer_start[c + 1]
    terms: StoreTerms
    units: Array  # (P, L) the base, before the scale
    scale: Array  # (P,) units → cases a day

    def __eq__(self, other: object) -> bool:
        return same(self, other)

    __hash__ = None  # type: ignore[assignment]

    @property
    def base(self) -> Array:
        """(P, L) cases a day at flat terms: the base × the scale (module docstring)."""
        return self.units * self.scale[:, None]

    def rebase(self, changed: Mapping[int, Spot], affluence: Mapping[str, float]) -> None:
        """Stores changed: the base of each changed point (by its index) again. The scale stays, so a bigger store
        sells more."""
        spots = list(self.spots)
        for p, spot in sorted(changed.items()):
            spots[p] = spot
            self.units[p] = base([spot], self.lines, self.terms, affluence)[0]
        self.spots = tuple(spots)


def spots(customers: Sequence[Customer], made: Customers, geography: Geography) -> list[Spot]:
    """Every customer's delivery points as spots, customers in their order, places in theirs."""
    stores = {store["osm_id"]: store for store in made.stores}
    out = []
    for customer in customers:
        if customer.channel == "big_retail":
            group: tuple[str, str | None] = ("big_retail", customer.chain)
        else:
            group = (customer.channel, None)
        for place in made.places[customer.code]:
            attrs = stores.get(place.store) if place.store else None
            store = (
                Store(
                    attrs["size_class"],
                    attrs["street_type"],
                    float(attrs["weekly_hours"]),
                    attrs["sunday_open"],
                    attrs["store_class"],
                )
                if attrs is not None
                else None
            )
            out.append(
                Spot(
                    row_id("delivery_point", customer=customer.code, place=place.key),
                    customer.customer_id,
                    customer.channel,
                    group,
                    place.okres,
                    geography.region_of[geography.area_of[place.okres]],
                    store,
                    customer.chain,
                )
            )
    return out


def starts(point_customer: Array, customers: int) -> Array:
    """(C + 1,) where each customer's points start; the points come customer by customer."""
    if len(point_customer) and (np.diff(point_customer) < 0).any():
        raise ValueError("the points are not in their customers' order")
    return np.concatenate([[0], np.cumsum(np.bincount(point_customer, minlength=customers))]).astype(np.int64)


def plans(world: WorldConfig) -> dict[str, float]:
    """A customer's expected gross a quarter by channel (module docstring)."""
    out = {str(channel): gross for channel, gross in world.pricing.budgets.opening_quarter_gross.items()}
    out.update(world.demand.quarter_gross)
    return out


def opening(
    customers: Sequence[Customer],
    made: Customers,
    geography: Geography,
    portfolio: Portfolio,
    listing_rows: Sequence[Row],
    world: WorldConfig,
    stories: StoriesConfig,
) -> Demand:
    """The demand state at the opening: the points, their base and scale (module docstring)."""
    line_set = lines()
    goods = catalogue(portfolio, line_set, listing_rows, world)
    found = spots(customers, made, geography)
    terms = StoreTerms.of(stories)
    index = affluence_index()
    units = base(found, line_set, terms, index)
    regions = tuple(dict.fromkeys(geography.region_of.values()))
    groups = {group: g for g, group in enumerate(goods.groups)}
    customer_index = {customer.customer_id: c for c, customer in enumerate(customers)}
    point_group = np.array([groups[spot.group] for spot in found], dtype=np.int64)
    first = world.history.warm_up_from
    _, listed, price_line = goods.shares(first, len(line_set))
    gross = (units * (listed * price_line)[point_group]).sum(axis=1)  # a day, at a scale of 1
    by_id = {customer.customer_id: customer for customer in customers}
    scale = np.zeros(len(found))
    plan = plans(world)
    chain_points: dict[int, list[int]] = defaultdict(list)
    channel_points: dict[str, list[int]] = defaultdict(list)
    for p, spot in enumerate(found):
        if spot.channel in KEY_ACCOUNTS:
            chain_points[spot.customer_id].append(p)
        elif by_id[spot.customer_id].active_on(first):
            channel_points[spot.channel].append(p)
    for customer_id, points in chain_points.items():
        scale[points] = plan[by_id[customer_id].channel] / QUARTER_DAYS / gross[points].sum()
    means = {channel: gross[points].mean() for channel, points in channel_points.items()}
    for p, spot in enumerate(found):
        if spot.channel not in KEY_ACCOUNTS:
            scale[p] = plan[spot.channel] / QUARTER_DAYS / means[spot.channel]
    return Demand(
        lines=line_set,
        catalogue=goods,
        spots=tuple(found),
        point_ids=np.array([spot.delivery_point_id for spot in found], dtype=np.int64),
        point_customer=np.array([customer_index[spot.customer_id] for spot in found], dtype=np.int64),
        point_channel=np.array([CHANNELS.index(spot.channel) for spot in found], dtype=np.int64),
        point_group=point_group,
        point_region=np.array([regions.index(spot.region) for spot in found], dtype=np.int64),
        point_affluence=np.array([index[spot.okres] for spot in found]),
        regions=regions,
        customer_ids=np.array([customer.customer_id for customer in customers], dtype=np.int64),
        customer_start=starts(np.array([customer_index[spot.customer_id] for spot in found]), len(customers)),
        terms=terms,
        units=units,
        scale=scale,
    )


def expected_gross(demand: Demand, day: date, points: Array | None = None) -> Array:
    """(P,) the expected gross a day of each point at flat terms on `day`'s listings and prices."""
    _, listed, price_line = demand.catalogue.shares(day, len(demand.lines))
    gross: Array = (demand.base * (listed * price_line)[demand.point_group]).sum(axis=1)
    return gross if points is None else gross[points]


# ── season · weather · weekday (S1) ──────────────────────────────────────────

TRADE = ("big_retail", "small_retail", "gas_ka", "vending", "horeca")  # s1.yaml's `trade` week
WEEKS = 53  # an ISO year's weeks at most; week 53 is week 52's


def _weeks(ranges: Sequence[Sequence[int]]) -> list[int]:
    """The ISO weeks (1–52) of inclusive ranges; a range may wrap over the new year ([45, 8])."""
    out: list[int] = []
    for first, last in ranges:
        out += list(range(first, last + 1)) if first <= last else [*range(first, 53), *range(1, last + 1)]
    return out


def curve(peak: Sequence[Sequence[int]], trough: Sequence[Sequence[int]], amplitude: float) -> Array:
    """A 52-week multiplier (index 0 = week 1) with mean 1 and peak weeks ÷ trough weeks = `amplitude`.

    The peak weeks stand at `amplitude`, the trough weeks at 1, a week between them on the line between the nearer
    peak and trough week; then the whole is scaled to a mean of 1. With no trough the other weeks stand at 1; with
    neither (Decaf) the curve is a cosine from its peak in week 1 to its trough in week 27."""
    weeks = np.arange(1, 53)
    up, down = _weeks(peak), _weeks(trough)
    if not up and not down:
        raw = 1 + (amplitude - 1) * (1 + np.cos(2 * np.pi * (weeks - 1) / 52)) / 2
    elif not down:
        raw = np.where(np.isin(weeks, up), amplitude, 1.0)
    else:
        raw = np.empty(52)
        for i, week in enumerate(range(1, 53)):
            if week in up:
                raw[i] = amplitude
            elif week in down:
                raw[i] = 1.0
            else:
                to_peak = min(_apart(week, w) for w in up)
                to_trough = min(_apart(week, w) for w in down)
                raw[i] = 1 + (amplitude - 1) * to_trough / (to_peak + to_trough)
    return np.asarray(raw / raw.mean(), dtype=float)


def _apart(first: int, second: int) -> int:
    gap = abs(first - second)
    return min(gap, 52 - gap)


@dataclass(frozen=True)
class Seasons:
    """S1's terms (`s1.yaml`): the season of each channel × line × ISO week, the December gift uplift, the cold
    weather's lift and the week's shape. The null world (`enabled: false`) flattens the season, the gift and the
    weather; the week's shape stays — it is when the trade orders and the consumers shop, not a story.

    **The week**: own shops and the web sell to consumers, so their latent demand follows their week (`weekday`); the
    trade's shops sell every day, and the trade's week is the day it orders on (`order_weekday`, read by the order
    trigger). Each shape is scaled to a mean of 1, so the week moves demand, never its volume."""

    season: Array  # (CH, L, 53) by CHANNELS × line × ISO week − 1
    gift: Array  # (L,) the December multiplier
    cold_pct: Array  # (L,) % more demand per °C below `cold_below_c`
    cold_below_c: float
    weekday: Array  # (CH, 7) the latent demand's week, Monday first
    order_weekday: Array  # (7,) the trade's order days

    @classmethod
    def of(cls, stories: StoriesConfig, line_set: Sequence[Line]) -> Seasons:
        story = stories.story("S1")
        parameters = story.parameters
        on = story.enabled
        season = np.ones((len(CHANNELS), len(line_set), WEEKS))
        for i, line in enumerate(line_set):
            if line.code in parameters["curves"]:
                spec = parameters["curves"][line.code]
                season[:, i, :52] = curve(spec["peak_weeks"], spec["trough_weeks"], spec["amplitude"])
            else:  # a Professional line follows its channel
                for channel, spec in parameters["channel_seasons"].items():
                    season[CHANNELS.index(channel), i, :52] = curve(spec["weeks"], [], spec["amplitude"])
        season[:, :, 52] = season[:, :, 51]
        gifts = parameters["december_gift_uplift"]
        weekday = np.ones((len(CHANNELS), 7))
        for channel in ("own_shops", "web"):
            shape = np.array(parameters["weekday"][channel], dtype=float)
            weekday[CHANNELS.index(channel)] = shape / shape.mean()
        trade = np.array(parameters["weekday"]["trade"], dtype=float)
        return cls(
            season=season if on else np.ones_like(season),
            gift=np.array([gifts.get(line.code, 1.0) if on else 1.0 for line in line_set]),
            cold_pct=np.array(
                [parameters["curves"].get(line.code, {}).get("cold_pct_per_c", 0) if on else 0 for line in line_set],
                dtype=float,
            ),
            cold_below_c=float(parameters["cold_below_c"]),
            weekday=weekday,
            order_weekday=trade / trade.sum(),
        )

    def of_day(self, day: date) -> Array:
        """(CH, L) the season × the gift × the week's shape on `day`."""
        week = day.isocalendar().week
        out = self.season[:, :, week - 1] * self.weekday[:, day.weekday(), None]
        return out * self.gift if day.month == 12 else out

    def cold(self, temps: Array) -> Array:
        """(R, L) the weather term for the regions' temperatures of the day: + `cold_pct` % per °C below."""
        below = np.maximum(self.cold_below_c - temps, 0.0)
        return np.asarray(1 + below[:, None] * self.cold_pct / 100)


# ── promotion · display · discount · competitor (S2, S5, S6, A-6, A-10) ───────


@dataclass(frozen=True)
class SkuLift:
    """A lift on one SKU, not on its whole line: `uplift` × the SKU's own cases at each point (0 where it does not
    apply, negative for a dip); `cannibalises` = its line siblings give up s2.yaml's share of the extra cases."""

    sku: int  # the catalogue's index
    line: int
    uplift: Array  # (P,)
    cannibalises: bool = True


@dataclass(frozen=True)
class Today:
    """The day's multiplicative terms, (P, L), and the SKU lifts on top of them."""

    line: Array
    skus: tuple[SkuLift, ...] = ()


@dataclass(frozen=True)
class PromotionTerms:
    """S2's uplift model (`s2.yaml`): `uplift = base_uplift(mechanic) × premium_factor × affluence_index ^ exponent ×
    tv_factor`, the post-promotion dip and the siblings' share. The null world keeps the mechanics' base uplift and
    the dip and flattens the three factors: a promotion still lifts, but no more for premium, rich or TV.

    A promotion with no SKU named (`price_off`, `free_goods_same`) lifts its whole line at the eligible points; one that
    names SKUs lifts those: a target SKU by its mechanic, a condition SKU by `conditional_condition`, a conditional
    price-off's reward SKU by `conditional_reward` (a free-goods reward is given, not bought). A lifted SKU takes
    `cannibalisation_share` of its extra cases from its line siblings. The dip follows for `weeks` on what was lifted.
    A-10's promotion (when enabled) lifts its SKU by the case's `uplift_pct` and takes `sibling_dip_pct` from the one
    named sibling instead. The demand counts the cases that move: a 1+1's free half is the pricing's (S1.3)."""

    base_uplift: Mapping[str, float]
    premium_factor: Mapping[str, float]
    affluence_exponent: Mapping[str, float]
    tv_factor: Mapping[str, float]
    dip: float  # the multiplier after a promotion, 1 + pct ÷ 100
    dip_days: int
    cannibalisation: float
    cannibal: Mapping[str, Any] | None  # A-10's parameters when enabled

    @classmethod
    def of(cls, stories: StoriesConfig) -> PromotionTerms:
        story = stories.story("S2")
        parameters = story.parameters
        flat = not story.enabled
        a10 = stories.anomaly("A-10")
        return cls(
            dict(parameters["base_uplift"]),
            {key: 1.0 for key in parameters["premium_factor"]} if flat else dict(parameters["premium_factor"]),
            {key: 0.0 for key in parameters["affluence_exponent"]} if flat else dict(parameters["affluence_exponent"]),
            {key: 1.0 for key in parameters["tv_factor"]} if flat else dict(parameters["tv_factor"]),
            1 + parameters["post_promotion_dip"]["pct"] / 100,
            7 * parameters["post_promotion_dip"]["weeks"],
            parameters["cannibalisation_share"],
            dict(a10.parameters) if a10.enabled else None,
        )

    def price_off(self, pct: float) -> float:
        """The base uplift of a price-off of `pct` %: on the line through 0, s2.yaml's 10 % and its 20 % points."""
        ten, twenty = self.base_uplift["price_off_10"], self.base_uplift["price_off_20"]
        return ten * pct / 10 if pct <= 10 else ten + (twenty - ten) * (pct - 10) / 10

    def mechanic(self, promotion: Promotion) -> float:
        """The base uplift of a promotion's own mechanic (a whole line, or its target SKU)."""
        if promotion.mechanic == "free_goods_same":
            return self.base_uplift["free_goods_1_1"]
        return self.price_off(promotion.discount_pct or 0.0)

    def factor(self, promotion: Promotion, line: Line, affluence: Array) -> Array:
        """(P,) premium × affluence × TV at each point's affluence index."""
        tv = "tv" if promotion.tv else "no_tv"
        premium = self.premium_factor[tv] if line.is_premium else self.premium_factor["not_premium"]
        return np.asarray(premium * affluence ** self.affluence_exponent[tv] * self.tv_factor[tv])

    def named(self, promotion: Promotion) -> list[tuple[str, float]]:
        """The SKUs a promotion lifts by name, with their base uplift; empty = the whole line."""
        if promotion.mechanic == "conditional_price_off":
            out = [(promotion.condition_sku, self.base_uplift["conditional_condition"])]
            out.append((promotion.reward_sku, self.base_uplift["conditional_reward"]))
        elif promotion.mechanic == "free_goods_other":
            out = [(promotion.condition_sku, self.base_uplift["conditional_condition"])]
        else:
            out = [(promotion.target_sku, self.mechanic(promotion))] if promotion.target_sku else []
        return [(code, uplift) for code, uplift in out if code is not None]


def eligible(promotion: Promotion, demand: Demand, phases: Sequence[str] | None = None) -> Array:
    """(P,) the points a promotion's eligibility admits: every scope kind it names must hold (`promotion_eligibility`:
    channel, chain, region, store_class, phase — the customer's phase on the day, `phases` in the customers' order)."""
    mask = np.ones(len(demand.spots), dtype=bool)
    for kind, values in promotion.eligibility:
        if kind == "channel":
            mask &= np.isin(demand.point_channel, [CHANNELS.index(value) for value in values if value in CHANNELS])
        elif kind == "region":
            mask &= np.isin(demand.point_region, [demand.regions.index(v) for v in values if v in demand.regions])
        elif kind == "chain":
            mask &= np.array([spot.chain in values for spot in demand.spots], dtype=bool)
        elif kind == "store_class":
            mask &= np.array([spot.store is not None and spot.store.store_class in values for spot in demand.spots])
        elif kind == "phase":
            if phases is None:
                raise ValueError(f"{promotion.code}: a phase scope needs the customers' phases")
            mask &= np.isin(np.array(phases)[demand.point_customer], values)
        else:
            raise ValueError(f"{promotion.code}: no scope kind {kind!r}")
    return mask


@dataclass(frozen=True)
class Push:
    """A competitor's push (A-6): the line's demand × (1 − `cut`) in the region from `first` to `last`."""

    region: str
    line: str
    cut: float
    first: date
    last: date


def pushes(stories: StoriesConfig) -> tuple[Push, ...]:
    """The planted competitor pushes: A-6 when enabled."""
    a6 = stories.anomaly("A-6")
    if not a6.enabled:
        return ()
    p = a6.parameters
    return (Push(p["region"], p["line"], p["dip_pct"] / 100, p["from"], p["until"]),)


def elasticity(demand: Demand, stories: StoriesConfig) -> Array:
    """(C, L) each customer × line's hidden elasticity (S6): `high` for about `high_share` of them, else `low`, drawn
    from stream `elasticity` by (customer, line). A high one is likelier where shoppers watch prices: by the street of
    the customer's first store (`street_factor`, 1.0 without a store) ÷ the mean affluence of its points, scaled so the
    share of high ones stays `high_share`. The null world gives every customer × line the mean elasticity."""
    story = stories.story("S6")
    spec = story.parameters["elasticity"]
    shape = (len(demand.customer_ids), len(demand.lines))
    if not story.enabled:
        return np.full(shape, float(spec["high_share"] * spec["high"] + (1 - spec["high_share"]) * spec["low"]))
    first = demand.customer_start[:-1]
    rich = np.add.reduceat(demand.point_affluence, first) / np.diff(demand.customer_start)
    streets = [demand.spots[p].store for p in first]
    street = np.array([spec["street_factor"].get(store.street_type, 1.0) if store else 1.0 for store in streets])
    weight = street / rich
    chance = np.clip(spec["high_share"] * weight / weight.mean(), 0.0, 0.95)
    lines_ids = np.arange(len(demand.lines), dtype=np.int64)
    draw = uniforms("elasticity", ids=pairs(demand.customer_ids[:, None], lines_ids[None, :]))
    return np.asarray(np.where(draw < chance[:, None], spec["high"], spec["low"]), dtype=float)


@dataclass
class Terms:
    """The terms of a run, from the configs and the `Demand` — never pickled: a prolong builds them again. Caches each
    promotion's eligible points and the points' and customers' index by id."""

    demand: Demand
    seasons: Seasons
    promotions: PromotionTerms
    pushes: tuple[Push, ...]
    elasticity: Array  # (C, L)
    _eligible: dict[int, Array] = field(default_factory=dict, repr=False)
    _point: dict[int, int] = field(default_factory=dict, repr=False)
    _customer: dict[int, int] = field(default_factory=dict, repr=False)

    @classmethod
    def of(cls, demand: Demand, stories: StoriesConfig) -> Terms:
        return cls(
            demand,
            Seasons.of(stories, demand.lines),
            PromotionTerms.of(stories),
            pushes(stories),
            elasticity(demand, stories),
        )

    def point(self, delivery_point_id: int) -> int:
        if not self._point:
            self._point = {int(point): p for p, point in enumerate(self.demand.point_ids)}
        return self._point[delivery_point_id]

    def customer(self, customer_id: int) -> int:
        if not self._customer:
            self._customer = {int(customer): c for c, customer in enumerate(self.demand.customer_ids)}
        return self._customer[customer_id]

    def today(self, world: World, day: date, temps: Array) -> Today:
        """`day`'s terms (`04` §8): season × gift × week and the weather (S1); promotions and their dips (S2); displays
        while a loan runs (S5); agreed discounts × the customer-line's elasticity (S6); competitors' pushes (A-6).
        `temps` are the regions' temperatures of the day, in `Demand.regions`' order."""
        demand = self.demand
        index = {line.code: i for i, line in enumerate(demand.lines)}
        factor = self.seasons.of_day(day)[demand.point_channel] * self.seasons.cold(temps)[demand.point_region]
        skus: list[SkuLift] = []
        last = timedelta(days=self.promotions.dip_days)
        for promotion in world.promotions.values():
            if promotion.valid_from <= day <= promotion.valid_to + last:
                skus += self._promotion(promotion, day, factor, index, world)
        for loan in world.loans.values():
            if loan.lent_on <= day and (loan.returned_on is None or day < loan.returned_on):
                factor[self.point(loan.delivery_point_id), index[loan.line]] *= 1 + loan.uplift
        for agreement in world.agreements.values():
            if agreement.valid_from <= day <= agreement.valid_to:
                c, i = self.customer(agreement.customer_id), index[agreement.line]
                points = slice(int(demand.customer_start[c]), int(demand.customer_start[c + 1]))
                factor[points, i] *= 1 + self.elasticity[c, i] * agreement.discount_pct / 100
        for push in self.pushes:
            if push.first <= day <= push.last and push.region in demand.regions:
                factor[demand.point_region == demand.regions.index(push.region), index[push.line]] *= 1 - push.cut
        return Today(factor, tuple(skus))

    def _mask(self, promotion: Promotion, day: date, world: World) -> Array:
        if any(kind == "phase" for kind, _ in promotion.eligibility):
            return eligible(promotion, self.demand, [c.phase_on(day) for c in world.customers.values()])
        found = self._eligible.get(promotion.promotion_id)
        if found is None:
            found = self._eligible[promotion.promotion_id] = eligible(promotion, self.demand)
        return found

    def _promotion(
        self, promotion: Promotion, day: date, factor: Array, index: Mapping[str, int], world: World
    ) -> list[SkuLift]:
        terms, demand = self.promotions, self.demand
        mask = self._mask(promotion, day, world)
        line = index[promotion.line]
        after = day > promotion.valid_to
        named = terms.named(promotion)
        if not named:
            if after:
                factor[mask, line] *= terms.dip
            else:
                lift = terms.mechanic(promotion) * terms.factor(promotion, demand.lines[line], demand.point_affluence)
                factor[mask, line] *= 1 + lift[mask]
            return []
        codes = demand.catalogue.sku_codes
        a10 = terms.cannibal if terms.cannibal and terms.cannibal["promotion"] == promotion.code else None
        out = []
        for code, base_uplift in named:
            sku = codes.index(code)
            own = int(demand.catalogue.line[sku])
            if after:
                out.append(SkuLift(sku, own, np.where(mask, terms.dip - 1, 0.0), cannibalises=False))
            elif a10 is not None and code == a10["sku"]:
                out.append(SkuLift(sku, own, np.where(mask, a10["uplift_pct"] / 100, 0.0), cannibalises=False))
                sibling = codes.index(a10["sibling"])
                dip = np.where(mask, -a10["sibling_dip_pct"] / 100, 0.0)
                out.append(SkuLift(sibling, int(demand.catalogue.line[sibling]), dip, cannibalises=False))
            else:
                lift = base_uplift * terms.factor(promotion, demand.lines[own], demand.point_affluence)
                out.append(SkuLift(sku, own, np.where(mask, lift, 0.0)))
        return out


def lifted(
    cases: Array, lifts: Sequence[SkuLift], share: Array, cannibalisation: float
) -> tuple[Array, dict[int, Array]]:
    """The day's line cases (P, L) with the SKU lifts on top, and each lifted SKU's cases (P,) beyond its share of
    its line's pool. `share` is each point's SKU shares, (P, S) — `Catalogue.shares` by the points' groups.

    A lift of `u` on SKU `s` (share σ of its line's `D` cases) adds `σ·D·u` cases of `s`; when it cannibalises, its
    siblings give up `c·σ·D·u` of theirs. The line gains the difference; `s`'s extra (beyond σ of the pool) is set so
    that `sku_cases` gives `s` exactly `σ·D·(1 + u)` and the siblings their share of the rest."""
    out = cases.copy()
    extras: dict[int, Array] = {}
    for lift in lifts:
        sigma = share[:, lift.sku]
        extra = sigma * cases[:, lift.line] * lift.uplift
        alone = sigma >= 1.0  # no sibling to take from
        taken = np.where(lift.cannibalises & ~alone, cannibalisation * extra, 0.0)
        out[:, lift.line] += extra - taken
        back = np.divide(sigma * taken, 1 - sigma, out=np.zeros_like(sigma), where=~alone)
        extras[lift.sku] = extras.get(lift.sku, np.zeros_like(sigma)) + extra + back
    return out, extras


def sku_cases(line_cases: Array, extras: Mapping[int, Array], share: Array, sku_line: Array) -> Array:
    """(P, S) each SKU's cases: its share of its line's pool (the line's cases less its SKUs' extras) plus its extra;
    never below 0."""
    pool = line_cases.copy()
    for sku, extra in extras.items():
        pool[:, sku_line[sku]] -= extra
    out = share * pool[:, sku_line]
    for sku, extra in extras.items():
        out[:, sku] += extra
    return np.asarray(np.maximum(out, 0.0))
