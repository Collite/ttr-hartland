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
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from typing import Any

import numpy as np

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import Row, read
from hartland_sim.setup.affluence import load as affluence_index
from hartland_sim.setup.customers import Customers
from hartland_sim.setup.portfolio import Portfolio
from hartland_sim.setup.reference import Geography
from hartland_sim.world import Customer

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
                Store(attrs["size_class"], attrs["street_type"], float(attrs["weekly_hours"]), attrs["sunday_open"])
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
                )
            )
    return out


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
        terms=terms,
        units=units,
        scale=scale,
    )


def expected_gross(demand: Demand, day: date, points: Array | None = None) -> Array:
    """(P,) the expected gross a day of each point at flat terms on `day`'s listings and prices."""
    _, listed, price_line = demand.catalogue.shares(day, len(demand.lines))
    gross: Array = (demand.base * (listed * price_line)[demand.point_group]).sum(axis=1)
    return gross if points is None else gross[points]
