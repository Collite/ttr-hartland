"""Setup steps 2 and 5 (`05` §2): the market (`store`, `chain`, from `setup.market`) and the customers with their
delivery points, their KAMs, the planted cases' pins and the first lifecycle phases.

**Who is a customer** (`01` §4.3, NH-D5):
- **Big retail**: each key-account chain of `world.yaml channels.big_retail.accounts` is one customer; its stores
  are its delivery points. The chains that order to their own DCs (`dc_level_orders`: Lidl, Kaufland, Penny) deliver
  to those DCs instead — one per Hartland DC, in that DC's okres, with no store; their stores stay in the market and
  take no orders (⚑R196-F).
- **Gas stations, key accounts**: OMV and Shell, one customer each, their stations the delivery points.
- **Small retail**: every other grocery store of the market that a field rep serves — the independents and the
  convenience franchises (Žabka, Hruška, …) — and every other station shop (Orlen, MOL, EuroOil, …, sub-channel
  `special_retail`). A modern-trade chain that is no key account (Norma) is market only. Each store is its own
  customer: `active` with probability `active_share_at_start` (40 %), its `since` drawn over 2019–2023 (stream
  `acquire`), else `potential`.
- **HoReCa**: the sample of `setup.horeca` (600 customers, 2 000 potential), and two gastro wholesalers.
- **Vending**: the operators (40), synthetic companies, each with a depot in an okres drawn by population.
- **Own shops**: the 12 shops, customers at the transfer list.
- **Web**: the consumers (30 000), synthetic and anonymous, each in a leaf okres drawn by population × affluence,
  their first order drawn over 2019–2023.

A customer is active from its `since` (`World.Customer.active_on`), so a store whose `since` falls in 2023 joins
during the warm-up. Codes: `KA-<chain>`, `MO-`, `GA-`, `VO-`, `AU-`, `VP-`, `ES-` + a number in market order.

**Delivery points**: one per customer and place — a store, a station, a chain's DC, a depot, an own shop, a web
consumer's town — each in a leaf okres (one of Praha's 22 districts, never Praha), served by its region's DC, with
the window of its kind (`world.yaml delivery.windows`). An own shop is a `site`, not a store: its point has no store.

**KAMs** (`kam_customer`): the eight big-retail KAMs one or two chains each, the gas KAM OMV and Shell, the vending
KAMs the operators in turn, the HoReCa KAMs the region-ordered HoReCa customers (potential included) in four blocks.

**Pins** (`anomalies.yaml pins.customers`): the `rank`-th active customer of the channel and area (and chain or store
class) in `osm_order` gets the case's trait. A-1's and A-14's customers are renamed to an invented Czech name on the
customer, its store and its delivery point (⚑R196-B): their behaviour is invented, the business is real.

**Lifecycle**: a small-retail customer's first episode is `potential` from the warm-up's first day, or `occasional`
from its `since` (after a `potential` one when it joins during the warm-up); the month-ends re-phase it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import Row, read
from hartland_sim.rng import Generator, rng
from hartland_sim.setup import market as market_setup
from hartland_sim.setup.affluence import load as affluence
from hartland_sim.setup.classifier import osm_order
from hartland_sim.setup.classifier.chains import BY_CODE
from hartland_sim.setup.market import UNNAMED, Market
from hartland_sim.setup.org import Org, assignment
from hartland_sim.setup.reference import Geography
from hartland_sim.setup.rows import Tables, row
from hartland_sim.world import OPEN, Customer, Duty

SMALL_RETAIL_CLASSES = ("super", "convenience", "minimarket", "village", "petrol", "hyper", "discount")
RETAIL_FRANCHISES = ("convenience_franchise", "petrol")
HORECA_WORD = {"restaurant": "Restaurace", "cafe": "Kavárna", "bar": "Hostinec", "canteen": "Jídelna"}
VENDING_WORDS = ("Nápojové automaty", "Kávové automaty", "Automaty")
WHOLESALERS = ("Gastro velkoobchod Čechy s.r.o.", "Gastro velkoobchod Morava s.r.o.")
FINANCIAL_TRAITS = ("chronic_late_payer", "crosses_credit_limit")  # renamed (⚑R196-B)
PRAHA = "CZ0100"


@dataclass(frozen=True)
class Place:
    """Where a delivery point is: a store, a site, a chain's DC, a depot or a consumer's town."""

    key: str  # the `place` of the delivery point's business key
    kind: str  # a `world.yaml delivery.windows` key
    name: str
    okres: str
    town: str
    store: str | None = None  # osm_id
    street: str | None = None
    postcode: str | None = None
    lat: float | None = None
    lon: float | None = None


@dataclass
class Customers:
    customers: list[Customer]
    names: dict[str, str]  # customer code → its name
    places: dict[str, list[Place]]  # customer code → its delivery points
    pins: dict[str, str]  # customer code → the trait a pin gave it
    stores: list[Row]  # the market's stores, renamed where a pin renames one


def build(
    out: Tables,
    world: WorldConfig,
    stories: StoriesConfig,
    geography: Geography,
    org: Org,
    market: Market | None = None,
) -> Customers:
    """`chain`, `store`, `customer`, `delivery_point`, the KAMs' `assignment`s and `lifecycle_phase` into `out`."""
    market = market or market_setup.build()
    copies = [dict(store) for store in market.stores]  # a pin renames a store: never the market's own row
    stores = {store["osm_id"]: store for store in copies}
    out["chain"] += [
        row(
            "chain",
            chain_id=row_id("chain", code=chain["code"]),
            code=chain["code"],
            name=chain["name"],
            kind=chain["kind"],
            is_key_account=chain["is_key_account"],
        )
        for chain in market.chains
    ]
    made = Customers([], {}, {}, {}, copies)
    _key_accounts(made, world, geography, copies)
    _small_retail(made, world, copies)
    _horeca(made, world, copies, market, geography)
    _vending(made, world, geography)
    _own_shops(made, world, geography)
    _web(made, world, geography)
    _pin(made, stories, stores, {pick.osm_id: pick.kind for pick in market.horeca})
    for store in copies:
        out["store"].append(_store_row(store))
    for customer in made.customers:
        out["customer"].append(_customer_row(customer, made.names[customer.code], world))
        for place in made.places[customer.code]:
            out["delivery_point"].append(_point_row(customer.code, place, world, geography))
    duties = kam_duties(made.customers, org, geography)
    org.duties += duties
    out["assignment"] += [assignment(duty, org.codes) for duty in duties]
    out["lifecycle_phase"] += phases(made.customers, world)
    return made


# ── who the customers are ────────────────────────────────────────────────────


def _active(world: WorldConfig, *key: object) -> tuple[bool, date]:
    """Whether a store starts as a customer, and from when (stream `acquire`)."""
    settings = world.channels.small_retail
    assert settings.active_share_at_start is not None and settings.since_from and settings.since_until
    draw = rng("acquire", *key)
    active = bool(draw.random() < settings.active_share_at_start)
    return active, _day(draw, settings.since_from, settings.since_until)


def _day(draw: Generator, first: date, last: date) -> date:
    return first + timedelta(days=int(draw.integers(0, (last - first).days + 1)))


def _since_from(world: WorldConfig) -> date:
    assert world.channels.small_retail.since_from is not None
    return world.channels.small_retail.since_from


def _store_place(store: Row, kind: str, name: str | None = None) -> Place:
    return Place(
        key=store["osm_id"],
        kind=kind,
        name=name or store["name"],
        okres=store["okres"],
        town=store["town"],
        store=store["osm_id"],
        street=store["street"],
        postcode=store["postcode"],
        lat=store["lat"],
        lon=store["lon"],
    )


def _customer(
    code: str, name: str, channel: str, status: str, since: date | None, *, okres: str | None, **extra: Any
) -> Customer:
    phase = "potential" if status == "potential" else "occasional"
    return Customer(
        row_id("customer", code=code),
        code,
        channel,
        extra.get("sub_channel"),
        status,
        since,
        None,
        okres,
        extra.get("area"),
        extra.get("store"),
        extra.get("chain"),
        phase,
    )


def _add(made: Customers, customer: Customer, name: str, places: Sequence[Place]) -> None:
    made.customers.append(customer)
    made.names[customer.code] = name
    made.places[customer.code] = list(places)


def _key_accounts(made: Customers, world: WorldConfig, geography: Geography, stores: Sequence[Row]) -> None:
    by_chain: dict[str, list[Row]] = defaultdict(list)
    for store in stores:
        if store["chain"] and store["store_class"] != "horeca":
            by_chain[store["chain"]].append(store)
    since = _since_from(world)
    accounts = [("big_retail", code) for code in world.channels.big_retail.accounts]
    accounts += [("gas_ka", code) for code in world.channels.gas_ka.accounts]
    for channel, chain in accounts:
        name = BY_CODE[chain].name
        code = f"KA-{chain}"
        if chain in world.channels.big_retail.dc_level_orders:
            places = [_chain_dc(name, dc, geography) for dc in world.sites.dcs]
        else:
            places = [_store_place(store, "chain_store") for store in by_chain[chain]]
        customer = _customer(code, name, channel, "active", since, okres=None, chain=chain)
        _add(made, customer, name, places)


def _chain_dc(chain: str, dc: str, geography: Geography) -> Place:
    site = geography.sites[dc]
    return Place(
        key=f"DC:{dc}",
        kind="chain_dc",
        name=f"{chain} – distribuční centrum {site['town']}",
        okres=site["okres"],
        town=site["town"],
        lat=site["lat"],
        lon=site["lon"],
    )


def small_retail_pool(stores: Sequence[Row]) -> list[Row]:
    """The stores a field rep serves (module docstring), in market order."""
    pool = []
    for store in stores:
        if store["store_class"] not in SMALL_RETAIL_CLASSES:
            continue
        chain = BY_CODE[store["chain"]] if store["chain"] else None
        if chain is None or (not chain.is_key_account and chain.kind in RETAIL_FRANCHISES):
            pool.append(store)
    return pool


def _small_retail(made: Customers, world: WorldConfig, stores: Sequence[Row]) -> None:
    for n, store in enumerate(small_retail_pool(stores), start=1):
        active, since = _active(world, "small_retail", store["osm_id"])
        station = store["store_class"] == "petrol"
        customer = _customer(
            f"MO-{n:06d}",
            store["name"],
            "small_retail",
            "active" if active else "potential",
            since if active else None,
            okres=store["okres"],
            sub_channel="special_retail" if station else None,
            area=store["area"],
            store=store["osm_id"],
            chain=store["chain"],
        )
        _add(made, customer, store["name"], [_store_place(store, "store")])


def _horeca(made: Customers, world: WorldConfig, copies: Sequence[Row], market: Market, geography: Geography) -> None:
    stores = {store["osm_id"]: store for store in copies}
    settings = world.channels.horeca
    spread = world.channels.small_retail
    assert spread.since_from and spread.since_until
    for n, pick in enumerate(market.horeca, start=1):
        store = stores[pick.osm_id]
        since = _day(rng("acquire", "horeca", pick.osm_id), spread.since_from, spread.since_until)
        active = pick.role == "customer"
        customer = _customer(
            f"GA-{n:06d}",
            store["name"],
            "horeca",
            "active" if active else "potential",
            since if active else None,
            okres=store["okres"],
            area=store["area"],
            store=pick.osm_id,
        )
        _add(made, customer, store["name"], [_store_place(store, "horeca")])
    assert settings.wholesalers is not None
    sides = list(world.sites.dcs)
    for n, name in enumerate(WHOLESALERS[: settings.wholesalers], start=1):
        dc = sides[(n - 1) % len(sides)]
        okres = _depot_okres(geography, "wholesaler", n, dc)
        customer = _customer(f"VO-{n:02d}", name, "horeca", "active", _since_from(world), okres=okres)
        _add(made, customer, name, [_depot(okres, name, geography)])


def _population(geography: Geography, dc: str | None = None) -> tuple[list[str], np.ndarray]:
    leaves = [code for code in geography.leaves if dc is None or geography.dc_for(code) == dc]
    weights = np.array([float(geography.okres[code]["population"]) for code in leaves])
    return leaves, weights / weights.sum()


def _depot_okres(geography: Geography, who: str, n: int, dc: str | None = None) -> str:
    leaves, weights = _population(geography, dc)
    return str(rng("acquire", who, n).choice(leaves, p=weights))


def _town(geography: Geography, okres: str) -> str:
    """An okres's town: Praha for a Prague district, the city of a city okres, else the okres's name."""
    if okres.startswith(PRAHA):
        return "Praha"
    return str(geography.okres[okres]["name"]).removesuffix("-město")


def _depot(okres: str, name: str, geography: Geography) -> Place:
    return Place(key=okres, kind="depot", name=name, okres=okres, town=_town(geography, okres))


def _vending(made: Customers, world: WorldConfig, geography: Geography) -> None:
    operators = world.channels.vending.operators
    assert operators is not None
    surnames = [name["male"] for name in read("names_last")]
    used: list[str] = []
    for n in range(1, operators + 1):
        draw = rng("acquire", "vending", n)
        while True:
            name = f"{VENDING_WORDS[(n - 1) % len(VENDING_WORDS)]} {draw.choice(surnames)} s.r.o."
            if name not in used:
                used.append(name)
                break
        okres = _depot_okres(geography, "vending", n)
        customer = _customer(f"AU-{n:02d}", name, "vending", "active", _since_from(world), okres=okres)
        _add(made, customer, name, [_depot(okres, name, geography)])


def _own_shops(made: Customers, world: WorldConfig, geography: Geography) -> None:
    for site in geography.sites.values():
        if site["kind"] != "shop":
            continue
        code = f"VP-{site['code'].removeprefix('SH-')}"
        place = Place(
            key=site["code"],
            kind="own_shop",
            name=site["name"],
            okres=site["okres"],
            town=site["town"],
            lat=site["lat"],
            lon=site["lon"],
        )
        customer = _customer(code, site["name"], "own_shops", "active", _since_from(world), okres=site["okres"])
        _add(made, customer, site["name"], [place])


def _web(made: Customers, world: WorldConfig, geography: Geography) -> None:
    consumers = world.channels.web.consumers
    assert consumers is not None
    index = affluence()
    leaves = list(geography.leaves)
    weights = np.array([float(geography.okres[code]["population"]) * index[code] for code in leaves])
    weights /= weights.sum()
    settings = world.channels.small_retail
    assert settings.since_from and settings.since_until
    for n in range(1, consumers + 1):
        draw = rng("acquire", "web", n)
        okres = str(draw.choice(leaves, p=weights))
        since = _day(draw, settings.since_from, settings.since_until)
        code = f"ES-{n:06d}"
        name = f"Zákazník e-shopu {n:06d}"
        town = _town(geography, okres)
        customer = _customer(code, name, "web", "active", since, okres=okres)
        _add(made, customer, name, [Place(key=okres, kind="web", name=f"{town} – zásilka", okres=okres, town=town)])


# ── the planted cases ────────────────────────────────────────────────────────


def _pin(made: Customers, stories: StoriesConfig, stores: Mapping[str, Row], horeca_kind: Mapping[str, str]) -> None:
    surnames = [name["male"] for name in read("names_last")]
    for pin in stories.pins.customers:
        if not stories.anomaly(pin.anomaly).enabled:
            continue
        matches = [
            customer
            for customer in made.customers
            if customer.status == "active"
            and customer.channel == pin.channel
            and customer.area == pin.area
            and customer.store is not None
            and (pin.chain is None or customer.chain == pin.chain)
            and (pin.store_class is None or stores[customer.store]["store_class"] == pin.store_class)
        ]
        matches.sort(key=lambda customer: osm_order(customer.store or ""))
        if len(matches) < pin.rank:
            raise ValueError(f"{pin.anomaly}: no active {pin.channel} customer #{pin.rank} in {pin.area}")
        customer = matches[pin.rank - 1]
        customer.traits = (*customer.traits, pin.trait)
        made.pins[customer.code] = pin.trait
        if pin.trait in FINANCIAL_TRAITS:
            assert customer.store is not None
            store = stores[customer.store]
            family = family_plural(str(rng("org", "pin", pin.anomaly, customer.code).choice(surnames)))
            word = HORECA_WORD.get(horeca_kind.get(customer.store, ""), UNNAMED.get(store["store_class"], "Prodejna"))
            name = f"{word} U {family}"
            store["name"] = name
            made.names[customer.code] = name
            made.places[customer.code] = [_store_place(store, made.places[customer.code][0].kind)]


def family_plural(surname: str) -> str:
    """A family by its surname, as `U …` names it — the genitive plural: Novák → Nováků, Černý → Černých, Svoboda →
    Svobodů, Marek → Marků, Vaněk → Vaňků, Němec → Němců."""
    if surname.endswith("ý"):
        return surname[:-1] + "ých"
    if surname.endswith("í"):
        return surname + "ch"
    if surname.endswith("a"):
        return surname[:-1] + "ů"
    for soft, hard in (("něk", "ňk"), ("děk", "ďk"), ("těk", "ťk")):
        if surname.endswith(soft):
            return surname[: -len(soft)] + hard + "ů"
    if surname.endswith("ek"):
        return surname[:-2] + "ků"
    if surname.endswith("ec"):
        return surname[:-2] + "ců"
    return surname + "ů"


# ── duties and phases ────────────────────────────────────────────────────────


def kam_duties(customers: Sequence[Customer], org: Org, geography: Geography) -> list[Duty]:
    """Who answers for each key account, vending operator and HoReCa customer (module docstring)."""
    by_channel: dict[str, list[Customer]] = defaultdict(list)
    for customer in customers:
        by_channel[customer.channel].append(customer)
    duties = []
    for channel in ("big_retail", "gas_ka", "vending"):
        kams = org.kams(channel)
        for i, customer in enumerate(by_channel[channel]):
            kam = kams[i % len(kams)]
            duties.append(Duty(kam.employee_id, "kam_customer", customer.code, kam.hired_on))
    regions = list(dict.fromkeys(geography.region_of.values()))
    horeca = sorted(
        by_channel["horeca"],
        key=lambda c: (regions.index(geography.region_of[geography.area_of[c.okres]]) if c.okres else 0, c.code),
    )
    kams = org.kams("horeca")
    block = -(-len(horeca) // len(kams))
    for i, customer in enumerate(horeca):
        kam = kams[i // block]
        duties.append(Duty(kam.employee_id, "kam_customer", customer.code, kam.hired_on))
    return duties


def phases(customers: Sequence[Customer], world: WorldConfig) -> list[Row]:
    """The first `lifecycle_phase` episodes of the small-retail customers (module docstring)."""
    start = world.history.warm_up_from
    rows = []
    for customer in customers:
        if customer.channel != "small_retail":
            continue
        episodes: list[tuple[str, date, date, str]] = []
        if customer.status == "potential" or customer.since is None:
            episodes.append(("potential", start, OPEN, "Neodebírá"))
        elif customer.since > start:
            episodes.append(("potential", start, customer.since - timedelta(days=1), "Neodebírá"))
            episodes.append(("occasional", customer.since, OPEN, "První objednávka"))
        else:
            episodes.append(("occasional", customer.since, OPEN, "První objednávka"))
        for phase, valid_from, valid_to, reason in episodes:
            rows.append(
                row(
                    "lifecycle_phase",
                    lifecycle_phase_id=row_id("lifecycle_phase", customer=customer.code, valid_from=valid_from),
                    customer_id=customer.customer_id,
                    phase=phase,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    reason=reason,
                )
            )
    return rows


# ── rows ─────────────────────────────────────────────────────────────────────


def _store_row(store: Row) -> Row:
    return row(
        "store",
        store_id=row_id("store", osm_id=store["osm_id"]),
        osm_id=store["osm_id"],
        name=store["name"],
        brand=store["brand"],
        chain_id=row_id("chain", code=store["chain"]) if store["chain"] else None,
        store_class=store["store_class"],
        size_class=store["size_class"],
        street_type=store["street_type"],
        weekly_hours=store["weekly_hours"],
        sunday_open=store["sunday_open"],
        lat=store["lat"],
        lon=store["lon"],
        street=store["street"],
        town=store["town"],
        postcode=store["postcode"],
        okres_id=row_id("okres", code=store["okres"]),
        area_id=row_id("area", code=store["area"]),
        opened_on=store["opened_on"],
        closed_on=store["closed_on"],
    )


def _customer_row(customer: Customer, name: str, world: WorldConfig) -> Row:
    terms = "special_retail" if customer.sub_channel == "special_retail" else customer.channel
    limits = {str(channel): limit for channel, limit in world.policies.credit_limit.items()}
    days = {str(channel): days for channel, days in world.finance.payment_terms_days.items()}
    return row(
        "customer",
        customer_id=customer.customer_id,
        code=customer.code,
        name=name,
        channel=customer.channel,
        sub_channel=customer.sub_channel,
        chain_id=row_id("chain", code=customer.chain) if customer.chain else None,
        store_id=row_id("store", osm_id=customer.store) if customer.store else None,
        okres_id=row_id("okres", code=customer.okres) if customer.okres else None,
        legal_form="s.r.o." if name.endswith(" s.r.o.") else None,
        vat_id=None,
        payment_terms_days=days[terms],
        credit_limit=limits.get(terms),
        price_list_channel=customer.channel,
        since=customer.since,
        until=customer.until,
        status=customer.status,
    )


def _point_row(customer: str, place: Place, world: WorldConfig, geography: Geography) -> Row:
    window = world.delivery.windows[place.kind]
    return row(
        "delivery_point",
        delivery_point_id=row_id("delivery_point", customer=customer, place=place.key),
        customer_id=row_id("customer", code=customer),
        store_id=row_id("store", osm_id=place.store) if place.store else None,
        name=place.name,
        street=place.street,
        town=place.town,
        postcode=place.postcode,
        okres_id=row_id("okres", code=place.okres),
        lat=place.lat,
        lon=place.lon,
        dc_site_id=row_id("site", code=geography.dc_for(place.okres)),
        time_window_from=window.opens,
        time_window_to=window.closes,
        unloading_minutes=window.unloading_minutes,
    )
