"""The base term of the demand model (G1 S1.2 T2, `04` S4): the street mixes, Sunday opening, the hours elasticity
and the affluence of the okres; the null world's flat mixes; the base and scale cached at setup."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.loop.demand import KEY_ACCOUNTS, QUARTER_DAYS, Spot, Store, StoreTerms, base, expected_gross, lines
from hartland_sim.setup.run import Setup

STORIES = StoriesConfig.load()
LINES = lines()
CODES = [line.code for line in LINES]
STREETS = ("residential", "transport_hub", "village", "high_street", "shopping_centre", "petrol")
AFFLUENCE = {"CZ0201": 1.0, "CZ0100": 1.3}


def _spot(street: str, okres: str = "CZ0201", **store: object) -> Spot:
    attrs = {"size_class": "M", "weekly_hours": 70.0, "sunday_open": False, "store_class": "convenience", **store}
    return Spot(1, 1, "small_retail", ("small_retail", None), okres, "SJC", Store(street_type=street, **attrs))  # type: ignore[arg-type]


def _base(spots: list[Spot], stories: StoriesConfig = STORIES) -> np.ndarray:
    return base(spots, LINES, StoreTerms.of(stories), AFFLUENCE)


def _ratio(street: str, stories: StoriesConfig = STORIES) -> dict[str, float]:
    """The street's base over a residential store's, line by line (the same size, hours and okres)."""
    found = _base([_spot(street), _spot("residential")], stories)
    return {code: round(float(found[0, i] / found[1, i]), 6) for i, code in enumerate(CODES)}


def test_six_stores_one_per_street_reproduce_the_street_mixes() -> None:
    mixes = STORIES.story("S4").parameters["street_mix"]
    assert set(mixes) == set(STREETS)
    for street in STREETS:
        ratio = _ratio(street)
        assert ratio == {code: mixes[street].get(code, 1.0) for code in CODES}, street
    assert _ratio("transport_hub")["SMESI"] == 2.0 and _ratio("transport_hub")["KELIMKY"] == 1.8
    assert _ratio("village")["POLEVKY"] == 1.6 and _ratio("high_street")["GOLD"] == 1.6
    assert _ratio("petrol")["SMESI"] == 1.8 and _ratio("shopping_centre")["BEZKOFEIN"] == 1.4


def test_the_tabled_line_leads_its_street_by_a_margin() -> None:
    """s4.yaml `leading_line`: with `line_base`, each street's leading line sells at least 1.3 × any other line."""
    leading = STORIES.story("S4").parameters["leading_line"]
    for street, line in leading.items():
        found = _base([_spot(street)])[0]
        lead = found[CODES.index(line)]
        others = np.delete(found, CODES.index(line))
        assert lead >= 1.3 * others.max(), street


def test_sunday_opening_the_hours_and_the_affluence() -> None:
    plain, sunday, longer, size_l = _base(
        [
            _spot("residential"),
            _spot("residential", sunday_open=True),
            _spot("residential", weekly_hours=140.0),
            _spot("residential", size_class="L"),
        ]
    )
    assert np.allclose(sunday / plain, 1.12)
    assert np.allclose(longer / plain, 2**0.3)
    assert np.allclose(size_l / plain, 5.0 / 2.2)
    poor, rich = _base([_spot("residential"), _spot("residential", okres="CZ0100")])
    premium = [i for i, line in enumerate(LINES) if line.is_premium]
    other = [i for i, line in enumerate(LINES) if not line.is_premium]
    assert np.allclose(rich[premium] / poor[premium], 1.3**0.5)
    assert np.allclose(rich[other], poor[other])


def test_a_point_without_a_store_has_only_the_line_base_and_the_affluence() -> None:
    depot = Spot(2, 2, "vending", ("vending", None), "CZ0201", "SJC", None)  # vending has no stores
    found = _base([depot])[0]
    line_base = STORIES.story("S4").parameters["line_base"]
    assert found.tolist() == [line_base.get(code, 1.0) for code in CODES]


def test_a_point_without_a_store_among_its_channel_s_stores_takes_their_mean_store_factor() -> None:
    """A wholesaler among HoReCa's venues sells as an average venue; a key account's DC keeps its own factor."""
    venues = [
        replace(_spot("residential"), channel="horeca", group=("horeca", None)),
        replace(_spot("residential", size_class="L", sunday_open=True), channel="horeca", group=("horeca", None)),
    ]
    wholesaler = Spot(3, 3, "horeca", ("horeca", None), "CZ0201", "SJC", None)
    dc = Spot(4, 4, "big_retail", ("big_retail", "CH1"), "CZ0201", "SJC", None, "CH1")
    store = replace(_spot("residential"), channel="big_retail", group=("big_retail", "CH1"))
    found = _base([*venues, wholesaler, dc, store])
    alone = _base([Spot(5, 5, "vending", ("vending", None), "CZ0201", "SJC", None)])[0]
    factors = found[:2, 0] / alone[0]  # residential: the mix is 1.0 on every line
    assert np.allclose(found[2], alone * factors.mean())
    assert np.allclose(found[3], alone)


def test_the_null_world_flattens_the_mixes_and_keeps_the_rest() -> None:
    null = replace(STORIES, stories=tuple(replace(s, enabled=False) if s.story == "S4" else s for s in STORIES.stories))
    for street in STREETS:
        assert set(_ratio(street, null).values()) == {1.0}, street
    plain, sunday = _base([_spot("residential"), _spot("residential", sunday_open=True)], null)
    assert np.allclose(sunday / plain, 1.12)


def test_setup_caches_the_base_of_every_delivery_point(opening: Setup) -> None:
    demand = opening.world.demand
    assert demand is not None
    points = opening.rows("delivery_point")
    assert demand.point_ids.tolist() == [row["delivery_point_id"] for row in points]
    assert demand.base.shape == (len(points), len(LINES))
    assert (demand.base >= 0).all() and (demand.base.sum(axis=1) > 0).all()
    assert demand.customer_ids.tolist() == list(opening.world.customers)


def test_the_scale_meets_each_channel_s_plan_at_the_opening(opening: Setup) -> None:
    """Over the lines each customer buys (`carried`): small retail's 3–7 lines meet its plan, not its full listing."""
    demand = opening.world.demand
    assert demand is not None
    world = WorldConfig.load()
    first = world.history.warm_up_from
    quarter = expected_gross(demand, first) * QUARTER_DAYS
    plan = dict(world.pricing.budgets.opening_quarter_gross) | dict(world.demand.quarter_gross)
    customers = list(opening.world.customers.values())
    owner = demand.point_customer
    for channel, gross in plan.items():
        if channel in KEY_ACCOUNTS:
            chains = {c for c, customer in enumerate(customers) if customer.channel == channel}
            for chain in chains:
                assert quarter[owner == chain].sum() == pytest.approx(gross), channel
        else:
            active = [
                p for p, c in enumerate(owner) if customers[c].channel == channel and customers[c].active_on(first)
            ]
            assert quarter[active].mean() == pytest.approx(gross), channel


def test_a_small_retail_point_gets_the_plan_by_its_carried_gross(opening: Setup) -> None:
    """One point's expected gross = the plan × its gross ÷ the channel's mean gross, both over the carried lines at the
    opening; the same point over its full listing sells more than that."""
    demand = opening.world.demand
    assert demand is not None
    world = WorldConfig.load()
    first = world.history.warm_up_from
    customers = list(opening.world.customers.values())
    _, listed, price_line = demand.catalogue.shares(first, len(LINES))
    priced = (listed * price_line)[demand.point_group]
    carried = demand.carried[demand.point_customer]
    gross = (demand.units * priced * carried).sum(axis=1)
    active = [
        p
        for p, c in enumerate(demand.point_customer)
        if customers[c].channel == "small_retail" and customers[c].active_on(first)
    ]
    p = next(p for p in active if not carried[p].all())
    plan = world.pricing.budgets.opening_quarter_gross["small_retail"]
    quarter = expected_gross(demand, first) * QUARTER_DAYS
    assert quarter[p] == pytest.approx(plan * gross[p] / gross[active].mean())
    full = (demand.base[p] * priced[p]).sum() * QUARTER_DAYS
    assert full > quarter[p]


def test_the_gastro_wholesalers_meet_horeca_s_plan(opening: Setup) -> None:
    """The two store-less HoReCa points sell as an average venue (the channel's mean store factor)."""
    demand = opening.world.demand
    assert demand is not None
    world = WorldConfig.load()
    first = world.history.warm_up_from
    wholesalers = [p for p, spot in enumerate(demand.spots) if spot.channel == "horeca" and spot.store is None]
    assert len(wholesalers) == 2
    quarter = expected_gross(demand, first) * QUARTER_DAYS
    plan = world.pricing.budgets.opening_quarter_gross["horeca"]
    for p in wholesalers:
        assert quarter[p] == pytest.approx(plan, rel=0.15)  # ≈ 110 000: its okres and no street mix


def test_a_store_that_changes_gets_its_base_again(opening: Setup) -> None:
    demand = opening.world.demand
    assert demand is not None
    p = next(i for i, spot in enumerate(demand.spots) if spot.store and spot.store.size_class == "S")
    spot = demand.spots[p]
    assert spot.store is not None
    copy = replace(demand, units=demand.units.copy(), spots=demand.spots)
    copy.rebase(
        {p: replace(spot, store=replace(spot.store, size_class="L"))}, {spot.okres: float(demand.point_affluence[p])}
    )
    assert np.allclose(copy.base[p] / demand.base[p], 5.0)
    changed = copy.spots[p].store
    assert changed is not None and changed.size_class == "L"
    assert np.array_equal(np.delete(copy.units, p, axis=0), np.delete(demand.units, p, axis=0))


def test_a_store_less_point_rebased_unchanged_keeps_its_channel_s_mean_store_factor(opening: Setup) -> None:
    """A gastro wholesaler rebased alone keeps the factor the opening gave it from HoReCa's stores (review-207 RR1)."""
    demand = opening.world.demand
    assert demand is not None
    copy = replace(demand, units=demand.units.copy(), spots=demand.spots)
    wholesalers = [p for p, spot in enumerate(demand.spots) if spot.channel == "horeca" and spot.store is None]
    affluence = {demand.spots[p].okres: float(demand.point_affluence[p]) for p in wholesalers}
    copy.rebase({p: demand.spots[p] for p in wholesalers}, affluence)
    assert copy.store_means["horeca"] > 1.0
    assert np.array_equal(copy.units[wholesalers], demand.units[wholesalers])


def test_the_listing_group_lists_what_channel_listing_lists(opening: Setup) -> None:
    demand = opening.world.demand
    assert demand is not None
    goods = demand.catalogue
    _, listed, _ = goods.shares(date(2024, 6, 1), len(LINES))
    vending = goods.groups.index(("vending", None))
    assert {CODES[i] for i in np.flatnonzero(listed[vending])} == {"SMESI", "PROFIKAVA"}
    tesco = goods.groups.index(("big_retail", "TESCO"))
    assert not listed[tesco, CODES.index("PROFIKAVA")] and listed[tesco, CODES.index("GOLD")]
