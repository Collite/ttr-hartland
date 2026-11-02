"""Baskets and order triggers (G1 S1.2 T5): the day's latent demand fills each point's basket; a basket orders on its
point's cycle, when it has run the shelf dry, or when due; the lines of an order are the same for the same basket on
the same day; the rounding keeps the volume."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.loop.demand import CHANNELS, DRY_CYCLES, REFERENCE_MONDAY, BasketReady, Demand, Terms, step
from hartland_sim.loop.weather import Climate
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, World

STORIES = StoriesConfig.load()
WORLD = WorldConfig.load()
CLIMATE = Climate.load()
FIRST = date(2023, 3, 1)


def _copy(opening: Setup) -> tuple[World, Demand, Terms]:
    world = Checkpoint.of(opening.world).restore()
    assert world.demand is not None
    return world, world.demand, Terms.of(world.demand, STORIES)


def _temps(demand: Demand, day: date) -> np.ndarray:
    return np.array([CLIMATE.day(region, day).temp_mean_c for region in demand.regions])


def _run(world: World, terms: Terms, first: date, days: int) -> list[BasketReady]:
    assert world.demand is not None
    out = []
    for n in range(days):
        day = first + timedelta(days=n)
        out += step(world, terms, day, _temps(world.demand, day))
    return out


def _small_retail(demand: Demand, day: date, on_cycle: bool) -> int:
    """An active small-retail point whose cycle day `day` is (or is not)."""
    today = day.toordinal()
    for p in range(len(demand.spots)):
        c = demand.point_customer[p]
        if (
            demand.point_channel[p] == CHANNELS.index("small_retail")
            and demand.customer_since[c] <= today <= demand.customer_until[c]
            and ((today - demand.offset[p]) % demand.cycle[p] == 0) == on_cycle
        ):
            return p
    raise AssertionError("no such point")


def test_step_refuses_the_terms_of_another_demand(opening: Setup) -> None:
    """A restored checkpoint holds a new `Demand`: terms built before the restore would index the old one's arrays."""
    world, demand, terms = _copy(opening)
    restored = Checkpoint.of(world).restore()
    with pytest.raises(ValueError, match="another Demand"):
        step(restored, terms, FIRST, _temps(demand, FIRST))


def test_the_day_s_shares_passed_in_give_the_same_orders(opening: Setup) -> None:
    first, _, first_terms = _copy(opening)
    second, demand, second_terms = _copy(opening)
    shares = demand.catalogue.shares(FIRST, len(demand.lines))
    temps = _temps(demand, FIRST)
    assert step(first, first_terms, FIRST, temps) == step(second, second_terms, FIRST, temps, shares)


@pytest.fixture(scope="module")
def ran(opening: Setup) -> tuple[World, Demand, list[BasketReady]]:
    """A copy of the opening world run for 42 days from FIRST."""
    world, demand, terms = _copy(opening)
    return world, demand, _run(world, terms, FIRST, 42)


def test_a_day_s_latent_demand_goes_into_the_basket(opening: Setup) -> None:
    world, demand, terms = _copy(opening)
    p = _small_retail(demand, FIRST, on_cycle=False)
    before = demand.basket[p].copy()
    temps = _temps(demand, FIRST)
    today = terms.today(world, FIRST, temps)
    _, listed, _ = demand.catalogue.shares(FIRST, len(demand.lines))
    buys = listed[demand.point_group[p]] & demand.carried[demand.point_customer[p]]
    expected = demand.base[p] * today.line[p] * buys
    events = step(world, terms, FIRST, temps)
    plain = [i for i in range(len(demand.lines)) if i not in {lift.line for lift in today.skus}]
    assert plain and np.allclose((demand.basket[p] - before)[plain], expected[plain])
    assert (demand.basket[p] - before)[~buys].sum() == 0  # a line it does not buy waits for a cross-sale
    assert demand.point_ids[p] not in {event.delivery_point_id for event in events}


def test_a_point_orders_on_its_cycle_days(ran: tuple[World, Demand, list[BasketReady]]) -> None:
    _, demand, events = ran
    days: dict[int, list[date]] = {}
    for event in events:
        days.setdefault(event.delivery_point_id, []).append(event.day)
    index = {int(point): p for p, point in enumerate(demand.point_ids)}
    for channel, cycle in (("small_retail", 14), ("big_retail", 7), ("horeca", 10)):
        points = [p for p in days if demand.point_channel[index[p]] == CHANNELS.index(channel)]
        assert len(points) > 50, channel
        on_cycle = [
            (day.toordinal() - demand.offset[index[p]]) % demand.cycle[index[p]] == 0 for p in points for day in days[p]
        ]
        assert np.mean(on_cycle) > 0.9, channel  # the rest ran dry or were due
        assert all(demand.cycle[index[p]] == cycle for p in points), channel


def test_the_trade_orders_on_mondays_and_tuesdays_most(opening: Setup) -> None:
    demand = opening.world.demand
    assert demand is not None
    weekly = demand.cycle % 7 == 0
    weekdays = (demand.offset[weekly] - REFERENCE_MONDAY) % 7  # the order days' weekday, Monday 0
    shares = np.bincount(weekdays, minlength=7) / len(weekdays)
    trade = np.array(STORIES.story("S1").parameters["weekday"]["trade"])
    assert np.allclose(shares, trade / trade.sum(), atol=0.015)  # ≈ 5 700 points: 2–3 standard errors


def test_a_shelf_run_dry_orders_before_its_day(opening: Setup) -> None:
    world, demand, terms = _copy(opening)
    p = _small_retail(demand, FIRST, on_cycle=False)
    demand.basket[p] = demand.base[p] * demand.cycle[p] * DRY_CYCLES  # more sold since the last order than shelved
    events = step(world, terms, FIRST, _temps(demand, FIRST))
    assert demand.point_ids[p] in {event.delivery_point_id for event in events}
    assert demand.basket[p].sum() == 0


def test_a_due_point_orders_and_an_empty_basket_never_does(opening: Setup) -> None:
    world, demand, terms = _copy(opening)
    due = _small_retail(demand, FIRST, on_cycle=False)
    demand.due[due] = True
    demand.basket[due] = demand.base[due] * 14
    idle = _small_retail(demand, FIRST, on_cycle=True)
    demand.customer_until[demand.point_customer[idle]] = 0  # lost: no demand, no order
    points = {event.delivery_point_id for event in step(world, terms, FIRST, _temps(demand, FIRST))}
    assert demand.point_ids[due] in points and not demand.due[due]
    assert demand.point_ids[idle] not in points


def test_a_basket_s_lines_are_the_same_for_the_same_customer_and_day(opening: Setup) -> None:
    lines = []
    for alone in (False, True):
        world, demand, terms = _copy(opening)
        p = _small_retail(demand, FIRST, on_cycle=True)
        demand.basket[p] = demand.base[p] * 14
        if alone:  # every other customer gone: the draws of this basket do not depend on theirs
            keep = demand.point_customer[p]
            demand.customer_until[np.arange(len(demand.customer_until)) != keep] = 0
        (event,) = [
            e for e in step(world, terms, FIRST, _temps(demand, FIRST)) if e.delivery_point_id == demand.point_ids[p]
        ]
        lines.append(event.lines)
    assert lines[0] == lines[1] and lines[0]


def test_whole_units_and_the_volume_kept(ran: tuple[World, Demand, list[BasketReady]], opening: Setup) -> None:
    _, demand, events = ran
    units = {sku.sku_id: sku.units_per_case for sku in opening.portfolio.skus}
    for event in events[:5000]:
        for sku, cases, pieces in event.lines:
            if event.channel in ("web", "own_shops"):
                assert pieces == int(pieces) and cases == pytest.approx(pieces / units[sku])
            else:
                assert cases == int(cases) and pieces == cases * units[sku]
    # A second run of the same days orders the same.
    world, again, terms = _copy(opening)
    assert _run(world, terms, FIRST, 42) == events


def test_the_rounding_keeps_the_volume(opening: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    """Baskets set by hand on a day that adds nothing: the orders hold what the baskets held, whole units drawn up
    with the chance of their fraction."""
    world, demand, terms = _copy(opening)
    today = terms.today(world, FIRST, _temps(demand, FIRST))
    monkeypatch.setattr(terms, "today", lambda *_: type(today)(np.zeros_like(today.line)))
    small = np.flatnonzero(demand.point_channel == CHANNELS.index("small_retail"))[:2000]
    demand.customer_since[demand.point_customer[small]] = 0
    _, listed, _ = demand.catalogue.shares(FIRST, len(demand.lines))
    demand.basket[small] = demand.base[small] * listed[demand.point_group[small]] * 20
    demand.due[small] = True
    held = demand.basket[small].sum()
    events = step(world, terms, FIRST, _temps(demand, FIRST))
    ordered = sum(c for e in events for _, c, _ in e.lines)
    assert len(events) > 1900
    assert ordered == pytest.approx(held, rel=0.01)
