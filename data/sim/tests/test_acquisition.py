"""Acquisition, churn and breadth (G1 S1.2 T6, `04` S3): each mechanism's monthly chance from the rep assigned on the
day; a year of them applied to the world; what a gained, a lost and a cross-sold customer carry."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig
from hartland_sim.loop.acquisition import LOST, Outcome, Reps, chances, daily, step
from hartland_sim.loop.demand import NEVER, Demand
from hartland_sim.setup.customers import FIRST_ORDER
from hartland_sim.setup.opening import state_rows
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, World

STORIES = StoriesConfig.load()
S3 = STORIES.story("S3").parameters
PINS = {pin.profile: pin for pin in STORIES.pins.reps}
MOVE = S3["area_move_days"][0]


def _copy(opening: Setup) -> tuple[World, Demand, Reps]:
    world = Checkpoint.of(opening.world).restore()
    assert world.demand is not None
    return world, world.demand, Reps.of(world, STORIES)


def _rep(world: World, area: str, day: date) -> int:
    duty = world.duty("rep_area", area, day)
    assert duty is not None
    return duty.employee_id


def _customers_of(reps: Reps, area: str) -> np.ndarray:
    return np.asarray(reps.small[reps.area == reps.areas.index(area)])


def test_each_pinned_rep_s_chances_carry_its_skill(opening: Setup) -> None:
    world, demand, reps = _copy(opening)
    day = date(2023, 6, 1)  # every rep on the area it opened with
    acquire, churn, cross = chances(world, demand, reps, day)
    appeal = dict(zip(reps.small.tolist(), reps.appeal.tolist(), strict=True))
    for profile, (kind, found, base) in {
        "hunter": ("acquisition", acquire, S3["acquisition_base"]),
        "farmer": ("retention", churn, S3["churn_base"]),
        "cross_seller": ("cross_sell", cross, S3["cross_sell_base"]),
    }.items():
        pin = PINS[profile]
        skill = getattr(pin, kind)
        assert world.actors[_rep(world, pin.slot, day)].skills.profile == profile
        customers = [c for c in _customers_of(reps, pin.slot) if found[c] > 0]
        assert len(customers) > 10, profile
        for c in customers:
            if kind == "acquisition":
                assert found[c] == pytest.approx(base * skill * appeal[c])
            elif kind == "retention":
                assert found[c] == pytest.approx(base / skill)
            else:
                assert found[c] == pytest.approx(base * skill)


def test_the_chance_follows_the_rep_on_the_day_through_an_area_move(opening: Setup) -> None:
    world, demand, reps = _copy(opening)
    hunter = PINS["hunter"]
    before, after = MOVE - timedelta(days=1), MOVE
    person = _rep(world, hunter.slot, before)
    assert _rep(world, hunter.slot, after) != person  # the hunter has moved on
    new_area = next(
        duty.target
        for duty in world.duties
        if duty.employee_id == person and duty.kind == "rep_area" and duty.on(after)
    )
    successor = world.actors[_rep(world, hunter.slot, after)].skills.acquisition
    for day, area, skill in ((before, hunter.slot, 1.4), (after, hunter.slot, successor), (after, new_area, 1.4)):
        acquire, _, _ = chances(world, demand, reps, day)
        customers = [c for c in _customers_of(reps, area) if acquire[c] > 0]
        appeal = reps.appeal[np.searchsorted(reps.small, customers)]
        assert np.allclose(acquire[customers], S3["acquisition_base"] * skill * appeal), (day, area)


def test_a_year_of_the_mechanisms_follows_the_skills(opening: Setup) -> None:
    """12 months from 2023-01-01: per rep, the events ÷ what the base rates (skill 1) would give ≈ the skill (≈ 1 ÷
    the skill for churn). Each rep has too few customers for one rep's ratio to settle (a farmer loses ~6 a year),
    so the reps are pooled by skill: the weaker half and the stronger half, each within 15 % of its mean skill (the
    halves hold ~250 events each, a standard error of ~6 %)."""
    world, demand, reps = _copy(opening)
    expected: dict[str, dict[int, float]] = {kind: defaultdict(float) for kind in ("acquire", "churn", "cross")}
    observed: dict[str, dict[int, int]] = {kind: defaultdict(int) for kind in ("acquire", "churn", "cross")}
    rep_of = {int(c): int(a) for c, a in zip(reps.small, reps.area, strict=True)}
    ids = {int(customer): c for c, customer in enumerate(demand.customer_ids)}
    day = date(2023, 1, 1)
    for _ in range(365):
        held, acquisition, retention, cross_sell = reps.skills(world, day)
        people = {a: _rep(world, area, day) for a, area in enumerate(reps.areas) if held[a]}
        found = chances(world, demand, reps, day)
        skill_of = {"acquire": acquisition, "churn": 1 / retention, "cross": cross_sell}
        for kind, chance in zip(("acquire", "churn", "cross"), found, strict=True):
            at_risk = chance[reps.small] > 0
            base = daily(chance[reps.small] / skill_of[kind][reps.area])  # the day's chance at skill 1
            by_area = np.bincount(reps.area[at_risk], weights=base[at_risk], minlength=len(reps.areas))
            for a, person in people.items():
                expected[kind][person] += float(by_area[a])
        outcome = step(world, reps, day)
        for kind, customers in (("acquire", outcome.acquired), ("churn", outcome.lost)):
            for customer in customers:
                observed[kind][people[rep_of[ids[customer]]]] += 1
        for customer, _ in outcome.cross_sold:
            observed["cross"][people[rep_of[ids[customer]]]] += 1
        day += timedelta(days=1)
    for kind, attribute, invert in (
        ("acquire", "acquisition", False),
        ("churn", "retention", True),
        ("cross", "cross_sell", False),
    ):
        skill = {person: getattr(world.actors[person].skills, attribute) for person in expected[kind]}
        ranked = sorted(skill, key=skill.__getitem__)
        half = len(ranked) // 2
        for group in (ranked[:half], ranked[half:]):
            rate = sum(observed[kind][p] for p in group) / sum(expected[kind][p] for p in group)
            weight = np.array([expected[kind][p] for p in group])
            target = float(np.average([1 / skill[p] if invert else skill[p] for p in group], weights=weight))
            assert rate == pytest.approx(target, rel=0.15), (kind, rate, target)
        total = sum(observed[kind].values())
        assert total > 300, kind


def test_a_lost_customer_has_until_and_phase_lost_and_buys_nothing(opening: Setup) -> None:
    world, demand, reps = _copy(opening)
    day = date(2023, 3, 1)
    outcome = Outcome()
    while not outcome.lost:
        outcome = step(world, reps, day)
        day += timedelta(days=1)
    lost_on = day - timedelta(days=1)
    customer = world.customers[outcome.lost[0]]
    assert customer.status == "lost" and customer.until == lost_on - timedelta(days=1)
    assert customer.episodes[-1].phase == "lost" and customer.episodes[-1].valid_from == lost_on
    assert customer.episodes[-1].reason == LOST
    assert customer.phase_on(lost_on) == "lost" and not customer.active_on(lost_on)
    c = list(world.customers).index(customer.customer_id)
    assert demand.customer_until[c] == customer.until.toordinal()
    assert not demand.basket[demand.customer_start[c]].any()
    rows = state_rows(world, opening.config)
    (row,) = [r for r in rows["customer"] if r["customer_id"] == customer.customer_id]
    assert row["status"] == "lost" and row["until"] == customer.until
    phases = [r for r in rows["lifecycle_phase"] if r["customer_id"] == customer.customer_id]
    assert phases[-1]["phase"] == "lost" and phases[-2]["valid_to"] == customer.until


def test_a_gained_customer_orders_today_and_turns_occasional(opening: Setup) -> None:
    world, demand, reps = _copy(opening)
    day = date(2023, 3, 1)
    outcome = step(world, reps, day)
    assert outcome.acquired
    customer = world.customers[outcome.acquired[0]]
    c = list(world.customers).index(customer.customer_id)
    p = demand.customer_start[c]
    assert customer.status == "active" and customer.since == day and customer.active_on(day)
    assert customer.episodes[-1] == type(customer.episodes[-1])("occasional", day, FIRST_ORDER)
    assert demand.customer_since[c] == day.toordinal() and demand.customer_until[c] == NEVER
    assert demand.due[p] and demand.basket[p].sum() == pytest.approx(
        (demand.base[p] * demand.carried[c]).sum() * demand.cycle[p], rel=0.05
    )


def test_a_cross_sale_adds_the_best_line_never_bought(opening: Setup) -> None:
    world, demand, reps = _copy(opening)
    before = demand.carried.copy()
    day = date(2023, 3, 1)
    sold: list[tuple[int, str]] = []
    while not sold:
        sold = list(step(world, reps, day).cross_sold)
        day += timedelta(days=1)
    customer, line = sold[0]
    c = list(world.customers).index(customer)
    i = [line.code for line in demand.lines].index(line)
    assert not before[c, i] and demand.carried[c, i]
    assert (demand.carried[c] & ~before[c]).sum() == 1
    p = demand.customer_start[c]
    never = np.flatnonzero(~before[c] & (demand.units[p] > 0))
    _, listed, _ = demand.catalogue.shares(day, len(demand.lines))
    open_lines = [n for n in never if listed[demand.point_group[p], n]]
    assert i == max(open_lines, key=lambda n: demand.units[p, n])


def test_small_retail_opens_buying_three_to_seven_lines_its_leader_among_them(opening: Setup) -> None:
    demand = opening.world.demand
    assert demand is not None
    customers = list(opening.world.customers.values())
    small = [c for c, customer in enumerate(customers) if customer.channel == "small_retail"]
    counts = demand.carried[small].sum(axis=1)
    assert counts.min() == 3 and counts.max() == 7
    _, listed, _ = demand.catalogue.shares(date(2023, 1, 1), len(demand.lines))
    for c in small[:500]:
        p = demand.customer_start[c]
        leader = int(np.argmax(np.where(listed[demand.point_group[p]], demand.units[p], -1.0)))
        assert demand.carried[c, leader]
    others = [c for c, customer in enumerate(customers) if customer.channel != "small_retail"]
    assert demand.carried[others].all()
