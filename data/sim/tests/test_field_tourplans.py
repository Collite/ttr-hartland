"""The tourplans (G2 S2.1 T1, `loop.field`): every store a rep serves — a small-retail customer, a potential one, a
gas key account's station — has one open tourplan for its area's rep, its cycle by its class (a potential customer's
every 8 weeks), its weekday and week drawn once by the store; an area move re-cuts the area's on the move day, the
month-end a store whose customer was won or lost, from the next month's first day."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import is_working_day
from hartland_sim.ids import row_id
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import field
from hartland_sim.loop.pricing import Rules
from hartland_sim.loop.warmup import Output
from hartland_sim.period import Closing, month_end
from hartland_sim.setup.rows import columns
from hartland_sim.setup.run import Setup
from hartland_sim.world import OPEN, Checkpoint, Customer, Tourplan, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
FIRST = WORLD.history.warm_up_from
MOVE = date.fromisoformat(str(STORIES.story("S3").parameters["area_move_days"][0]))
NEVER = date(2100, 1, 1)  # an `Output` that writes nothing


@pytest.fixture
def world(opening: Setup) -> World:
    return Checkpoint.of(opening.world).restore()


def _open(world: World) -> dict[str, Tourplan]:
    return {store: world.tourplans[tourplan_id] for store, tourplan_id in world.tourplan_of.items()}


def test_every_store_a_rep_serves_has_one_open_tourplan(opening: Setup) -> None:
    world = opening.world
    expected = {c.store for c in world.customers.values() if c.channel == "small_retail" and c.status != "lost"}
    stations = {
        place.store for code in ("KA-OMV", "KA-SHELL") for place in opening.customers.places[code] if place.store
    }
    assert len(stations) > 200
    held = _open(world)
    assert set(held) == expected | stations
    assert len(world.tourplans) == len(held)  # the opening's are all open
    assert 7_000 < len(held) < 10_000  # 03 §9: ~7 500
    for store, tourplan in held.items():
        assert (tourplan.store, tourplan.valid_from, tourplan.valid_to) == (store, FIRST, OPEN)
        duty = world.duty("rep_area", tourplan.area, FIRST)
        assert duty is not None and tourplan.employee_id == duty.employee_id
        assert tourplan.employee == world.actors[duty.employee_id].code


def test_a_tourplan_s_cycle_is_its_class_s_a_potential_customer_s_every_8_weeks(opening: Setup) -> None:
    world = opening.world
    stores = {store["osm_id"]: store for store in opening.customers.stores}
    by_store = {c.store: c for c in world.customers.values() if c.channel == "small_retail"}
    seen: Counter[str] = Counter()
    for store, tourplan in _open(world).items():
        row = stores[store]
        assert (tourplan.area, tourplan.store_class) == (row["area"], row["store_class"])
        assert tourplan.store_id == row_id("store", osm_id=store)
        customer = by_store.get(store)
        if customer is None:  # a key account's station
            assert world.customers[tourplan.customer_id].channel == "gas_ka" and tourplan.store_class == "petrol"
            assert tourplan.cycle_weeks == WORLD.field.cycle_weeks["petrol"]
            seen["station"] += 1
            continue
        assert tourplan.customer_id == customer.customer_id
        expected = "potential" if not customer.active_on(FIRST) else row["store_class"]
        assert tourplan.cycle_weeks == WORLD.field.cycle_weeks[expected]
        seen[expected] += 1
    assert seen["potential"] > 3_000 and seen["village"] > 500 and seen["convenience"] > 1_000


def test_the_weekdays_and_weeks_spread_and_follow_the_store(opening: Setup) -> None:
    held = list(_open(opening.world).values())
    weekdays, weeks = Counter(t.weekday for t in held), Counter(t.week for t in held)
    assert set(weekdays) == {1, 2, 3, 4, 5} and set(weeks) == set(range(8))
    assert max(weekdays.values()) < 1.1 * min(weekdays.values())
    assert max(weeks.values()) < 1.2 * min(weeks.values())
    assert all((t.weekday, t.week) == field.slot(t.store) for t in held)


def test_week_of_counts_from_monday() -> None:
    monday = date(2024, 1, 1)
    assert monday.isoweekday() == 1
    sunday = monday - timedelta(days=1)
    assert field.week_of(monday) == field.week_of(monday + timedelta(days=6)) == field.week_of(sunday) + 1


def test_an_area_move_recuts_the_area_s_tourplans_on_the_move_day(world: World) -> None:
    before = {store: (id_, world.tourplans[id_]) for store, id_ in world.tourplan_of.items()}
    moved = {d.target for d in world.duties if d.kind == "rep_area" and d.valid_from == MOVE}
    assert moved
    assert field.recut(world, MOVE, WORLD.field) > 1_000
    reps = 0
    for store, (old_id, old) in before.items():
        new = world.tourplans[world.tourplan_of[store]]
        duty = world.duty("rep_area", old.area, MOVE)
        assert duty is not None and new.employee_id == duty.employee_id
        cycle_weeks = field.cycle(world.customers[old.customer_id], old.store_class, MOVE, WORLD.field)
        if (duty.employee_id, cycle_weeks) == (old.employee_id, old.cycle_weeks):
            assert world.tourplan_of[store] == old_id and old.valid_to == OPEN
            continue
        reps += duty.employee_id != old.employee_id
        assert old.area in moved or duty.employee_id == old.employee_id
        assert old.valid_to == MOVE - timedelta(days=1)
        assert world.tourplan_of[store] == row_id("tourplan", employee=new.employee, store=store, valid_from=MOVE)
        assert (new.valid_from, new.valid_to, new.cycle_weeks) == (MOVE, OPEN, cycle_weeks)
        assert (new.weekday, new.week, new.customer_id) == (old.weekday, old.week, old.customer_id)
    assert reps > 1_000
    assert field.recut(world, MOVE, WORLD.field) == 0  # nothing left to change


def _small_retail(world: World, status: str) -> Customer:
    return next(c for c in world.customers.values() if c.channel == "small_retail" and c.status == status)


def test_the_month_end_recuts_a_store_won_or_lost_from_the_next_month(
    world: World, opening: Setup, tmp_path: Path
) -> None:
    day, first = date(2023, 1, 31), date(2023, 2, 1)
    won, lost = _small_retail(world, "potential"), _small_retail(world, "active")
    won.status, won.since = "active", date(2023, 1, 20)
    lost.status, lost.until = "lost", date(2023, 1, 25)
    assert won.store is not None and lost.store is not None
    was_won, was_lost = world.tourplan_of[won.store], world.tourplan_of[lost.store]
    others = {store: id_ for store, id_ in world.tourplan_of.items() if store not in (won.store, lost.store)}
    rules = Rules.of(opening.world, WORLD, STORIES)
    closing = Closing.of(opening.world, WORLD, STORIES, opening.portfolio.skus, rules)
    with CsvWriter(tmp_path) as writer:
        month_end.close(world, closing, day, Output(writer, NEVER))
    assert world.tourplans[was_won].valid_to == day and world.tourplans[was_won].cycle_weeks == 8
    now = world.tourplans[world.tourplan_of[won.store]]
    assert (now.valid_from, now.cycle_weeks) == (first, WORLD.field.cycle_weeks[now.store_class])
    assert world.tourplans[was_lost].valid_to == day and lost.store not in world.tourplan_of
    joined = {c.store for c in world.customers.values() if not c.active_on(FIRST) and c.active_on(first)}
    assert {store: id_ for store, id_ in world.tourplan_of.items() if store in others and store not in joined} == {
        store: id_ for store, id_ in others.items() if store not in joined
    }


def test_a_tourplan_row_is_its_columns(opening: Setup) -> None:
    tourplan_id, tourplan = next(iter(opening.world.tourplans.items()))
    row = field.tourplan_row(tourplan_id, tourplan)
    assert tuple(row) == columns("tourplan")
    assert row["tourplan_id"] == row_id(
        "tourplan", employee=tourplan.employee, store=tourplan.store, valid_from=tourplan.valid_from
    )
    assert (row["store_id"], row["customer_id"], row["valid_to"]) == (tourplan.store_id, tourplan.customer_id, OPEN)
    assert [r["tourplan_id"] for r in opening.state["tourplan"]] == list(opening.world.tourplans)


# ── the visit plan (T2) ────────────────────────────────────────────────────────

WEEKS = [date(2023, 1, 2) + timedelta(days=k) for k in range(56)]  # eight weeks from a Monday


def test_the_day_s_plan_is_the_tourplans_due_that_weekday_in_their_cycle_s_week(opening: Setup) -> None:
    world = opening.world
    for day in WEEKS:
        planned = field.plan(world, day)
        due = {
            id_
            for id_ in world.tourplan_of.values()
            if (t := world.tourplans[id_]).weekday == day.isoweekday()
            and field.week_of(day) % t.cycle_weeks == t.week % t.cycle_weeks
        }
        if not is_working_day(day):
            assert planned == [] and (day.isoweekday() > 5 or day == date(2023, 1, 1))
            continue
        assert {p.tourplan_id for p in planned} == due
        for p in planned:
            assert p.tourplan is world.tourplans[p.tourplan_id]
            assert p.visit_plan_id == row_id("visit_plan", tourplan=p.tourplan_id, planned_day=day)


def test_every_store_is_planned_once_a_cycle_and_a_rep_has_8_to_12_a_day(opening: Setup) -> None:
    world = opening.world
    seen: Counter[str] = Counter()
    per_rep: Counter[tuple[int, date]] = Counter()
    for day in WEEKS:
        for p in field.plan(world, day):
            seen[p.tourplan.store] += 1
            per_rep[p.tourplan.employee_id, day] += 1
    held = _open(world)
    off = {store for store, t in held.items() if not all(is_working_day(d) for d in _calls(t))}
    assert all(seen[store] == 8 // t.cycle_weeks for store, t in held.items() if store not in off)
    assert len(off) < 0.05 * len(held)  # a call on a day off is not made up
    days = [day for day in WEEKS if is_working_day(day)]
    reps = {t.employee_id for t in held.values()}
    mean = sum(per_rep.values()) / (len(days) * len(reps))
    assert 7 <= mean <= 12  # 03 §9: ~8–12 a rep a day, ~95 k a year


def _calls(tourplan: Tourplan) -> list[date]:
    cycle = tourplan.cycle_weeks
    return [
        d for d in WEEKS if d.isoweekday() == tourplan.weekday and field.week_of(d) % cycle == tourplan.week % cycle
    ]


def test_a_visit_plan_row_is_its_columns(opening: Setup) -> None:
    day = date(2023, 1, 3)
    p = field.plan(opening.world, day)[0]
    row = field.visit_plan_row(p, day)
    assert tuple(row) == columns("visit_plan")
    assert (row["visit_plan_id"], row["tourplan_id"], row["planned_day"]) == (p.visit_plan_id, p.tourplan_id, day)
    assert (row["employee_id"], row["store_id"]) == (p.tourplan.employee_id, p.tourplan.store_id)
