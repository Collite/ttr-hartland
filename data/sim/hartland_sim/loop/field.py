"""Step 2 of the day loop (`05` §3), the field (G2 S2.1): the reps' tourplans and the day's visit plan.

**Tourplans** (`tourplan`, T1; `World.Tourplan`): a rep calls on every store of its area that is a small-retail
customer or a potential one, and on every station of the gas key accounts — OMV's and Shell's stations are on the
reps' tourplans and merchandised (`01` §4.3, NH-D5). The cycle is `world.yaml field.cycle_weeks` by the store's class,
`potential`'s while its customer is not yet one (a key account's station: `petrol`'s). Each store's weekday (Monday to
Friday) and its week in an 8-week cycle are drawn once, by the store (`slot`, stream `visit`), so its calls keep their
day whoever holds the area. Setup opens one per store for its area's rep on the warm-up's first day
(`setup.org.tourplans`); `recut` closes one the day before a change and opens the next from that day, keeping the
store's slot:
- the area's rep changes — `04` S3's two area moves, on their days (the day loop's field step);
- the customer is won (its class's cycle from then) or lost (no next tourplan) — the month-end's, from the next
  month's first day (`period.month_end.close`).

The rows are rendered from the `World` when a build ends (`setup.opening.state_rows`): a tourplan's `valid_to` moves
when it is re-cut, as a duty's does.

**The visit plan** (`visit_plan`, T2; `plan`): on a working day (`dates.is_working_day`) the open tourplans due — the
day's weekday, and a week whose index ≡ the store's week (mod the cycle) — in the tourplans' order; a call that falls
on a day off is not made up. Its id is from the tourplan's and the day.

**The visits** (`visit`, T3; `calls`, `visit`, `take_orders`), every draw from stream `visit` by the plan's, the
tourplan's or the visit's id:
- a planned visit is made with its rep's adherence (`Crew`): `world.yaml field.adherence` ± `adherence_spread`, drawn
  once per rep, and A-7's (`anomalies.yaml`) for the three reps who skip visits; then the ad-hoc ones: each open
  tourplan of a rep not planned that day is visited with the chance that makes `field.ad_hoc_share` of the rep's
  planned visits, with no `visit_plan_id`;
- its kind: `merchandising` at a key account's station, `acquisition` at a store whose customer is not one (yet),
  `sales` at a customer's; its outcome: closed, refused or not found by the kind's shares (`field.outcomes`), else a
  sales visit converts with S3's chance, `conversion_base` × the rep's conversion skill (`04` S3, read on the visit),
  and any other has no order;
- a converting visit takes the customer's basket — what it ordered since its last order — as its `visit` order: the
  basket that triggered that day, else the basket as it stands (`demand.take`); never a second order. Its outcome is
  `order` when the basket held anything, else `no_order`. A key account's station orders through the key account,
  and an acquisition leaves S1.2's acquisitions as they are: neither visit takes an order;
- the rep's day starts at `field.day_starts`; each visit after a travel of `field.travel_minutes`, lasting
  `field.visit_minutes` (Prague time), and a day of more than fit by `field.day_ends` is compressed into it, every
  travel and visit by the same factor (the areas' loads differ, a dense one's calls are quick); A-8's rep leaves
  `unrecorded_share` of its visits unrecorded — no times, no observations, no note.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from hartland_sim.config import Field, StoriesConfig, WorldConfig
from hartland_sim.dates import is_working_day
from hartland_sim.ids import row_id
from hartland_sim.loop.demand import BasketReady, Shares, take
from hartland_sim.rng import rng, uniforms
from hartland_sim.setup.rows import Row, row
from hartland_sim.world import OPEN, Customer, Tourplan, World

CYCLE_WEEKS = 8  # a store's week is drawn in the longest cycle, so every shorter one divides it
PRAGUE = ZoneInfo("Europe/Prague")
UNREACHED = ("closed", "refused", "not_found")  # the outcomes of a visit that dealt with no one, in draw order
Array = np.ndarray[Any, Any]


@dataclass(frozen=True)
class Planned:
    """A planned visit (`visit_plan`): its id, its tourplan's id and the tourplan."""

    visit_plan_id: int
    tourplan_id: int
    tourplan: Tourplan


def week_of(day: date) -> int:
    """The week's index: weeks run Monday to Sunday from 1 January of the year 1, a Monday."""
    return (day.toordinal() - 1) // 7


def slot(store: str) -> tuple[int, int]:
    """A store's weekday (1 Monday … 5 Friday) and its week in an 8-week cycle, drawn by the store (stream
    `visit`)."""
    draw = rng("visit", "tourplan", store)
    return int(draw.integers(1, 6)), int(draw.integers(0, CYCLE_WEEKS))


def cycle(customer: Customer, store_class: str, day: date, field: Field) -> int:
    """The weeks between calls on a store from `day`: its class's, or `potential`'s while its customer is not one."""
    return field.cycle_weeks[store_class if customer.active_on(day) else "potential"]


def open_tourplan(
    world: World, employee_id: int, store: str, customer: Customer, area: str, store_class: str, day: date, field: Field
) -> int:
    """A tourplan of `employee_id` on `store` from `day`, the store's open one; its id."""
    weekday, week = slot(store)
    return _add(
        world,
        Tourplan(
            employee_id=employee_id,
            employee=world.actors[employee_id].code,
            store=store,
            store_id=row_id("store", osm_id=store),
            customer_id=customer.customer_id,
            area=area,
            store_class=store_class,
            cycle_weeks=cycle(customer, store_class, day, field),
            weekday=weekday,
            week=week,
            valid_from=day,
        ),
    )


def _add(world: World, tourplan: Tourplan) -> int:
    tourplan_id = row_id("tourplan", employee=tourplan.employee, store=tourplan.store, valid_from=tourplan.valid_from)
    if tourplan_id in world.tourplans:
        raise ValueError(f"tourplan {tourplan_id} of {tourplan.store} from {tourplan.valid_from} is in the World")
    world.tourplans[tourplan_id] = tourplan
    world.tourplan_of[tourplan.store] = tourplan_id
    return tourplan_id


def _close(world: World, tourplan_id: int, day: date) -> None:
    """The tourplan ends the day before `day`; one that would not have held a day is dropped."""
    held = world.tourplans[tourplan_id]
    if held.valid_from >= day:
        del world.tourplans[tourplan_id]
    else:
        held.valid_to = day - timedelta(days=1)
    if world.tourplan_of.get(held.store) == tourplan_id:
        del world.tourplan_of[held.store]


def recut(world: World, day: date, field: Field) -> int:
    """The open tourplans whose rep, cycle or customer changes on `day` closed the day before and, unless the customer
    is lost, the next opened from `day` (module docstring); how many changed. An area nobody holds keeps its own."""
    changed = 0
    for tourplan_id in list(world.tourplan_of.values()):
        held = world.tourplans[tourplan_id]
        customer = world.customers[held.customer_id]
        if customer.status == "lost":
            _close(world, tourplan_id, day)
            changed += 1
            continue
        duty = world.duty("rep_area", held.area, day)
        cycle_weeks = cycle(customer, held.store_class, day, field)
        if duty is None or (duty.employee_id, cycle_weeks) == (held.employee_id, held.cycle_weeks):
            continue
        _close(world, tourplan_id, day)
        employee = world.actors[duty.employee_id].code
        _add(
            world,
            replace(
                held,
                employee_id=duty.employee_id,
                employee=employee,
                cycle_weeks=cycle_weeks,
                valid_from=day,
                valid_to=OPEN,
            ),
        )
        changed += 1
    return changed


def tourplan_row(tourplan_id: int, tourplan: Tourplan) -> Row:
    return row(
        "tourplan",
        tourplan_id=tourplan_id,
        employee_id=tourplan.employee_id,
        store_id=tourplan.store_id,
        customer_id=tourplan.customer_id,
        cycle_weeks=tourplan.cycle_weeks,
        weekday=tourplan.weekday,
        valid_from=tourplan.valid_from,
        valid_to=tourplan.valid_to,
    )


def plan(world: World, day: date) -> list[Planned]:
    """The day's planned visits (module docstring)."""
    if not is_working_day(day):
        return []
    weekday, week = day.isoweekday(), week_of(day)
    out = []
    for tourplan_id in world.tourplan_of.values():
        held = world.tourplans[tourplan_id]
        due = held.weekday == weekday and week % held.cycle_weeks == held.week % held.cycle_weeks
        if due and held.valid_from <= day:
            visit_plan_id = row_id("visit_plan", tourplan=tourplan_id, planned_day=day)
            out.append(Planned(visit_plan_id, tourplan_id, held))
    return out


def visit_plan_row(planned: Planned, day: date) -> Row:
    return row(
        "visit_plan",
        visit_plan_id=planned.visit_plan_id,
        tourplan_id=planned.tourplan_id,
        planned_day=day,
        employee_id=planned.tourplan.employee_id,
        store_id=planned.tourplan.store_id,
    )


# ── the visits (T3) ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Crew:
    """What the visits read beside the `World`, for a run (never pickled; built again from the configs): `field`,
    S3's conversion base, each rep's adherence and its unrecorded share (A-8), and each customer's first point (its
    index in the demand's points: a small-retail customer's only one)."""

    field: Field
    conversion_base: float
    adherence: Mapping[int, float]  # a rep's employee_id → its share of planned visits made
    unrecorded: Mapping[int, float]  # a rep's employee_id → its share of visits left unrecorded
    points: Mapping[int, int]  # customer_id → its first point's index

    @classmethod
    def of(cls, world: World, config: WorldConfig, stories: StoriesConfig) -> Crew:
        field, skipped = config.field, float(stories.anomaly("A-7").parameters["adherence"])
        silent = float(stories.anomaly("A-8").parameters["unrecorded_share"])
        adherence, unrecorded = {}, {}
        for actor in world.actors.values():
            if actor.role != "rep":
                continue
            drawn = field.adherence + field.adherence_spread * (2 * rng("visit", "adherence", actor.code).random() - 1)
            adherence[actor.employee_id] = skipped if actor.skills.skips_visits else drawn
            unrecorded[actor.employee_id] = silent if actor.skills.records_nothing else 0.0
        demand = world.demand
        if demand is None:
            raise ValueError("the World has no demand state (setup makes it)")
        points = {int(customer): int(demand.customer_start[c]) for c, customer in enumerate(demand.customer_ids)}
        conversion = float(stories.story("S3").parameters["conversion_base"])
        return cls(field, conversion, adherence, unrecorded, points)


@dataclass
class Visit:
    """A visit made (`visit`): its planned visit (None ad hoc), its tourplan, its kind and outcome, whether its
    conversion fired (a sales visit that takes the basket), whether recorded, its times (Prague; None unrecorded)."""

    visit_id: int
    visit_plan_id: int | None
    tourplan: Tourplan
    kind: str  # sales · merchandising · acquisition
    outcome: str  # order · no_order · closed · refused · not_found
    converts: bool
    recorded: bool
    time_in: datetime | None
    time_out: datetime | None
    note: str | None = None


def calls(world: World, crew: Crew, day: date, planned: Sequence[Planned]) -> list[tuple[int | None, Tourplan]]:
    """The day's visits to make: the planned ones made, then the ad-hoc ones (module docstring), each with its
    `visit_plan_id` (None ad hoc) and its tourplan."""
    if not planned:
        return []
    made = uniforms("visit", "made", ids=np.array([p.visit_plan_id for p in planned], dtype=np.int64))
    out: list[tuple[int | None, Tourplan]] = [
        (p.visit_plan_id, p.tourplan)
        for p, draw in zip(planned, made.tolist(), strict=True)
        if draw < crew.adherence.get(p.tourplan.employee_id, crew.field.adherence)
    ]
    per_rep = Counter(p.tourplan.employee_id for p in planned)
    due = {p.tourplan_id for p in planned}
    others = [
        (tourplan_id, world.tourplans[tourplan_id])
        for tourplan_id in world.tourplan_of.values()
        if tourplan_id not in due and world.tourplans[tourplan_id].valid_from <= day
    ]
    if not others:
        return out
    held = Counter(t.employee_id for _, t in others)
    draws = uniforms("visit", "ad_hoc", day, ids=np.array([tourplan_id for tourplan_id, _ in others], dtype=np.int64))
    for (_, t), draw in zip(others, draws.tolist(), strict=True):
        if draw < crew.field.ad_hoc_share * per_rep[t.employee_id] / held[t.employee_id]:
            out.append((None, t))
    return out


def kind_of(customer: Customer, day: date) -> str:
    if customer.channel == "gas_ka":
        return "merchandising"
    return "sales" if customer.active_on(day) else "acquisition"


def _minutes(span: tuple[int, int], draw: float) -> int:
    return span[0] + min(int(draw * (span[1] - span[0] + 1)), span[1] - span[0])


def visit(world: World, crew: Crew, day: date, made: Sequence[tuple[int | None, Tourplan]]) -> list[Visit]:
    """The visits of `made` (`calls`), their kinds, outcomes, conversions and times (module docstring); the outcome
    `order` is `take_orders`'."""
    if not made:
        return []
    ids = [row_id("visit", employee=t.employee, store=t.store, day=day) for _, t in made]
    keys = np.array(ids, dtype=np.int64)
    reach, convert, record = (uniforms("visit", name, ids=keys).tolist() for name in ("outcome", "convert", "recorded"))
    travel, length = (uniforms("visit", name, ids=keys).tolist() for name in ("travel", "minutes"))
    start = datetime.combine(day, time.fromisoformat(crew.field.day_starts), PRAGUE)
    minutes = [
        (_minutes(crew.field.travel_minutes, travel[k]), _minutes(crew.field.visit_minutes, length[k]))
        for k in range(len(made))
    ]
    needed: Counter[int] = Counter()
    for k, (_, t) in enumerate(made):
        needed[t.employee_id] += sum(minutes[k])
    day_length = (datetime.combine(day, time.fromisoformat(crew.field.day_ends), PRAGUE) - start).total_seconds() / 60
    clock: dict[int, float] = {}  # a rep's minutes since the day's start
    out = []
    for k, (visit_plan_id, t) in enumerate(made):
        kind = kind_of(world.customers[t.customer_id], day)
        outcome, low = "no_order", 0.0
        for name in UNREACHED:
            share = crew.field.outcomes[kind][name]
            if reach[k] < low + share:
                outcome = name
                break
            low += share
        skill = world.actors[t.employee_id].skills.conversion
        converts = outcome == "no_order" and kind == "sales" and convert[k] < crew.conversion_base * skill
        scale = min(1.0, day_length / needed[t.employee_id])
        arrived = clock.get(t.employee_id, 0.0) + minutes[k][0] * scale
        left = clock[t.employee_id] = arrived + minutes[k][1] * scale
        recorded = record[k] >= crew.unrecorded.get(t.employee_id, 0.0)
        time_in = start + timedelta(minutes=round(arrived)) if recorded else None
        time_out = start + timedelta(minutes=round(left)) if recorded else None
        out.append(Visit(ids[k], visit_plan_id, t, kind, outcome, converts, recorded, time_in, time_out))
    return out


def take_orders(
    world: World, crew: Crew, day: date, visits: Sequence[Visit], events: Sequence[BasketReady], shares: Shares
) -> list[BasketReady]:
    """The day's baskets with the converting visits' orders (module docstring): a basket that triggered today taken
    by its customer's visit, the others' baskets taken as they stand, after the day's."""
    out = list(events)
    first: dict[int, int] = {}
    for k, event in enumerate(out):
        first.setdefault(event.customer_id, k)
    taking: dict[int, Visit] = {}
    for v in visits:
        if not v.converts:
            continue
        customer = v.tourplan.customer_id
        at = first.get(customer)
        if at is None:
            taking.setdefault(customer, v)
        elif out[at].visit is None:
            out[at] = replace(out[at], visit=(v.visit_id, v.tourplan.employee_id))
            v.outcome = "order"
    if taking:
        points = np.array([crew.points[customer] for customer in taking], dtype=np.int64)
        for event in take(world, day, points, shares):
            v = taking[event.customer_id]
            out.append(replace(event, visit=(v.visit_id, v.tourplan.employee_id)))
            v.outcome = "order"
    return out


def visit_row(v: Visit, day: date) -> Row:
    return row(
        "visit",
        visit_id=v.visit_id,
        visit_plan_id=v.visit_plan_id,
        employee_id=v.tourplan.employee_id,
        store_id=v.tourplan.store_id,
        customer_id=v.tourplan.customer_id,
        day=day,
        time_in=v.time_in,
        time_out=v.time_out,
        kind=v.kind,
        outcome=v.outcome,
        note=v.note,
        recorded=v.recorded,
    )
