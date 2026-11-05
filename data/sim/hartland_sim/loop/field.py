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
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

from hartland_sim.config import Field
from hartland_sim.dates import is_working_day
from hartland_sim.ids import row_id
from hartland_sim.rng import rng
from hartland_sim.setup.rows import Row, row
from hartland_sim.world import OPEN, Customer, Tourplan, World

CYCLE_WEEKS = 8  # a store's week is drawn in the longest cycle, so every shorter one divides it


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
