"""Step 2 of the day loop (`05` §3), the field (G2 S2.1): the reps' tourplans and the day's visit plan.

**Tourplans** (`tourplan`, T1; `World.Tourplan`): a rep calls on every store of its area that is a small-retail
customer or a potential one, and on every station of the gas key accounts — OMV's and Shell's stations are on the
reps' tourplans and merchandised (`01` §4.3, NH-D5). The cycle is `world.yaml field.cycle_weeks` by the store's class,
`potential`'s while its customer is not yet one (a key account's station: `petrol`'s). Each store's weekday (Monday to
Friday) and its week in an 8-week cycle are drawn once, by the store (`slot`, stream `visit`), so its calls keep their
day whoever holds the area. Setup opens one per store for its area's rep on the warm-up's first day
(`setup.org.tourplans`); `recut` closes one the day before a change and opens the next from that day, keeping the
store's slot. It runs on an area move's day (`04` S3's two, the day loop's field step) and at the month-end, from the
next month's first day (`period.month_end.close`), and re-cuts whatever has changed by its day:
- the area's rep changes;
- the customer is won (its class's cycle from then) or lost (no next tourplan) — the month-end's, and a move day's
  for a store won or lost since the last month-end.

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
- a converting visit takes the customer's basket — what it sold since its last order — as its `visit` order: the
  basket that triggered that day, else the basket as it stands (`demand.take`); never a second order. The visit's
  order stands in for the store's next scheduled one (⚑R231-A): the store's order days stay its own, so a call and an
  order day meet only where the store's slot falls on one. Its outcome is `order` when the basket held anything, else
  `no_order`. A key account's station orders through the key account, and an acquisition leaves S1.2's acquisitions
  as they are: neither visit takes an order;
- the rep's day starts at `field.day_starts`; each visit after a travel of `field.travel_minutes`, lasting
  `field.visit_minutes` (Prague time), and a day of more than fit by `field.day_ends` is compressed into it, every
  travel and visit by the same factor (the areas' loads differ, a dense one's calls are quick); A-8's rep leaves
  `unrecorded_share` of its visits unrecorded — no times, no observations, no note.

**The observations** (`visit_observation`, T4; `observe`): a recorded visit that found the shop (`SHUT`: not closed,
not gone) looks at its store class's lines (`field.observed_lines`; every line under `dials.observation_lines: all`),
before it takes any order, every draw from stream `observe` by the visit and the line:
- `facings`: the store's size class's shelf (`field.facings`) shared by the lines' parts of its demand (the base),
  × a draw in [0.7, 1.3], at least one — on a line the customer buys; none on another, nor at a potential store;
- `competitor_facings`: the line's slots × a draw in `field.competitor_facings`, and A-6's Kavalír adds its
  `competitor_facings_lift` beside its line in its region from its sightings to its end; `shelf_share_pct`
  Hartland's share of the two;
- `out_of_stock`: a line it buys whose basket — what the store sold since its last order, the order it placed
  today included (the visit sees the shelf before that delivery) — holds `field.shelf_cycles` of the store's order
  cycles of its demand: the shelf the last order filled has run dry;
- `secondary_placement`: a display of the line lent at the point that day (`World.loans`, G2 S2.3's);
- `promo_compliant`: at a sales visit, where a mandatory promotion of the line runs and admits the store, executed
  with `field.promo_compliance`; else null.

**The competitor sightings** (`competitor_sighting`, T5; `sight`), at most one a recorded visit that found the shop,
every draw from stream `sighting` by the visit: A-6's first — in its region from its sightings to its end, its
`sighting_share` of the visits see Kavalír's promotion of a product of its line, described in a hand-written note
(`PLANTED_SIGHTINGS`, the sighting `planted`) — then the background: `field.sightings.share` of the visits see a
competitor's activity on a product of one of the store's observed lines (`competitor_product.csv`'s equivalent
product), a competitor `home_weight` times as likely in its home region, the kind by `kinds` (no promotion of a
competitor in `promotions_planted`: Kavalír's are A-6's), the price seen × the kind's `price` draw of its typical
price, confirmed at `confirmed`, described in Czech.

**The notes** (`visit.note`, T6; `note`), after the visit's order, every draw from stream `note` by the visit: a
sentence of the visit's outcome (`visit_note_templates.yaml`; an acquisition's or a merchandising visit's `no_order`
reads its kind's, `no_order_by_kind`) and one of each situation it met, as its rows say it — a mandatory promotion of a
line running, executed or not (`promotion_executed`, `promotion_missing`), a line out of stock (`out_of_stock`), a
competitor seen (`competitor_<activity kind>`; A-6's hand-written sighting goes in verbatim), a display lent
(`display`) — at most three, and a second outcome sentence when it met none: 2–4 sentences. A visit that found the
shop closed or gone met none: two outcome sentences. The slots are the visit's: a product on sale that day (a SKU of
it launched and not delisted, `catalogue.csv`) of a line the customer buys (or of the situation's line) by its name or
its nickname (`product.csv`), the competitor's name, the promotion's, an objection. An unrecorded visit has no note,
nor has any visit of A-8's rep, who records nothing; S2.2's prompts add the listing situation and A-13's refusals.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from hartland_sim.config import Field, StoriesConfig, WorldConfig, read_yaml, validate
from hartland_sim.dates import is_working_day
from hartland_sim.ids import row_id
from hartland_sim.io.reference import REFERENCE_DIR, read
from hartland_sim.loop.demand import BasketReady, Shares, take
from hartland_sim.loop.pricing import admits, running
from hartland_sim.rng import Generator, pairs, rng, uniforms
from hartland_sim.setup.rows import Row, row
from hartland_sim.world import OPEN, Customer, Tourplan, World

CYCLE_WEEKS = 8  # a store's week is drawn in the longest cycle, so every shorter one divides it
PRAGUE = ZoneInfo("Europe/Prague")
UNREACHED = ("closed", "refused", "not_found")  # the outcomes of a visit that dealt with no one, in draw order
SHUT = ("closed", "not_found")  # of those, a visit that saw no shelf: the shop closed or gone (a refused one stood in)
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
            delivery_point_id=row_id("delivery_point", customer=customer.code, place=store),
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
class Kavalir:
    """A-6 as the observations read it: the competitor's extra facings beside a line in a region, over days."""

    region: str
    line: int  # the line's index in the demand's lines
    first: date
    last: date
    lift: float
    competitor: str
    share: float  # of the region's recorded visits that see its promotion


@dataclass(frozen=True)
class NoteBook:
    """`visit_note_templates.yaml`: the sentences by outcome, an acquisition's and a merchandising visit's `no_order`
    ones, the sentences by situation, the owners' objections."""

    outcomes: Mapping[str, tuple[str, ...]]
    no_order_by_kind: Mapping[str, tuple[str, ...]]
    situations: Mapping[str, tuple[str, ...]]
    objections: tuple[str, ...]

    @classmethod
    def load(cls, path: Path = REFERENCE_DIR / "visit_note_templates.yaml") -> NoteBook:
        data = read_yaml(path)
        validate(data, "visit_note_templates", path.name)
        return cls(
            {k: tuple(v) for k, v in data["outcomes"].items()},
            {k: tuple(v) for k, v in data["no_order_by_kind"].items()},
            {k: tuple(v) for k, v in data["situations"].items()},
            tuple(data["objections"]),
        )

    def outcome(self, v: Visit) -> tuple[str, ...]:
        """The sentences of `v`'s outcome; an acquisition's or a merchandising visit's `no_order`: its kind's."""
        if v.outcome == "no_order" and v.kind in self.no_order_by_kind:
            return self.no_order_by_kind[v.kind]
        return self.outcomes[v.outcome]


@dataclass(frozen=True)
class Product:
    """A product as a note names it (`product.csv`): its name, its nicknames, and when its SKUs are on sale
    (`catalogue.csv`: from the launch to the day before the delisting)."""

    name: str
    nicknames: tuple[str, ...]
    on_sale: tuple[tuple[date, date | None], ...]  # a SKU's launched_on, delisted_on

    def sells(self, day: date) -> bool:
        return any(launched <= day and (gone is None or day < gone) for launched, gone in self.on_sale)


@dataclass(frozen=True)
class Rival:
    """A competitor's product as the sightings read it (`competitor_product.csv`)."""

    competitor: str
    code: str
    name: str
    typical_price: float


@dataclass(frozen=True)
class Crew:
    """What the visits read beside the `World`, for a run (never pickled; built again from the configs): `field`,
    S3's conversion base, each rep's adherence and its unrecorded share (A-8), each delivery point's index in the
    demand's points, the lines a store class observes (by index) and each line's id, A-6 while enabled."""

    field: Field
    conversion_base: float
    adherence: Mapping[int, float]  # a rep's employee_id → its share of planned visits made
    unrecorded: Mapping[int, float]  # a rep's employee_id → its share of visits left unrecorded
    points: Mapping[int, int]  # delivery_point_id → its index in the demand's points
    observed: Mapping[str, tuple[int, ...]]  # store class → its observed lines' indices
    line_ids: tuple[int, ...]  # product_line_id, by line index
    kavalir: Kavalir | None
    rivals: Mapping[int, tuple[Rival, ...]]  # line index → the competitors' products of it
    notebook: NoteBook
    products: Mapping[int, tuple[Product, ...]]  # line index → its products
    competitors: Mapping[str, str]  # code → name
    sku_line: Mapping[int, int]  # sku_id → its line's index

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
        points = {int(point): p for p, point in enumerate(demand.point_ids)}
        conversion = float(stories.story("S3").parameters["conversion_base"])
        index = {line.code: i for i, line in enumerate(demand.lines)}
        every = tuple(range(len(demand.lines)))
        observed = {
            store_class: every if config.dials.observation_lines == "all" else tuple(index[code] for code in codes)
            for store_class, codes in field.observed_lines.items()
        }
        line_ids = tuple(row_id("product_line", code=line.code) for line in demand.lines)
        a6 = stories.anomaly("A-6")
        kavalir = None
        if a6.enabled:
            held = a6.parameters
            kavalir = Kavalir(
                str(held["region"]),
                index[str(held["line"])],
                date.fromisoformat(str(held["sightings_from"])),
                date.fromisoformat(str(held["until"])),
                float(held["competitor_facings_lift"]),
                str(held["competitor"]),
                float(held["sighting_share"]),
            )
        product_rows = read("product")
        line_of = {product["code"]: product["product_line"] for product in product_rows}
        rivals: dict[int, list[Rival]] = {i: [] for i in range(len(demand.lines))}
        for product in read("competitor_product"):
            rival = Rival(product["competitor"], product["code"], product["name"], float(product["typical_price"]))
            rivals[index[line_of[product["equivalent_product"]]]].append(rival)
        held_rivals = {i: tuple(found) for i, found in rivals.items()}
        on_sale: dict[str, list[tuple[date, date | None]]] = defaultdict(list)
        for sku in read("catalogue"):
            on_sale[sku["product"]].append((sku["launched_on"], sku["delisted_on"]))
        products: dict[int, list[Product]] = {i: [] for i in range(len(demand.lines))}
        for product in product_rows:
            nicknames = tuple(nick for nick in (product["nicknames_cs"] or "").split(";") if nick)
            found = Product(product["name_cs"], nicknames, tuple(on_sale[product["code"]]))
            products[index[product["product_line"]]].append(found)
        return cls(
            field,
            conversion,
            adherence,
            unrecorded,
            points,
            observed,
            line_ids,
            kavalir,
            held_rivals,
            NoteBook.load(),
            {i: tuple(found) for i, found in products.items()},
            {competitor["code"]: competitor["name"] for competitor in read("competitor")},
            {int(sku): int(line) for sku, line in zip(demand.catalogue.sku_ids, demand.catalogue.line, strict=True)},
        )


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
        points = np.array([crew.points[v.tourplan.delivery_point_id] for v in taking.values()], dtype=np.int64)
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


# ── the observations (T4) ──────────────────────────────────────────────────────


def observe(
    world: World, crew: Crew, day: date, visits: Sequence[Visit], events: Sequence[BasketReady] = ()
) -> list[Row]:
    """The `visit_observation` rows of the recorded visits that found the shop (module docstring), before the visits'
    orders; `events` are the day's baskets ordered so far, whose cases the shelves still lack."""
    recorded = [v for v in visits if v.recorded and v.outcome not in SHUT]
    demand = world.demand
    if not recorded or demand is None:
        return []
    ordered: dict[int, Array] = {}
    for event in events:
        point = crew.points.get(event.delivery_point_id)
        if point is not None:
            held = ordered.setdefault(point, np.zeros(len(demand.lines)))
            for sku_id, cases, _ in event.lines:
                held[crew.sku_line[sku_id]] += cases
    mandatory = [promotion for _, promotion in running(world, day) if promotion.mandatory]
    lent = {
        (loan.delivery_point_id, loan.line)
        for loan in world.loans.values()
        if loan.lent_on <= day and (loan.returned_on is None or day < loan.returned_on)
    }
    kavalir = crew.kavalir if crew.kavalir and crew.kavalir.first <= day <= crew.kavalir.last else None
    low, high = crew.field.competitor_facings
    rows = []
    for v in recorded:
        p = crew.points[v.tourplan.delivery_point_id]
        spot, customer = demand.spots[p], world.customers[v.tourplan.customer_id]
        lines = np.array(crew.observed[v.tourplan.store_class], dtype=np.int64)
        base = demand.base[p, lines]
        share = base / base.sum() if base.sum() > 0 else np.full(len(lines), 1 / len(lines))
        size = spot.store.size_class if spot.store else "S"
        slots = crew.field.facings[size] * share
        buys = demand.carried[demand.point_customer[p], lines] & (v.kind != "acquisition")
        keys = pairs(np.full(len(lines), v.visit_id, dtype=np.int64), np.array(crew.line_ids)[lines])
        drawn = slots * (0.7 + 0.6 * uniforms("observe", "facings", ids=keys))
        facings = np.where(buys, np.maximum(1, np.rint(drawn)), 0)
        theirs = slots * (low + (high - low) * uniforms("observe", "competitor", ids=keys))
        if kavalir is not None and spot.region == kavalir.region:
            theirs = np.where(lines == kavalir.line, theirs * (1 + kavalir.lift), theirs)
        competitor = np.rint(theirs)
        sold = demand.basket[p, lines] + (ordered[p][lines] if p in ordered else 0.0)
        dry = buys & (sold >= crew.field.shelf_cycles * demand.cycle[p] * base) & (base > 0)
        executed = uniforms("observe", "promo", ids=keys) < crew.field.promo_compliance
        for k, i in enumerate(lines.tolist()):
            code = demand.lines[i].code
            runs = v.kind == "sales" and any(m.line == code and admits(m, spot, customer, day) for m in mandatory)
            ours, other = int(facings[k]), int(competitor[k])
            rows.append(
                row(
                    "visit_observation",
                    visit_id=v.visit_id,
                    product_line_id=crew.line_ids[i],
                    facings=ours,
                    shelf_share_pct=round(100 * ours / (ours + other), 2) if ours + other else 0.0,
                    out_of_stock=bool(dry[k]),
                    secondary_placement=(v.tourplan.delivery_point_id, code) in lent,
                    promo_compliant=bool(executed[k]) if runs else None,
                    competitor_facings=other,
                    extra=None,
                )
            )
    return rows


# ── the competitor sightings (T5) ──────────────────────────────────────────────

PLANTED_SIGHTINGS = (  # A-6's, hand-written (H-B's dictated sighting)
    "Kavalír má zimní akci na Zlatý – sklenice s dárkem, cena výrazně pod naším Goldem.",
    "Kavalír Zlatý v letáku a na čele regálu, zákazníci se ptají, jestli zlevníme taky.",
    "Nová akce Kavalíru na zlatou kávu, majitel chce vědět, co nabídneme my.",
)
DESCRIBED = {  # the background's description, by activity kind
    "promotion": "Akce: {name} za {price} Kč.",
    "listing": "Nově v regálu: {name}, {price} Kč.",
    "display": "Vystavení u pokladny: {name}, {price} Kč.",
    "price_cut": "Zlevněno: {name} na {price} Kč.",
}


@dataclass(frozen=True)
class Sighting:
    """A competitor's activity seen at a visit (`competitor_sighting`)."""

    visit_id: int
    competitor: str  # its code
    product: str  # the competitor product's code
    kind: str  # promotion · listing · display · price_cut
    price: float
    description: str
    confirmed: bool
    planted: bool = False  # A-6's: its description is the rep's own, verbatim in the note


def _pick(weights: Sequence[float], draw: float) -> int:
    edges = np.cumsum(weights) / sum(weights)
    return min(int(np.searchsorted(edges, draw, side="right")), len(weights) - 1)


def _price(typical: float, span: tuple[float, float], draw: float) -> float:
    """A shelf price: the typical one × the draw within `span`, ending in ,90."""
    return max(0.9, round(round(typical * (span[0] + (span[1] - span[0]) * draw)) - 0.1, 2))


def sight(world: World, crew: Crew, day: date, visits: Sequence[Visit]) -> list[Sighting]:
    """The competitor sightings of the recorded visits that found the shop (module docstring)."""
    recorded = [v for v in visits if v.recorded and v.outcome not in SHUT]
    demand = world.demand
    if not recorded or demand is None:
        return []
    config = crew.field.sightings
    ids = np.array([v.visit_id for v in recorded], dtype=np.int64)
    planted, seen, rival, kind, price, confirmed, note = (
        uniforms("sighting", name, ids=ids).tolist()
        for name in ("planted", "seen", "rival", "kind", "price", "confirmed", "note")
    )
    kavalir = crew.kavalir if crew.kavalir and crew.kavalir.first <= day <= crew.kavalir.last else None
    out = []
    for k, v in enumerate(recorded):
        region = demand.spots[crew.points[v.tourplan.delivery_point_id]].region
        if kavalir is not None and region == kavalir.region and planted[k] < kavalir.share:
            products = [r for r in crew.rivals[kavalir.line] if r.competitor == kavalir.competitor]
            chosen = products[min(int(rival[k] * len(products)), len(products) - 1)]
            description = PLANTED_SIGHTINGS[min(int(note[k] * len(PLANTED_SIGHTINGS)), len(PLANTED_SIGHTINGS) - 1)]
            cost = _price(chosen.typical_price, config.price["promotion"], price[k])
            sure = confirmed[k] < config.confirmed
            out.append(Sighting(v.visit_id, chosen.competitor, chosen.code, "promotion", cost, description, sure, True))
            continue
        if seen[k] >= config.share:
            continue
        candidates = [r for i in crew.observed[v.tourplan.store_class] for r in crew.rivals[i]]
        if not candidates:
            continue
        weights = [config.home_weight if config.home.get(r.competitor) == region else 1.0 for r in candidates]
        chosen = candidates[_pick(weights, rival[k])]
        kinds = [
            name for name in config.kinds if name != "promotion" or chosen.competitor not in config.promotions_planted
        ]
        activity = kinds[_pick([config.kinds[name] for name in kinds], kind[k])]
        cost = _price(chosen.typical_price, config.price[activity], price[k])
        text = DESCRIBED[activity].format(name=chosen.name, price=f"{cost:.2f}".replace(".", ","))
        out.append(
            Sighting(v.visit_id, chosen.competitor, chosen.code, activity, cost, text, confirmed[k] < config.confirmed)
        )
    return out


def sighting_row(s: Sighting) -> Row:
    return row(
        "competitor_sighting",
        competitor_sighting_id=row_id("competitor_sighting", visit=s.visit_id, competitor=s.competitor, seq=1),
        visit_id=s.visit_id,
        competitor_id=row_id("competitor", code=s.competitor),
        competitor_product_id=row_id("competitor_product", code=s.product),
        activity_kind=s.kind,
        price_seen=s.price,
        description=s.description,
        confirmed=s.confirmed,
    )


# ── the notes (T6) ─────────────────────────────────────────────────────────────

SLOT = re.compile(r"\{\{ (\w+) \}\}")
SITUATIONS = ("promotion", "out_of_stock", "competitor", "display")  # in a note's order; `listing` is S2.2's


def _fill(sentence: str, values: Mapping[str, str]) -> str:
    return SLOT.sub(lambda match: values[match.group(1)], sentence)


def note(
    world: World,
    crew: Crew,
    day: date,
    visits: Sequence[Visit],
    observations: Sequence[Row],
    sightings: Sequence[Sighting],
) -> None:
    """Each visit's `note` (module docstring), from its observations and its sighting."""
    demand = world.demand
    if demand is None:
        return
    book = crew.notebook
    rows: dict[int, list[Row]] = defaultdict(list)
    for observation in observations:
        rows[observation["visit_id"]].append(observation)
    seen = {s.visit_id: s for s in sightings}
    lines = {line_id: i for i, line_id in enumerate(crew.line_ids)}
    mandatory = [promotion for _, promotion in running(world, day) if promotion.mandatory]
    for v in visits:
        if not v.recorded or crew.unrecorded.get(v.tourplan.employee_id, 0.0) > 0:
            v.note = None
            continue
        draw = rng("note", v.visit_id)
        p = crew.points[v.tourplan.delivery_point_id]
        spot, customer = demand.spots[p], world.customers[v.tourplan.customer_id]
        bought = np.flatnonzero(demand.carried[demand.point_customer[p]]).tolist()
        pool = bought or list(crew.observed[v.tourplan.store_class])
        met: dict[str, str] = {}
        inside = v.outcome not in SHUT  # a shop closed or gone: no shelf, no competitor
        for observation in rows[v.visit_id] if inside else ():
            line = lines[observation["product_line_id"]]
            code = demand.lines[line].code
            if "promotion" not in met and observation["promo_compliant"] is not None:
                promotion = next(m for m in mandatory if m.line == code and admits(m, spot, customer, day))
                held = {**_slots(crew, [line], day, draw), "promotion": promotion.name or promotion.code}
                said = "promotion_executed" if observation["promo_compliant"] else "promotion_missing"
                met["promotion"] = _fill(_one(book.situations[said], draw), held)
            if "out_of_stock" not in met and observation["out_of_stock"]:
                held = _slots(crew, [line], day, draw)
                met["out_of_stock"] = _fill(_one(book.situations["out_of_stock"], draw), held)
            if "display" not in met and observation["secondary_placement"]:
                met["display"] = _fill(_one(book.situations["display"], draw), _slots(crew, [line], day, draw))
        sighting = seen.get(v.visit_id) if inside else None
        if sighting is not None:
            if sighting.planted:
                met["competitor"] = sighting.description
            else:
                held = {**_slots(crew, pool, day, draw), "competitor": crew.competitors[sighting.competitor]}
                met["competitor"] = _fill(_one(book.situations[f"competitor_{sighting.kind}"], draw), held)
        sentences = [met[name] for name in SITUATIONS if name in met][:3]
        outcome = list(book.outcome(v))
        first = outcome.pop(int(draw.integers(len(outcome))))
        sentences.insert(0, _fill(first, _slots(crew, pool, day, draw)))
        if len(sentences) < 2:
            sentences.append(_fill(_one(outcome, draw), _slots(crew, pool, day, draw)))
        v.note = " ".join(sentences)


def _one(sentences: Sequence[str], draw: Generator) -> str:
    return sentences[int(draw.integers(len(sentences)))]


def _slots(crew: Crew, lines: Sequence[int], day: date, draw: Generator) -> dict[str, str]:
    """A sentence's slots: a product of `lines` on sale on `day` by its name and a nickname, an objection (the caller
    adds the rest)."""
    products = [product for i in lines for product in crew.products[i] if product.sells(day)]
    chosen = products[int(draw.integers(len(products)))]
    return {
        "sku": chosen.name,
        "nickname": _one(chosen.nicknames, draw) if chosen.nicknames else chosen.name,
        "objection": _one(crew.notebook.objections, draw),
        "competitor": "",
        "promotion": "",
    }
