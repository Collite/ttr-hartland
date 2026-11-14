"""The point of sale (G2 S2.3; `04` S5, `03` §9, `05` §3 step 2 and the month-end): the displays the area managers lend
to their reps' stores, the regions' asset pools, and what the field step's give-aways read (`field.handouts`).

**A display's uplift** (`Grid`, T1): `display_uplift(store) = display_uplift_base × street_fit[street type] ×
size_fit[size class]` (`s5.yaml`) — a hypermarket's coffee aisle in a shopping centre 0.35, a village shop 0.08; the
null world's is `null_uplift` at every store. `World.display_uplift` holds it for every small-retail store's delivery
point, under the line a display there would carry (S2.2's display prompt reads it), made again at each month-end.

**The displays** (`world.yaml field.pos_displays`): the asset materials a region lends to stores, each with its brand.
A store's display carries the store's best line (by its base, of the lines it buys): the first material in
`pos_displays`' order that brands that line or, brand-wide, its brand, with an asset in the region's stock. A display
of a line carries that line; a brand's display carries the brand's retail lines. The fridges and the vending wraps
stay in the pools: the HoReCa venues and the vending operators are the key accounts', not the area managers'.

**The month-end** (`close`, T2 and T5; `period.month_end.pos_loans`), on a month's last day after its orders, with
everything dated the next day, `first`:
- *the returns*: a loan whose term ends that day, or whose customer is no longer active on `first`, comes back; a
  skilled manager's (`pos_skill` above `skilled_above`) after its first `withdraw_after_months` months too, when its
  measured uplift is below `withdraw_below` — the cases of its lines the store ordered in those months a month over
  those of the two months before it was lent (`Measure`), less one; an unmeasurable one (nothing ordered before)
  stays. A loan comes back in a condition drawn from `pos_loans.condition_on_return` (stream `pos`, by the loan), or
  in the asset's own when that is worse (good < worn < damaged; a `new` asset is `good` once lent): a loan never
  mends its asset. The loan and the asset take it;
- *the ledger* (`ledger`): an asset `pos_asset_life_months` old is written off when it is in stock, and one that
  comes back damaged or past its life on its return; a `new` one is `good` after a year. On the first day of
  `pos_acquisitions.month` each region buys every asset material's pool back to `assets_per_region`
  (`pos_material.csv`), new, numbered on, and an `asset_register` row of the purchase (the PoS assets in aggregate,
  `03` §11: per region and year, on the region's field-sales cost centre, over the assets' life) is written;
- each area's conversion (`World.conversion`): the median over its open tourplans with a sales call of their calls
  converted ÷ their calls, and the share of the area's calls (its open tourplans, each 1 ÷ its cycle) made at the
  stores above it; the next month's give-aways read both. Then the stores' `World.display_uplift` again;
- *the placements* (`place`): each region's manager — its `area_manager_region` duty — lends displays up to the
  region's `pos_budget` × its `budget_factor` in their cost. The candidates are the small-retail stores on an open
  tourplan of the region whose customer is active on `first` and that hold no display; the manager takes them in a
  random order weighted by `display_uplift ^ (pos_skill × choice_exponent)` (stream `pos`, by the region, the month
  and the point: Efraimidis–Spirakis keys) — every manager, the skilled sharply, the weak nearly at random — each
  store's line and uplift read from `World.display_uplift`, made just before; lends each its display, and stops at
  the first whose display costs more than the budget left. A loan runs a whole number of months in
  `pos_loans.months` (stream `pos`, by the loan), from `first`; it is placed by the store's rep (its tourplan's) and
  approved by the manager. What is left of the budget is `World.display_budget`, which S2.2's display prompt reads;

**The effective day** (`revalidate`, G2 S2.5; the day loop's `placements` step): the month-end lends from `first`
before `first`'s acquisitions and losses run. After them, before its demand, a placement of the day whose customer
is not active on it (`Customer.active_on`, since and until both included: one lost that day is not) is cancelled —
no loan, its asset back in the region's stock and its cost back in `World.display_budget`. The other placements keep
their ids and term draws; a loan running at a customer lost later comes back at a month-end, as above. Losses on the
1st cancel; acquisitions on the 1st do not lend: a store acquired that day was potential at the month-end, so it has
no placement to keep, and it is a candidate from the next month-end on.

**The measure** (`Measure`): the orders step books each small-retail order's paid cases by line into its point's
month, the last `MONTHS` months kept: a loan's two months before and its first `withdraw_after_months`, the last of
them the month in progress — so `Terms.of` refuses a `withdraw_after_months` over `MONTHS` − 2.

**The rows**: `pos_asset` and `pos_loan` are state tables, rendered from the `World` when a build ends
(`setup.opening.state_rows`): an asset's condition and write-off, a loan's return move after they are made. The
warm-up's loans are among them, so 2024 starts with displays out (`05` §3.0). A build that ends on a month-end
renders that month-end's placements, pending: their day revalidates them, so a prolong may drop one — a loader
replaces `pos_loan`, never appends to it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import add_months
from hartland_sim.ids import row_id
from hartland_sim.io.reference import REFERENCE_DIR, read
from hartland_sim.loop.demand import Demand, Line, Store, same
from hartland_sim.loop.warmup import Output
from hartland_sim.rng import rng, uniforms
from hartland_sim.setup.rows import Row, row
from hartland_sim.world import Asset, Loan, Tourplan, World

MONTHS = 6  # the measure's months: a loan's two before and its first `withdraw_after_months`, 4 at most
CONDITIONS = ("good", "worn", "damaged")  # a return's, in draw order: each worse than the one before
Array = np.ndarray[Any, Any]


def month_index(day: date) -> int:
    return day.year * 12 + day.month - 1


@dataclass(frozen=True)
class Grid:
    """S5's display uplift (`s5.yaml`, module docstring); `flat` is the null world's."""

    base: float
    street_fit: Mapping[str, float]
    size_fit: Mapping[str, float]
    flat: float | None = None

    @classmethod
    def of(cls, stories: StoriesConfig) -> Grid:
        story = stories.story("S5")
        held = story.parameters
        flat = None if story.enabled else float(held["null_uplift"])
        return cls(float(held["display_uplift_base"]), dict(held["street_fit"]), dict(held["size_fit"]), flat)

    def uplift(self, store: Store) -> float:
        """`display_uplift(store)` (module docstring)."""
        if self.flat is not None:
            return self.flat
        return self.base * self.street_fit[store.street_type] * self.size_fit[store.size_class]


@dataclass(frozen=True)
class Material:
    """A `pos_material` row as the PoS reads it."""

    code: str
    pos_material_id: int
    kind: str  # giveaway · asset
    unit_cost: float
    line: str | None  # the product line it brands; None brand-wide or none
    assets_per_region: int


def materials(root: Path = REFERENCE_DIR) -> tuple[Material, ...]:
    """`pos_material.csv`, in its order."""
    return tuple(
        Material(
            m["code"],
            row_id("pos_material", code=m["code"]),
            m["kind"],
            float(m["unit_cost"]),
            m["product_line"] or None,
            int(m["assets_per_region"]),
        )
        for m in read("pos_material", root)
    )


@dataclass(frozen=True)
class Terms:
    """What the PoS reads beside the `World`, for a run (never pickled; built again from the configs): S5's grid and
    its skilled managers' rules, `world.yaml`'s field.pos_*, the materials, the displays with the lines each carries,
    the regions (in `region.csv`'s order) with their names, field-sales cost centres and areas."""

    grid: Grid
    choice_exponent: float
    skilled_above: float
    withdraw_after_months: int
    withdraw_below: float
    giveaway_cut: float
    config: WorldConfig
    materials: Mapping[str, Material]  # by code, in pos_material.csv's order
    displays: Mapping[str, tuple[str, ...]]  # a display's code → the lines it carries, in pos_displays' order
    brands: Mapping[str, str]  # a retail line's code → its brand
    regions: tuple[str, ...]
    region_names: Mapping[str, str]
    cost_centres: Mapping[str, str]  # region → its field-sales cost centre's code
    region_of: Mapping[str, str]  # area code → its region

    @classmethod
    def of(
        cls, config: WorldConfig, stories: StoriesConfig, line_set: Sequence[Line], root: Path = REFERENCE_DIR
    ) -> Terms:
        s5 = stories.story("S5").parameters
        after = int(s5["withdraw_after_months"])
        if not 1 <= after <= MONTHS - 2:
            raise ValueError(
                f"s5.yaml withdraw_after_months: {after} — the measure keeps {MONTHS} months, a loan's two before"
                f" and at most {MONTHS - 2} after"
            )
        found = {m.code: m for m in materials(root)}
        brands = {line["code"]: line["brand"] for line in read("product_line", root) if not line["is_professional"]}
        retail = [line.code for line in line_set if line.code in brands]
        displays = {}
        for code, brand in config.field.pos_displays.items():
            held = found[code]
            if held.kind != "asset":
                raise ValueError(f"world.yaml field.pos_displays: {code} is not an asset")
            displays[code] = (held.line,) if held.line else tuple(line for line in retail if brands[line] == brand)
        regions = read("region", root)
        centres = {unit["code"].removeprefix("OU-TP-"): unit["cost_centre"] for unit in read("org_unit", root)}
        return cls(
            Grid.of(stories),
            float(s5["choice_exponent"]),
            float(s5["skilled_above"]),
            after,
            float(s5["withdraw_below"]),
            float(s5["skilled_giveaway_cut"]),
            config,
            found,
            displays,
            brands,
            tuple(region["code"] for region in regions),
            {region["code"]: region["name"] for region in regions},
            {region["code"]: centres[region["code"]] for region in regions},
            {area["code"]: area["region"] for area in read("area", root)},
        )

    def manager(self, world: World, region: str, day: date) -> tuple[int, float, float] | None:
        """The region's area manager on `day`: (employee_id, pos_skill, pos_budget_factor); None when unheld."""
        duty = world.duty("area_manager_region", region, day)
        if duty is None:
            return None
        skills = world.actors[duty.employee_id].skills
        return duty.employee_id, skills.pos_skill, skills.pos_budget_factor

    def skilled(self, pos_skill: float) -> bool:
        return pos_skill > self.skilled_above


@dataclass(eq=False)
class Measure:
    """The small-retail points' paid cases by line and month (module docstring), the month's slot its index mod
    `MONTHS`."""

    rows: dict[int, int]  # delivery_point_id → its row
    cases: Array  # (MONTHS, R, L)
    months: Array  # (MONTHS,) the month index a slot holds; −1 none

    def __eq__(self, other: object) -> bool:
        return same(self, other)

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def of(cls, demand: Demand) -> Measure:
        """The opening's: every small-retail delivery point, nothing booked."""
        points = [spot.delivery_point_id for spot in demand.spots if spot.channel == "small_retail"]
        rows = {point: r for r, point in enumerate(points)}
        return cls(rows, np.zeros((MONTHS, len(rows), len(demand.lines))), np.full(MONTHS, -1, dtype=np.int64))

    def start(self, month: int) -> None:
        """`month`'s slot emptied for it."""
        slot = month % MONTHS
        if self.months[slot] != month:
            self.cases[slot] = 0.0
            self.months[slot] = month

    def book(self, delivery_point_id: int, day: date, line: int, cases: float) -> None:
        r = self.rows.get(delivery_point_id)
        if r is None:
            return
        month = month_index(day)
        self.start(month)
        self.cases[month % MONTHS, r, line] += cases

    def held(self, delivery_point_id: int, first: int, months: int, lines: Sequence[int]) -> float | None:
        """The point's cases of `lines` over `months` months from the month index `first`; None when a month is not
        kept."""
        r = self.rows.get(delivery_point_id)
        if r is None:
            return None
        total = 0.0
        for month in range(first, first + months):
            if self.months[month % MONTHS] != month:
                return None
            total += float(self.cases[month % MONTHS, r, list(lines)].sum())
        return total


def best_line(demand: Demand, point: int, retail: Mapping[str, str]) -> int:
    """The point's best retail line: the largest base of the lines its customer buys, of all when it buys none."""
    units = demand.units[point]
    carried = demand.carried[demand.point_customer[point]]
    sold = [i for i, line in enumerate(demand.lines) if line.code in retail]
    if carried[sold].any():
        sold = [i for i in sold if carried[i]]
    return max(sold, key=lambda i: (float(units[i]), -i))


def uplifts(world: World, terms: Terms) -> dict[int, dict[str, float]]:
    """`World.display_uplift` (module docstring): every small-retail store's point, under its best line."""
    demand = world.demand
    if demand is None:
        raise ValueError("the World has no demand state (setup makes it)")
    out = {}
    for p, spot in enumerate(demand.spots):
        if spot.channel == "small_retail" and spot.store is not None:
            line = demand.lines[best_line(demand, p, terms.brands)].code
            out[spot.delivery_point_id] = {line: terms.grid.uplift(spot.store)}
    return out


def fits(terms: Terms, line: str) -> list[str]:
    """The displays that may carry `line`, in `pos_displays`' order: its own, or its brand's."""
    return [code for code, lines in terms.displays.items() if line in lines]


# ── the month-end ───────────────────────────────────────────────────────────


def close(world: World, terms: Terms, day: date, output: Output) -> None:
    """The PoS month-end of `day`, a month's last day (module docstring)."""
    first = day + timedelta(days=1)
    returns(world, terms, day, first)
    ledger(world, terms, first, output)
    medians(world)
    world.display_uplift = uplifts(world, terms)
    place(world, terms, first)
    if world.measure is not None:
        world.measure.start(month_index(first))


def returns(world: World, terms: Terms, day: date, first: date) -> list[int]:
    """The loans that come back on `first` (module docstring), in `World.lent`'s order; their ids."""
    out = []
    for loan_id in list(world.lent.values()):
        loan = world.loans[loan_id]
        done = loan.until <= day or not world.customers[loan.customer_id].active_on(first)
        if not done:
            held = terms.manager(world, loan.region, day)
            run = month_index(day) - month_index(loan.lent_on) + 1
            if held is not None and terms.skilled(held[1]) and run == terms.withdraw_after_months:
                measured = uplift_measured(world, loan, terms.withdraw_after_months)
                done = measured is not None and measured < terms.withdraw_below
        if done:
            give_back(world, terms, loan_id, first)
            out.append(loan_id)
    return out


def uplift_measured(world: World, loan: Loan, months: int) -> float | None:
    """A skilled manager's measure of `loan` over its first `months` (module docstring); None when it cannot be
    measured."""
    demand, measure = world.demand, world.measure
    if demand is None or measure is None:
        return None
    lines = [i for i, line in enumerate(demand.lines) if line.code in loan.lines]
    lent = month_index(loan.lent_on)
    before = measure.held(loan.delivery_point_id, lent - 2, 2, lines)
    after = measure.held(loan.delivery_point_id, lent, months, lines)
    if before is None or after is None or before <= 0:
        return None
    return (after / months) / (before / 2) - 1


def drawn(terms: Terms, loan_id: int) -> str:
    """The condition `loan_id` comes back in by its draw (stream `pos`, `pos_loans.condition_on_return`)."""
    shares = terms.config.field.pos_loans.condition_on_return
    draw, low = float(rng("pos", "return", loan_id).random()), 0.0
    for name in CONDITIONS:
        low += shares[name]
        if draw < low:
            return name
    return CONDITIONS[-1]


def give_back(world: World, terms: Terms, loan_id: int, first: date) -> None:
    """`loan_id` back on `first`, in the worse of its draw and the asset's own condition (module docstring); the asset
    written off when drawn damaged or past its life."""
    loan = world.loans[loan_id]
    asset = world.assets[loan.pos_asset_id]
    draw, own = drawn(terms, loan_id), "good" if asset.condition == "new" else asset.condition
    condition = max(draw, own, key=CONDITIONS.index)
    loan.returned_on, loan.condition_on_return = first, condition
    asset.loan, asset.condition = None, condition
    if draw == "damaged" or add_months(asset.acquired_on, terms.config.field.pos_asset_life_months) <= first:
        asset.written_off_on = first
    del world.lent[loan.delivery_point_id]


def ledger(world: World, terms: Terms, first: date, output: Output) -> list[int]:
    """The write-offs and, in the acquisitions' month, the purchases of `first` (module docstring); the ids bought."""
    life = terms.config.field.pos_asset_life_months
    for asset in world.assets.values():
        if asset.written_off_on is not None or asset.loan is not None:
            continue
        if add_months(asset.acquired_on, life) <= first:
            asset.written_off_on = first
        elif asset.condition == "new" and add_months(asset.acquired_on, 12) <= first:
            asset.condition = "good"
    if first.month != terms.config.field.pos_acquisitions.month:
        return []
    held: dict[tuple[str, str], int] = {}
    for asset in world.assets.values():
        if asset.written_off_on is None:
            held[(asset.region, asset.material)] = held.get((asset.region, asset.material), 0) + 1
    bought, rows = [], []
    for region in terms.regions:
        spent = 0.0
        for material in terms.materials.values():
            if material.kind != "asset":
                continue
            for _ in range(material.assets_per_region - held.get((region, material.code), 0)):
                prefix = f"{material.code}-{region}"
                n = world.counters[prefix] = world.counters.get(prefix, 0) + 1
                asset_no = f"{prefix}-{n:04d}"
                asset_id = row_id("pos_asset", asset_no=asset_no)
                world.assets[asset_id] = Asset(material.code, asset_no, region, first, material.unit_cost, "new")
                bought.append(asset_id)
                spent += material.unit_cost
        if spent:
            name, centre = terms.region_names[region], terms.cost_centres[region]
            rows.append(register_row(region, name, centre, first, spent, terms.config.field.pos_asset_life_months))
    output.append(first, "asset_register", rows)
    return bought


def register_row(region: str, name: str, cost_centre: str, acquired_on: date, cost: float, life: int) -> Row:
    """The `asset_register` row of a region's (`name`: its Czech name) PoS purchase of `acquired_on`'s year, on its
    field-sales `cost_centre`, over `life` months (module docstring)."""
    code = f"MAJ-POS-{region}-{acquired_on.year}"
    return row(
        "asset_register",
        asset_register_id=row_id("asset_register", code=code),
        code=code,
        name=f"Majetek prodejní podpory – {name}, pořízení {acquired_on.year}",
        cost_centre_id=row_id("cost_centre", code=cost_centre),
        acquired_on=acquired_on,
        cost=round(cost, 2),
        useful_life_months=life,
    )


def medians(world: World) -> None:
    """Each area's median conversion and the share of its calls above it (module docstring)."""
    held = [world.tourplans[tourplan_id] for tourplan_id in world.tourplan_of.values()]
    rates: dict[str, list[float]] = {}
    for t in held:
        if t.calls:
            rates.setdefault(t.area, []).append(t.converted / t.calls)
    median = {area: float(np.median(found)) for area, found in rates.items()}
    calls: dict[str, list[float]] = {}  # an area's calls a week: all, then at the stores above its median
    for t in held:
        found = calls.setdefault(t.area, [0.0, 0.0])
        found[0] += 1 / t.cycle_weeks
        if t.calls and t.converted / t.calls > median.get(t.area, 1.0):
            found[1] += 1 / t.cycle_weeks
    world.conversion = {area: (median[area], calls[area][1] / calls[area][0]) for area in sorted(median)}


def place(world: World, terms: Terms, first: date) -> list[int]:
    """Each region's placements of `first` (module docstring), the stores' lines and uplifts from
    `World.display_uplift`; the loans' ids."""
    stock: dict[tuple[str, str], list[int]] = {}
    for asset_id, asset in world.assets.items():
        if asset.written_off_on is None and asset.loan is None and asset.material in terms.displays:
            stock.setdefault((asset.region, asset.material), []).append(asset_id)
    candidates: dict[str, list[Tourplan]] = {}
    for tourplan_id in world.tourplan_of.values():
        held = world.tourplans[tourplan_id]
        customer = world.customers[held.customer_id]
        if (
            customer.channel == "small_retail"
            and customer.active_on(first)
            and held.delivery_point_id not in world.lent
        ):
            candidates.setdefault(terms.region_of[held.area], []).append(held)
    out = []
    for region in terms.regions:
        manager = terms.manager(world, region, first)
        found = candidates.get(region, [])
        if manager is None:
            world.display_budget[region] = 0.0
            continue
        employee_id, skill, factor = manager
        left = float(terms.config.field.pos_budget[region]) * factor
        if found:
            held_by = [world.display_uplift.get(t.delivery_point_id, {}) for t in found]  # {line: uplift}, or none
            uplift = np.array([next(iter(held.values()), 0.0) for held in held_by])
            ids = np.array([t.delivery_point_id for t in found], dtype=np.int64)
            draw = uniforms("pos", "place", region, first, ids=ids)
            weight = uplift ** (skill * terms.choice_exponent)
            key = np.where(weight > 0, np.log(np.maximum(draw, 1e-300)) / np.where(weight > 0, weight, 1.0), -np.inf)
            for k in np.argsort(-key, kind="stable").tolist():
                if not np.isfinite(key[k]):
                    break
                t, line = found[k], next(iter(held_by[k]))
                display = next((code for code in fits(terms, line) if stock.get((region, code))), None)
                if display is None:
                    continue
                asset_id = stock[(region, display)][0]
                asset = world.assets[asset_id]
                if asset.cost > left:
                    break
                stock[(region, display)].pop(0)
                out.append(lend(world, terms, t, asset_id, float(uplift[k]), first, employee_id))
                left -= asset.cost
        world.display_budget[region] = round(left, 2)
    return out


def lend(world: World, terms: Terms, t: Tourplan, asset_id: int, uplift: float, first: date, by: int) -> int:
    """`asset_id` lent to `t`'s store from `first`, carrying its display's lines; the loan's id."""
    asset = world.assets[asset_id]
    loan_id = row_id("pos_loan", pos_asset=asset.asset_no, lent_on=first)
    low, high = terms.config.field.pos_loans.months
    months = int(rng("pos", "term", loan_id).integers(low, high + 1))
    world.loans[loan_id] = Loan(
        pos_asset_id=asset_id,
        customer_id=t.customer_id,
        delivery_point_id=t.delivery_point_id,
        store_id=t.store_id,
        region=asset.region,
        lines=terms.displays[asset.material],
        uplift=uplift,
        lent_on=first,
        until=add_months(first, months) - timedelta(days=1),
        placed_by=t.employee_id,
        approved_by=by,
    )
    world.lent[t.delivery_point_id] = loan_id
    asset.loan = loan_id
    return loan_id


def revalidate(world: World, day: date) -> list[int]:
    """The placements lent from `day` whose customer is not active on it, cancelled after the day's acquisitions and
    losses (module docstring): the loan gone, its asset in stock again, its cost back in the region's display budget;
    their ids, in `World.lent`'s order."""
    out = []
    for point, loan_id in list(world.lent.items()):
        loan = world.loans[loan_id]
        if loan.lent_on != day or world.customers[loan.customer_id].active_on(day):
            continue
        asset = world.assets[loan.pos_asset_id]
        asset.loan = None
        world.display_budget[loan.region] = round(world.display_budget.get(loan.region, 0.0) + asset.cost, 2)
        del world.lent[point], world.loans[loan_id]
        out.append(loan_id)
    return out


# ── the rows ────────────────────────────────────────────────────────────────


def asset_row(asset_id: int, asset: Asset) -> Row:
    return row(
        "pos_asset",
        pos_asset_id=asset_id,
        pos_material_id=row_id("pos_material", code=asset.material),
        asset_no=asset.asset_no,
        acquired_on=asset.acquired_on,
        cost=asset.cost,
        region_id=row_id("region", code=asset.region),
        condition=asset.condition,
        written_off_on=asset.written_off_on,
    )


def loan_row(loan_id: int, loan: Loan) -> Row:
    return row(
        "pos_loan",
        pos_loan_id=loan_id,
        pos_asset_id=loan.pos_asset_id,
        customer_id=loan.customer_id,
        store_id=loan.store_id,
        lent_on=loan.lent_on,
        returned_on=loan.returned_on,
        placed_by_employee_id=loan.placed_by,
        approved_by_employee_id=loan.approved_by,
        condition_on_return=loan.condition_on_return,
    )
