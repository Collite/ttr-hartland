"""S5's opt-in pilot (G2 S2.6; NH-D21-B design08 §§3–4 and §9, NH-D22): customer pairs, a persistent holdout, half
the display budgets and a cost-aware score, with its two audit tables. A build is a pilot only in one of the protocol's
five modes (`World.pilot`); the default build stays the legacy one (`loop.pos`).

**The protocol** (`hartland_sim.protocol.Protocol`, `data/s5_protocol.yaml`) is committed before any pilot-mode build
runs and never edited: the pilot's revisions, dates, budget share, modes, acceptance bars and seeds.

**The PoS-free months** (`pos_free`, January–February 2023): no display is lent and no give-away handed out; each
region's display budget is nil, so no rep is prompted to place one. Every other warm-up mechanism runs as built (the
warm-up hands out no give-aways at all: their rows are the written days').

**The observed history** (`book`, `handed`), in haléře: every small-retail order's Customer Margin — as `sales_pl`
makes it before G3/G4's costs: list price × cases, rounded half up to the haléř, less its discounts — by day, store
(its delivery point) and line, and over the baseline days by customer; every give-away's cost by day and customer.
The month-end reads the last `baseline_days` days of both, then drops what no later month-end reads.

**The assignment** (`assign`, the month-end of the last PoS-free day, dated `assigned_on`): every small-retail customer
registered then — potential, active or lost — with its region, the one of all its delivery points (one that spans
two regions is `outside`, `cross_region`). Within a region its customers sort by their portfolio — the sorted
(store class, street type, size class) of their stores, `portfolio_key` — then their baseline, then customer_id;
consecutive ones form pairs, numbered from 1 (`pair_id`, a hash of the revision, the region and the number), and a last
one left alone is `outside`, `unpaired`. A fair coin per pair (stream `pos`, keyed by `pilot_revision`,
`assignment_revision`, the region and the pair_id) gives its first customer the programme or the holdout, the other the
other arm. Only the `story` mode draws this map: every other mode takes the story pre-pilot's (`Pilot.story_map`) and
keeps its own baselines, so `assignment_digest` — a sha256 of the map without the baselines — is the same in all five.
A customer registered later is in no arm: it receives no pilot PoS while the pilot runs.

**The arms**: a holdout or outside customer gets no display, no give-away (`hands_out`) and no display prompt
(`programme`) through `window_until`; a programme customer only what its manager chooses. A pilot-mode build ends by
then (`loop.build`). The two no-PoS modes lend nothing and hand out nothing at all, but assign.

**The placements** (`place`, every month-end from the last PoS-free day on, dated the next day, `first`): the
candidates are the legacy ones (`loop.pos.place`) of the programme customers — their stores on an open tourplan,
active on `first`, holding no display — each a decision of its month-end's day (`Decision`, `decision_day`). Before
the draw each is forecast, its inputs kept: the store's line and catalogue fit (`World.display_uplift`, the declared
fit, 0.15 in the full-null world); the forecast material, the first display that may carry the line with an asset in
the region's stock at the round's start; the last `baseline_days` days' margin of that display's lines at the store and
the customer's give-aways; `H`, the mean loan term in days, mean months × 365.25 ÷ 12 (`forecast_days` rounds it);
- `forecast_gain = max(0, line margin ÷ days) × fit × H`;
- `forecast_cost = asset cost × H ÷ (pos_asset_life_months × asset_days_per_month) + give-aways ÷ days × H`;
- `score = gain ÷ cost`, each rounded as its column, the score from the rounded two. No material: `no_asset`, score 0;
  a cost not positive and finite: `ineligible`, score 0; no gain: `nonpositive_score`, score 0. A region without a
  manager decides nothing: its candidates are `ineligible`.
The manager — every manager — draws the stores in a random order weighted by `score ^ (pos_skill × choice_exponent)`
(stream `pos`, keyed by `pilot_revision`, the region and `first`, by the point: Efraimidis–Spirakis keys), lends each
the first display that may carry its line with an asset in stock now (`no_asset` when none is left) while the budget
— `pos_budget × budget_factor × placement_budget_share` — covers it, and stops at the first it does not: that one and
the rest are `no_budget`. Loans, their terms and returns, the give-aways' rules and the physical uplift are
`loop.pos`'s, unchanged.

**The effective day** (`revalidate`, NH-D22): a selected placement that `loop.pos.revalidate` cancels keeps its
decision — its key, its month-end's day, its inputs — as `cancelled`, its loan link NULL, before the day's output and
checkpoint.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from fractions import Fraction
from typing import Any

import numpy as np

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import SEPARATOR, business_id, row_id
from hartland_sim.loop import pos
from hartland_sim.loop.pricing import Priced, Rules, halere, milli
from hartland_sim.loop.warmup import Output
from hartland_sim.protocol import PROTOCOL_FILE, Protocol
from hartland_sim.rng import uniforms
from hartland_sim.setup.rows import Row, row
from hartland_sim.world import Customer, Tourplan, World

# ── the state (`World.pilot`) and its rows ──────────────────────────────────

ARMS = ("programme", "holdout", "outside")
OUTCOMES = ("selected", "no_budget", "no_asset", "nonpositive_score", "ineligible", "cancelled")


@dataclass(frozen=True)
class Mapped:
    """A customer's place on the frozen map (the story pre-pilot's, design08 §3): its region, arm, pair and the
    portfolio it was sorted by."""

    region: str
    arm: str  # programme · holdout · outside
    pair_id: int | None
    outside_reason: str | None  # unpaired · cross_region; None in a pair
    portfolio_key: str


@dataclass
class Assigned:
    """A customer's `pos_pilot_assignment` row: its place on the map and its own observed baseline, the whole
    customer's Customer Margin over the baseline days, in haléře."""

    mapped: Mapped
    baseline: int


@dataclass
class Decision:
    """A `pos_placement_decision` row (design08 §§4, 9; NH-D22): the inputs snapshotted before the month's draw — the
    forecast material (None when none was in stock), the 56 days' margin of its lines at the store and the customer's
    give-aways (haléře), the asset's cost, the forecast gain and cost (None without a material), the catalogue fit,
    the score and the forecast's days — then its outcome and the loan it made."""

    customer_id: int
    delivery_point_id: int
    region: str
    manager_id: int | None
    material: str | None
    prior56_line: int
    prior56_handouts: int
    asset_cost: float | None
    gain: float | None
    cost: float | None
    fit: float
    score: float
    forecast_days: int
    outcome: str = "ineligible"
    loan_id: int | None = None


@dataclass
class Pilot:
    """A pilot-mode build's S5 state (module docstring): its mode and protocol, the story pre-pilot's map (the modes
    other than `story` take it), the assignments by customer_id and their digest, the decisions by (decision day,
    delivery_point_id), and the observed history the score reads — the baseline days' margin by customer, and by day
    (its ordinal) the margin of each (store's delivery point, line) and the give-aways of each customer, in haléře."""

    mode: str
    protocol: Protocol
    story_map: dict[int, Mapped] | None = None
    assigned: dict[int, Assigned] = field(default_factory=dict)
    digest: str = ""
    decisions: dict[tuple[date, int], Decision] = field(default_factory=dict)
    baseline: dict[int, int] = field(default_factory=dict)
    lines: dict[int, dict[tuple[int, int], int]] = field(default_factory=dict)
    handouts: dict[int, dict[int, int]] = field(default_factory=dict)


def assignment_row(pilot: Pilot, customer_id: int, held: Assigned) -> Row:
    p, m = pilot.protocol, held.mapped
    return row(
        "pos_pilot_assignment",
        pos_pilot_assignment_id=row_id("pos_pilot_assignment", pilot_revision=p.pilot_revision, customer=customer_id),
        pilot_revision=p.pilot_revision,
        customer_id=customer_id,
        region_id=row_id("region", code=m.region),
        pair_id=m.pair_id,
        arm=m.arm,
        outside_reason=m.outside_reason,
        portfolio_key=m.portfolio_key,
        assigned_on=p.assigned_on,
        baseline_from=p.baseline_from,
        baseline_until=p.pos_free_until,
        window_from=p.window_from,
        window_until=p.window_until,
        baseline_customer_margin=held.baseline / 100,
        baseline_days=p.baseline_days,
        assignment_digest=pilot.digest,
    )


def decision_row(pilot: Pilot, day: date, d: Decision) -> Row:
    p, point = pilot.protocol, d.delivery_point_id
    return row(
        "pos_placement_decision",
        pos_placement_decision_id=row_id(
            "pos_placement_decision", pilot_revision=p.pilot_revision, decision_day=day, delivery_point=point
        ),
        pilot_revision=p.pilot_revision,
        policy_revision=p.policy_revision,
        decision_day=day,
        customer_id=d.customer_id,
        delivery_point_id=d.delivery_point_id,
        region_id=row_id("region", code=d.region),
        manager_id=d.manager_id,
        forecast_material_id=row_id("pos_material", code=d.material) if d.material else None,
        outcome=d.outcome,
        prior56_line_margin=d.prior56_line / 100,
        prior56_handout_cost=d.prior56_handouts / 100,
        asset_cost=d.asset_cost,
        forecast_gain=d.gain,
        forecast_cost=d.cost,
        catalogue_fit=d.fit,
        score=d.score,
        forecast_days=d.forecast_days,
        loan_id=d.loan_id,
    )


def state_rows(pilot: Pilot | None) -> tuple[list[Row], list[Row]]:
    """The two audit tables' rows, rendered from the state (`setup.opening.state_rows`): the assignments in the map's
    order, the decisions by day then the order they were made; none outside a pilot-mode build."""
    if pilot is None:
        return [], []
    assigned = [assignment_row(pilot, customer_id, held) for customer_id, held in pilot.assigned.items()]
    return assigned, [decision_row(pilot, day, d) for (day, _), d in pilot.decisions.items()]


# ── the lifecycle ───────────────────────────────────────────────────────────

NO_POS = ("no-pos", "no-pos-null")  # the reference modes: no display, no give-away (design08 §8)


def start(mode: str, protocol: Protocol | None = None, story_map: dict[int, Mapped] | None = None) -> Pilot:
    """A pilot of `mode` before its first day; `ValueError` for a mode the protocol does not name, or when the story
    mode is given a map or another mode is not (module docstring)."""
    protocol = protocol or Protocol.load()
    if mode not in protocol.modes:
        raise ValueError(f"no pilot mode {mode!r}: the protocol's are {', '.join(protocol.modes)}")
    if (mode == "story") != (story_map is None):
        raise ValueError("the story mode draws the pilot's map; every other mode takes the story pre-pilot's")
    return Pilot(mode, protocol, story_map)


def no_pos(pilot: Pilot) -> bool:
    return pilot.mode in NO_POS


def programme(pilot: Pilot, customer_id: int) -> bool:
    """Whether `customer_id` is in the programme arm (a customer assigned no arm is not)."""
    held = pilot.assigned.get(customer_id)
    return held is not None and held.mapped.arm == "programme"


def hands_out(pilot: Pilot, customer_id: int, day: date) -> bool:
    """Whether a visit at `customer_id`'s store may hand out a give-away on `day` (module docstring)."""
    p = pilot.protocol
    return not no_pos(pilot) and p.assigned_on <= day <= p.window_until and programme(pilot, customer_id)


def margin(line: Priced) -> int:
    """A priced line's Customer Margin in haléře (module docstring): list price × cases rounded half up, less its
    discounts' amounts."""
    gross = (halere(line.list_price_case) * milli(line.cases) + 500) // 1000
    return gross - sum(halere(discount.amount_line) for discount in line.discounts)


def book(
    pilot: Pilot, rules: Rules, customer: Customer, delivery_point_id: int, day: date, priced: Sequence[Priced]
) -> None:
    """An order into the observed history (module docstring): a small-retail customer's only."""
    if customer.channel != "small_retail":
        return
    held = pilot.lines.setdefault(day.toordinal(), {})
    total = 0
    for line in priced:
        amount = margin(line)
        key = (delivery_point_id, rules.lines[rules.sku_line[line.sku_id]])
        held[key] = held.get(key, 0) + amount
        total += amount
    p = pilot.protocol
    if p.baseline_from <= day <= p.pos_free_until:
        pilot.baseline[customer.customer_id] = pilot.baseline.get(customer.customer_id, 0) + total


def handed(pilot: Pilot, customer_id: int, day: date, cost: float) -> None:
    """A give-away's `cost` into the observed history (module docstring)."""
    held = pilot.handouts.setdefault(day.toordinal(), {})
    held[customer_id] = held.get(customer_id, 0) + halere(cost)


def close(world: World, terms: pos.Terms, day: date, output: Output) -> None:
    """The PoS month-end of a pilot-mode build (module docstring): `loop.pos.close`'s steps, the assignment on the
    last PoS-free day, then the pilot's placements — none in the PoS-free months or a no-PoS mode."""
    pilot = world.pilot
    if pilot is None:
        raise ValueError("not a pilot-mode build: its month-end is loop.pos.close")
    p, first = pilot.protocol, day + timedelta(days=1)
    pos.review(world, terms, day, first, output)
    if day == p.pos_free_until:
        assign(world, pilot, terms)
    if first >= p.assigned_on and not pilot.assigned:
        raise ValueError(f"the pilot was never assigned (its day is {p.assigned_on}): build it from the warm-up")
    if first < p.assigned_on or no_pos(pilot):
        world.display_budget = dict.fromkeys(terms.regions, 0.0)
    else:
        place(world, terms, pilot, day, first)
    keep = day.toordinal() - p.baseline_days + 1  # no later month-end reads a day before this one's window
    for history in (pilot.lines, pilot.handouts):
        for old in [d for d in history if d < keep]:
            del history[old]
    if world.measure is not None:
        world.measure.start(pos.month_index(first))


def assign(world: World, pilot: Pilot, terms: pos.Terms) -> None:
    """The assignment (module docstring): the story mode's drawn map, or the story pre-pilot's, with this build's
    baselines; `ValueError` when the map is not of this build's customers."""
    demand = world.demand
    if demand is None:
        raise ValueError("the World has no demand state (setup makes it)")
    universe = [c for c in world.customers.values() if c.channel == "small_retail"]
    mapped = pilot.story_map if pilot.story_map is not None else draw(pilot, universe, demand.spots, terms)
    differ = set(mapped) ^ {c.customer_id for c in universe}
    if differ:
        raise ValueError(f"the pilot's map and this build's small-retail customers differ in {len(differ)}")
    pilot.assigned = {
        c.customer_id: Assigned(mapped[c.customer_id], pilot.baseline.get(c.customer_id, 0)) for c in universe
    }
    pilot.digest = digest(mapped)


def _outlet(spot: Any) -> tuple[str, str, str]:
    store = spot.store
    return (store.store_class, store.street_type, store.size_class) if store is not None else ("-", "-", "-")


def draw(pilot: Pilot, universe: Sequence[Customer], spots: Sequence[Any], terms: pos.Terms) -> dict[int, Mapped]:
    """The story mode's map of `universe` (module docstring), from its stores (`spots`) and its baselines."""
    p = pilot.protocol
    held: dict[int, list[Any]] = {}
    for spot in spots:
        held.setdefault(spot.customer_id, []).append(spot)
    out: dict[int, Mapped] = {}
    sorting: dict[str, list[tuple[tuple[tuple[str, str, str], ...], int, int, str]]] = {}
    for customer in universe:
        mine = held.get(customer.customer_id, [])
        regions = sorted({spot.region for spot in mine})
        if not regions:
            if customer.area is None:
                raise ValueError(f"customer {customer.code} has no delivery point and no area: it has no region")
            regions = [terms.region_of[customer.area]]
        portfolio = tuple(sorted(_outlet(spot) for spot in mine))
        key = ";".join("/".join(outlet) for outlet in portfolio) or "-"
        if len(regions) > 1:
            out[customer.customer_id] = Mapped(regions[0], "outside", None, "cross_region", key)
            continue
        baseline = pilot.baseline.get(customer.customer_id, 0)
        sorting.setdefault(regions[0], []).append((portfolio, baseline, customer.customer_id, key))
    stray = sorted(set(sorting) - set(terms.regions))
    if stray:
        raise ValueError(f"customers in regions the region list lacks: {stray}")
    for region in terms.regions:
        ranked = sorted(sorting.get(region, []))
        pairs = [
            business_id("pos_pilot_pair", SEPARATOR.join((p.pilot_revision, region, str(n))))
            for n in range(1, len(ranked) // 2 + 1)
        ]
        coins = uniforms("pos", p.pilot_revision, p.assignment_revision, region, ids=np.array(pairs, dtype=np.int64))
        for n, pair_id in enumerate(pairs):
            (*_, one, one_key), (*_, other, other_key) = ranked[2 * n], ranked[2 * n + 1]
            arms = ("programme", "holdout") if coins[n] < 0.5 else ("holdout", "programme")
            out[one] = Mapped(region, arms[0], pair_id, None, one_key)
            out[other] = Mapped(region, arms[1], pair_id, None, other_key)
        if len(ranked) % 2:
            *_, last, last_key = ranked[-1]
            out[last] = Mapped(region, "outside", None, "unpaired", last_key)
    return out


def digest(mapped: dict[int, Mapped]) -> str:
    """`assignment_digest`: the map's sha256, by customer_id, without the baselines (module docstring)."""
    out = hashlib.sha256()
    for customer_id in sorted(mapped):
        m = mapped[customer_id]
        pair = "" if m.pair_id is None else str(m.pair_id)
        line = f"{customer_id}\t{m.region}\t{m.arm}\t{pair}\t{m.outside_reason or ''}\t{m.portfolio_key}\n"
        out.update(line.encode())
    return out.hexdigest()


# ── the placements ──────────────────────────────────────────────────────────


def term_days(config: WorldConfig) -> Fraction:
    """`H` (module docstring), exact: the mean of `pos_loans.months`' whole months × 365.25 ÷ 12."""
    low, high = config.field.pos_loans.months
    return Fraction(low + high, 2) * Fraction(36525, 1200)


def place(world: World, terms: pos.Terms, pilot: Pilot, day: date, first: date) -> list[int]:
    """The pilot's placements of `first`, decided on `day` (module docstring); the loans' ids."""
    p = pilot.protocol
    demand = world.demand
    if demand is None:
        raise ValueError("the World has no demand state (setup makes it)")
    stock: dict[tuple[str, str], list[int]] = {}
    for asset_id, asset in world.assets.items():
        if asset.written_off_on is None and asset.loan is None and asset.material in terms.displays:
            stock.setdefault((asset.region, asset.material), []).append(asset_id)
    start = {key: ids[0] for key, ids in stock.items()}  # the round's first-fit asset by (region, material)
    candidates: dict[str, list[Tourplan]] = {}
    for tourplan_id in world.tourplan_of.values():
        held = world.tourplans[tourplan_id]
        customer = world.customers[held.customer_id]
        if (
            customer.channel == "small_retail"
            and programme(pilot, held.customer_id)
            and customer.active_on(first)
            and held.delivery_point_id not in world.lent
        ):
            candidates.setdefault(terms.region_of[held.area], []).append(held)
    history = History.of(pilot, day, {line.code: i for i, line in enumerate(demand.lines)}, terms.config)
    out = []
    for region in terms.regions:
        found = candidates.get(region, [])
        manager = terms.manager(world, region, first)
        decided = [forecast(world, terms, history, t, region, manager, start) for t in found]
        for t, d in zip(found, decided, strict=True):
            pilot.decisions[(day, t.delivery_point_id)] = d
        if manager is None:
            world.display_budget[region] = 0.0
            continue
        employee_id, skill, factor = manager
        left = float(terms.config.field.pos_budget[region]) * factor * p.placement_budget_share
        if found:
            score = np.array([d.score for d in decided])
            ids = np.array([t.delivery_point_id for t in found], dtype=np.int64)
            draw = uniforms("pos", p.pilot_revision, "place", region, first, ids=ids)
            weight = score ** (skill * terms.choice_exponent)
            key = np.where(weight > 0, np.log(np.maximum(draw, 1e-300)) / np.where(weight > 0, weight, 1.0), -np.inf)
            for k in np.argsort(-key, kind="stable").tolist():
                if not np.isfinite(key[k]):
                    break
                t, d = found[k], decided[k]
                line, uplift = next(iter(world.display_uplift[t.delivery_point_id].items()))
                display = next((code for code in pos.fits(terms, line) if stock.get((region, code))), None)
                if display is None:
                    d.outcome = "no_asset"
                    continue
                asset_id = stock[(region, display)][0]
                cost = world.assets[asset_id].cost
                if cost > left:
                    break
                stock[(region, display)].pop(0)
                d.outcome, d.loan_id = "selected", pos.lend(world, terms, t, asset_id, uplift, first, employee_id)
                out.append(d.loan_id)
                left -= cost
        world.display_budget[region] = round(left, 2)
    return out


@dataclass(frozen=True)
class History:
    """The observed history a month-end's forecasts read (module docstring): over its last `baseline_days` days, the
    margin by (delivery point, line) and the give-aways by customer, in haléře; the days, `H`, the asset days."""

    lines: dict[tuple[int, int], int]
    handouts: dict[int, int]
    index: dict[str, int]  # a line's code → its column
    days: int
    term: Fraction
    asset_days: float

    @classmethod
    def of(cls, pilot: Pilot, day: date, index: dict[str, int], config: WorldConfig) -> History:
        p = pilot.protocol
        window = range(day.toordinal() - p.baseline_days + 1, day.toordinal() + 1)
        lines: dict[tuple[int, int], int] = {}
        handouts: dict[int, int] = {}
        for d in window:
            for key, amount in pilot.lines.get(d, {}).items():
                lines[key] = lines.get(key, 0) + amount
            for customer_id, amount in pilot.handouts.get(d, {}).items():
                handouts[customer_id] = handouts.get(customer_id, 0) + amount
        asset_days = config.field.pos_asset_life_months * p.asset_days_per_month
        return cls(lines, handouts, index, p.baseline_days, term_days(config), asset_days)


def forecast(
    world: World,
    terms: pos.Terms,
    history: History,
    t: Tourplan,
    region: str,
    manager: tuple[int, float, float] | None,
    start: dict[tuple[str, str], int],
) -> Decision:
    """A candidate's decision before the draw (module docstring): its inputs, its score and the outcome it has unless
    the draw reaches it — `no_budget` for a positive score."""
    point = t.delivery_point_id
    held = world.display_uplift.get(point, {})
    line, uplift = next(iter(held.items())) if held else (None, 0.0)
    codes = pos.fits(terms, line) if line is not None else []
    material = next((code for code in codes if (region, code) in start), None)
    carried = terms.displays[material or codes[0]] if codes else ((line,) if line is not None else ())
    lines = [history.index[code] for code in carried if code in history.index]
    margin56 = sum(history.lines.get((point, i), 0) for i in lines)
    given = history.handouts.get(t.customer_id, 0)
    h, fit = float(history.term), round(uplift, 8)
    decision = Decision(
        t.customer_id,
        point,
        region,
        manager[0] if manager is not None else None,
        material,
        margin56,
        given,
        None,
        None,
        None,
        fit,
        0.0,
        math.floor(history.term + Fraction(1, 2)),
        "no_asset",
    )
    if material is None:
        return decision
    asset_cost = world.assets[start[(region, material)]].cost
    decision.asset_cost = asset_cost
    decision.gain = round(max(0.0, margin56 / 100 / history.days) * fit * h, 2)
    decision.cost = round(asset_cost * h / history.asset_days + given / 100 / history.days * h, 2)
    if manager is None or not (math.isfinite(decision.cost) and decision.cost > 0):
        decision.outcome = "ineligible"
    elif decision.gain <= 0:
        decision.outcome = "nonpositive_score"
    else:
        decision.score, decision.outcome = round(decision.gain / decision.cost, 8), "no_budget"
    return decision


def revalidate(world: World, day: date) -> list[int]:
    """`loop.pos.revalidate`, and in a pilot-mode build each cancelled placement's decision kept as `cancelled`
    (module docstring, NH-D22); the cancelled loans' ids."""
    pilot = world.pilot
    if pilot is None:
        return pos.revalidate(world, day)
    points = {i: world.loans[i].delivery_point_id for i in world.lent.values() if world.loans[i].lent_on == day}
    cancelled = pos.revalidate(world, day)
    decided = day - timedelta(days=1)
    for loan_id in cancelled:
        held = pilot.decisions.get((decided, points[loan_id]))
        if held is None or held.loan_id != loan_id:
            raise ValueError(f"the cancelled pilot loan {loan_id} has no selected decision of {decided}")
        held.outcome, held.loan_id = "cancelled", None
    return cancelled


# ── the modes ───────────────────────────────────────────────────────────────


def stories_of(mode: str, stories: StoriesConfig) -> StoriesConfig:
    """A mode's stories (design08 §8): the story and its no-PoS reference the stories as they are; the skill-neutral
    control every area manager's `pos_skill` 1.0 (`StoriesConfig.skill_neutral`); the full null and its no-PoS
    reference the null world's (`StoriesConfig.null`)."""
    if mode == "skill-neutral":
        return stories.skill_neutral()
    if mode in ("full-null", "no-pos-null"):
        return stories.null()
    return stories


def metadata(pilot: Pilot, config: WorldConfig) -> tuple[tuple[str, str], ...]:
    """A pilot-mode build's `_sim_meta` keys: its design, mode and revisions, the protocol file's sha256, the window,
    `H` exact, the assignment's digest (`-` before the assignment day), and S5's status, never certified here."""
    p, h = pilot.protocol, term_days(config)
    return (
        ("pilot_mode", pilot.mode),
        ("pilot_design", p.design),
        ("pilot_revision", p.pilot_revision),
        ("policy_revision", p.policy_revision),
        ("pilot_protocol_sha256", hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest()),
        ("pilot_window", f"{p.window_from.isoformat()}/{p.window_until.isoformat()}"),
        ("pilot_term_days", f"{h.numerator}/{h.denominator}"),
        ("assignment_digest", pilot.digest or "-"),
        ("s5_status", "NOT_CERTIFIED"),
    )
