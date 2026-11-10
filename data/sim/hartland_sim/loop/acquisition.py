"""Acquisition, churn and breadth (`04` S3, `05` §3 step 3): the field reps' skills acting on small retail.

Each mechanism is a monthly chance (`s3.yaml`) applied daily as `1 − (1 − p) ^ (1 ÷ 30.44)`, drawn per customer and
day from its stream (`rng.uniforms`, so a customer's draw never depends on the others'), and reads the rep assigned to
the customer's area **on the day** (`World.duty`): a skill follows the person through the two area moves.

- **Acquisition** (stream `acquire`): a potential store becomes a customer with `acquisition_base × skill × appeal`
  a month; `appeal` is the store's base (`Demand.units`, its size, hours and street) as a square root, scaled to a
  mean of 1 over the small-retail stores (`Reps`). An area without a rep that day acquires no one.
- **Churn** (stream `churn`): an active customer is lost with `churn_base ÷ skill` (retention) a month. The planted
  cases' customers (`Customer.traits`) stay.
- **Breadth** (stream `breadth`): an active customer that does not buy every line its listing holds adds the next one,
  by its base, with `cross_sell_base × skill` a month — never the line A-13's store refuses (`demand.refusals`).

Without a rep the churn and breadth act at skill 1. The null world needs no switch here: setup gives every rep skill
1.0 when S3 is off, so the base rates act.

**Run first in the day** (`step` before `demand.step`): a store acquired today places its first order today — its
basket opens with one cycle of its demand and it is `due` — and a customer lost today buys nothing today: its `until`
is yesterday and its phase `lost` from today (in place of the phase yesterday's month-end gave it from today, so one
episode starts a day). The `World`'s customer carries the change (`status`, `since`/`until`, a new phase episode) and
the five state tables render it at the build's end (⚑R201-A); the `Demand` its arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from hartland_sim.config import StoriesConfig
from hartland_sim.loop.demand import BREADTH_CHANNELS, NEVER, Demand, Shares, refusals
from hartland_sim.rng import uniforms
from hartland_sim.setup.customers import FIRST_ORDER
from hartland_sim.world import Episode, World

DAYS_A_MONTH = 365.25 / 12
LOST = "Ukončená spolupráce"
Array = np.ndarray[Any, Any]


@dataclass(frozen=True)
class Rates:
    """S3's monthly base chances (`s3.yaml`)."""

    acquisition: float
    churn: float
    cross_sell: float

    @classmethod
    def of(cls, stories: StoriesConfig) -> Rates:
        parameters = stories.story("S3").parameters
        return cls(parameters["acquisition_base"], parameters["churn_base"], parameters["cross_sell_base"])


@dataclass(frozen=True)
class Outcome:
    """What a day changed: the customers acquired and lost, and the (customer, line) cross-sales."""

    acquired: tuple[int, ...] = ()
    lost: tuple[int, ...] = ()
    cross_sold: tuple[tuple[int, str], ...] = ()


def daily(monthly: Array) -> Array:
    """A monthly chance as a day's."""
    return np.asarray(1 - (1 - np.clip(monthly, 0.0, 1.0)) ** (1 / DAYS_A_MONTH))


@dataclass(eq=False)
class Reps:
    """The small-retail customers as the reps see them, for a run (never pickled): which customers (`small`, by their
    index in the `World`'s order), their areas, whether a planted case pins them, and each store's appeal (`appeal`,
    made again when the `Demand` is rebased)."""

    rates: Rates
    small: Array  # (n,) customer indices
    area: Array  # (n,) index into `areas`
    areas: tuple[str, ...]
    pinned: Array  # (n,) bool
    closed: Array  # (n, L) bool: the lines a customer never takes (A-13's store)
    _appeal: Array = field(repr=False)  # (n,)
    _generation: int = field(repr=False)  # the `Demand.generation` `_appeal` was made at

    @classmethod
    def of(cls, world: World, stories: StoriesConfig) -> Reps:
        demand = world.demand
        if demand is None:
            raise ValueError("the World has no demand state (setup makes it)")
        customers = list(world.customers.values())
        small = np.array(
            [c for c, customer in enumerate(customers) if customer.channel in BREADTH_CHANNELS and customer.area],
            dtype=np.int64,
        )
        codes = [customers[c].area or "" for c in small]
        closed = np.zeros((len(small), len(demand.lines)), dtype=bool)
        refused = refusals(customers, stories, demand.lines)
        for k, c in enumerate(small.tolist()):
            if c in refused:
                closed[k, refused[c]] = True
        areas = tuple(dict.fromkeys(codes))
        return cls(
            Rates.of(stories),
            small,
            np.array([areas.index(code) for code in codes], dtype=np.int64),
            areas,
            np.array([bool(customers[c].traits) for c in small], dtype=bool),
            closed,
            store_appeal(demand, small),
            demand.generation,
        )

    def appeal(self, demand: Demand) -> Array:
        """(n,) each store's appeal (module docstring), made again when `demand` has been rebased since."""
        if self._generation != demand.generation:
            self._appeal, self._generation = store_appeal(demand, self.small), demand.generation
        return self._appeal

    def skills(self, world: World, day: date) -> tuple[Array, Array, Array, Array]:
        """(A,) per area on `day`: whether a rep holds it, and the rep's acquisition, retention and cross-sell."""
        held = np.zeros(len(self.areas), dtype=bool)
        acquisition, retention, cross_sell = (np.ones(len(self.areas)) for _ in range(3))
        for a, area in enumerate(self.areas):
            duty = world.duty("rep_area", area, day)
            if duty is not None:
                skills = world.actors[duty.employee_id].skills
                held[a] = True
                acquisition[a], retention[a], cross_sell[a] = skills.acquisition, skills.retention, skills.cross_sell
        return held, acquisition, retention, cross_sell


def store_appeal(demand: Demand, small: Array) -> Array:
    """(n,) the stores of customers `small` by their base, as a square root scaled to a mean of 1 (module docstring)."""
    size = np.sqrt(demand.units[demand.customer_start[small]].sum(axis=1))
    return np.asarray(size / size.mean())


def chances(
    world: World, demand: Demand, reps: Reps, day: date, listed: Array | None = None
) -> tuple[Array, Array, Array]:
    """(C,) each customer's monthly chance on `day` to be acquired, lost and cross-sold (module docstring); 0 where a
    mechanism does not apply. `listed` is the day's (G, L) listing (`Catalogue.shares`), when the caller has it."""
    today = day.toordinal()
    held, acquisition, retention, cross_sell = reps.skills(world, day)
    area, small = reps.area, reps.small
    since, until = demand.customer_since[small], demand.customer_until[small]
    potential = since == NEVER
    active = (since <= today) & (today <= until)
    if listed is None:
        _, listed, _ = demand.catalogue.shares(day, len(demand.lines))
    open_lines = listed[demand.point_group[demand.customer_start[small]]] & ~demand.carried[small] & ~reps.closed
    room = open_lines.any(axis=1)
    acquire, churn, cross = (np.zeros(len(demand.customer_ids)) for _ in range(3))
    rates = reps.rates
    acquire[small] = np.where(potential & held[area], rates.acquisition * acquisition[area] * reps.appeal(demand), 0.0)
    churn[small] = np.where(active & ~reps.pinned, rates.churn / retention[area], 0.0)
    cross[small] = np.where(active & room, rates.cross_sell * cross_sell[area], 0.0)
    return acquire, churn, cross


def step(world: World, reps: Reps, day: date, shares: Shares | None = None) -> Outcome:
    """The day's acquisitions, losses and cross-sales (module docstring), applied to the `World` and its demand. Only
    the reps' customers are drawn: an id's draw is its own, so leaving the others out changes none. A customer lost
    today is not cross-sold today. `shares` are the day's `Catalogue.shares`, when the caller has them already."""
    demand = world.demand
    if demand is None:
        raise ValueError("the World has no demand state (setup makes it)")
    _, listed, _ = shares or demand.catalogue.shares(day, len(demand.lines))
    acquire, churn, cross = (chance[reps.small] for chance in chances(world, demand, reps, day, listed))
    ids = demand.customer_ids
    small = reps.small
    customers = list(world.customers.values())
    acquired = small[(acquire > 0) & (uniforms("acquire", day, ids=ids[small]) < daily(acquire))]
    lost = small[(churn > 0) & (uniforms("churn", day, ids=ids[small]) < daily(churn))]
    added = small[(cross > 0) & (uniforms("breadth", day, ids=ids[small]) < daily(cross))]
    added = added[~np.isin(added, lost)]
    for c in acquired:
        customer = customers[c]
        customer.status, customer.since = "active", day
        customer.episodes.append(Episode("occasional", day, FIRST_ORDER))
        demand.customer_since[c] = day.toordinal()
        points = slice(int(demand.customer_start[c]), int(demand.customer_start[c + 1]))
        buys = listed[demand.point_group[points]] & demand.carried[c]
        demand.basket[points] = demand.base[points] * buys * demand.cycle[points, None]
        demand.due[points] = True
    for c in lost:
        customer = customers[c]
        customer.status, customer.until = "lost", day - timedelta(days=1)
        if customer.episodes and customer.episodes[-1].valid_from == day:
            customer.episodes.pop()  # yesterday's month-end phased it from today: the loss replaces that
        customer.episodes.append(Episode("lost", day, LOST))
        demand.customer_until[c] = (day - timedelta(days=1)).toordinal()
        points = slice(int(demand.customer_start[c]), int(demand.customer_start[c + 1]))
        demand.basket[points] = 0.0
        demand.due[points] = False
        for sku in list(demand.extras):  # a lost point never orders again to clear its extras
            demand.extras[sku][points] = 0.0
            if not demand.extras[sku].any():
                del demand.extras[sku]
    sold = []
    row = {int(c): k for k, c in enumerate(small)}
    for c in added:
        p = int(demand.customer_start[c])
        open_lines = listed[demand.point_group[p]] & ~demand.carried[c] & ~reps.closed[row[int(c)]]
        line = int(np.argmax(np.where(open_lines, demand.units[p], -1.0)))
        demand.carried[c, line] = True
        sold.append((int(ids[c]), demand.lines[line].code))
    return Outcome(tuple(int(ids[c]) for c in acquired), tuple(int(ids[c]) for c in lost), tuple(sold))
