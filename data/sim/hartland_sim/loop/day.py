"""The day loop (`05` §3, G1 S1.4 T2): one simulated day, its steps in their order, then the period ends; a run moves
the `World` a day at a time from the warm-up's first day, its rows written through the warm-up switch (`Output`: none
before `history.first_day`).

**The steps** (`STEPS`), each a method of `Loop` taking the day:
- `weather`: the regions' temperatures of the day (S1's weather term) and its `weather_daily` rows;
- 3 `acquisition` — the acquisitions, losses and cross-sales — first, as S1.2 built it: a store acquired today
  orders today, and one lost today buys nothing;
- 1 `demand`: the day's latent demand into the baskets, and the baskets that order;
- 2 `field`: the tourplans, visits and prompts — G2's, a stub here;
- 4 `orders`: the customers' asks, then the baskets' orders, priced (`loop.orders`);
- 5 `supply`, 6 `delivery` (G3), 7 `finance`, 8 `shops` (G4), 8b `documents` (DOC): stubs, called and returning
  nothing;
- the period ends (`periods`), on the last calendar day of each: a quarter's runs `quarter_end.roll_lists` — M+1's
  channel lists need Q+1's standard list — then `month_end.close`, then `quarter_end.roll_budgets`, which reads the
  month-end's plan of M+1…M+3 (= Q+1); 31 December then `year_end.close`; any other month's last day
  `month_end.close`;
- 9 `checkpoint`: `World.day` and `last_day_done` move to the day. The checkpoint row itself is written when the
  build ends (`cli build-world`); the warm-up's last day keeps one (`warm_up_end`), from which a null-world or
  re-parameterised build can start.

`Loop` holds what a run reads beside the `World` — the demand's and the reps' terms, the order desk, the period ends'
terms, the climate — built again from the configs for a run (never pickled), and what the run did (`Report`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import month_end, quarter_end
from hartland_sim.loop import acquisition, demand, orders
from hartland_sim.loop.pricing import Rules
from hartland_sim.loop.warmup import Output
from hartland_sim.loop.weather import Climate
from hartland_sim.period import Closing
from hartland_sim.period import month_end as month_end_hook
from hartland_sim.period import quarter_end as quarter_end_hook
from hartland_sim.period import year_end as year_end_hook
from hartland_sim.setup.portfolio import Sku
from hartland_sim.world import Checkpoint, World

STEPS = (
    "weather",
    "acquisition",
    "demand",
    "field",
    "orders",
    "supply",
    "delivery",
    "finance",
    "shops",
    "documents",
    "periods",
    "checkpoint",
)
Array = np.ndarray[Any, Any]


def quarter(day: date) -> str:
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


@dataclass
class Report:
    """What a run did, by quarter (`2024-Q1`): its orders and lines, the asks granted and refused by reason, the orders
    held by reason."""

    quarters: dict[str, Counter[str]] = field(default_factory=dict)

    def add(self, day: date, done: orders.Day) -> None:
        held = self.quarters.setdefault(quarter(day), Counter())
        held["orders"] += done.orders
        held["lines"] += done.lines
        held["granted"] += done.granted
        for reason, n in done.refused.items():
            held[f"refused:{reason}"] += n
        for reason, n in done.held.items():
            held[f"held:{reason}"] += n


@dataclass
class Today:
    """What a day's steps hand on to the next ones."""

    temps: Array
    shares: demand.Shares
    events: list[demand.BasketReady] = field(default_factory=list)


@dataclass
class Loop:
    """A run's day loop (module docstring)."""

    world: World
    config: WorldConfig
    output: Output
    terms: demand.Terms
    reps: acquisition.Reps
    desk: orders.Desk
    closing: Closing
    climate: Climate
    last_day_done: date
    report: Report = field(default_factory=Report)
    warm_up_end: Checkpoint | None = None
    _today: Today | None = field(default=None, repr=False)

    @classmethod
    def of(cls, world: World, config: WorldConfig, stories: StoriesConfig, skus: Sequence[Sku], output: Output) -> Loop:
        state = world.demand
        if state is None:
            raise ValueError("the World has no demand state (setup makes it)")
        rules = Rules.of(world, config, stories)
        return cls(
            world,
            config,
            output,
            demand.Terms.of(state, stories),
            acquisition.Reps.of(world, stories),
            orders.Desk.of(world, rules, config),
            Closing.of(world, config, stories, skus, rules),
            Climate.load(),
            world.day,
        )

    def run(self, day: date) -> None:
        """One day (module docstring): the day after the `World`'s."""
        if day != self.world.day + timedelta(days=1):
            raise ValueError(f"the World is after {self.world.day}: the next day is not {day}")
        for step in STEPS:
            getattr(self, step)(day)

    def until(self, last: date) -> None:
        """Every day from the `World`'s next to `last`, both included."""
        day = self.world.day + timedelta(days=1)
        while day <= last:
            self.run(day)
            day += timedelta(days=1)

    @property
    def today(self) -> Today:
        if self._today is None:
            raise ValueError("the day's weather has not run")
        return self._today

    # ── the steps ──────────────────────────────────────────────────────────────

    def weather(self, day: date) -> None:
        state = self.world.demand
        assert state is not None
        temps = np.array([self.climate.day(region, day).temp_mean_c for region in state.regions])
        self.output.append(day, "weather_daily", self.climate.rows(day))
        self._today = Today(temps, state.catalogue.shares(day, len(state.lines)))

    def acquisition(self, day: date) -> None:
        acquisition.step(self.world, self.reps, day, self.today.shares)

    def demand(self, day: date) -> None:
        today = self.today
        today.events = demand.step(self.world, self.terms, day, today.temps, today.shares)

    def field(self, day: date) -> None:
        """Step 2: G2 S2.1–S2.3."""

    def orders(self, day: date) -> None:
        self.report.add(day, orders.step(self.world, self.desk, day, self.today.events, self.output))

    def supply(self, day: date) -> None:
        """Step 5: G3 S3.1."""

    def delivery(self, day: date) -> None:
        """Step 6: G3 S3.2–S3.3."""

    def finance(self, day: date) -> None:
        """Step 7: G4 S4.1."""

    def shops(self, day: date) -> None:
        """Step 8: G4 S4.2."""

    def documents(self, day: date) -> None:
        """Step 8b: DOC.2–DOC.3."""

    def periods(self, day: date) -> None:
        for _, hook in self.ends(day):
            hook(day)

    def checkpoint(self, day: date) -> None:
        self.world.day = self.last_day_done = day
        self._today = None
        if day == self.config.history.first_day - timedelta(days=1):
            self.warm_up_end = Checkpoint.of(self.world)

    # ── the period ends ────────────────────────────────────────────────────────

    def ends(self, day: date) -> list[tuple[str, Callable[[date], object]]]:
        """The period-end hooks of `day`, in the order they run (module docstring)."""
        world, closing, output = self.world, self.closing, self.output
        out: list[tuple[str, Callable[[date], object]]] = []
        last_of_quarter = day == quarter_end(day)
        if last_of_quarter:
            out.append(("roll_lists", lambda d: quarter_end_hook.roll_lists(world, closing, d, output)))
        if day == month_end(day):
            out.append(("month_end", lambda d: month_end_hook.close(world, closing, d, output)))
        if last_of_quarter:
            out.append(("roll_budgets", lambda d: quarter_end_hook.roll_budgets(world, closing, d)))
        if (day.month, day.day) == (12, 31):
            out.append(("year_end", lambda d: year_end_hook.close(world, closing, d, output)))
        return out
