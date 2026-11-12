"""S5 on paper (G2 S2.3 T6; review-199 R199-B/C): what `s5.sql` and `s5_best_worst.sql` should give each region, from
the mechanics of T1–T4 (`loop.pos`, `loop.field.handouts`), `s5.yaml`, `world.yaml` and the opening market — no build.
`s5_best_worst`'s threshold lies between the null world's spread and the story's, both from here (`test_s5_expected`),
never from a build; G5 S5.2 does not retune it. It is provisional — S2.4 corrects the gain and re-derives it (NH-D17);
it is not certified.

Per region, a day of the steady state:
- **the candidates**: the small-retail stores on its open tourplans whose customer is active on the warm-up's first
  day, each with its display uplift `u` (`pos.Grid`; the null world's flat) and the daily net revenue `m` of the lines
  its display carries (`pos.best_line`'s brand, or line: its base on the lines its customer buys × each line's mean
  case price of its listing group, at list). That is the Customer Margin while G3/G4 cost nothing (contracts §4);
  G5 S5.2 re-checks it on their costs;
- **the displays out**: the month's budget × the manager's `budget_factor` ÷ the displays' mean cost, placed each
  month, out for a loan's mean term (`pos_loans.months`). A skilled manager's early withdrawals and a lost customer's
  return shorten a loan's gain and cost alike: left out;
- **which stores hold them**: weighted sampling without replacement ∝ `u ^ (pos_skill × choice_exponent)`, a store's
  chance of holding one by the Poisson approximation `1 − exp(−λ w)`, λ so the chances sum to the displays out;
- **the gain**: each store's chance × `u × m` (`s5.sql` reads the margin during the loan less that before);
- **the cost**: each store's chance × its display's cost over `S5_LIFE_DAYS` (`s5.sql`'s), and the give-aways: the
  region's calls a day — each open tourplan 1 ÷ 7 its cycle's weeks, × the adherence, × 1 + the ad-hoc share, × the
  share of its kind's visits that deal with someone — × the give-away share (× the `budget_factor`; a skilled
  manager's less the cut, `field.handouts`) × a give-away's expected cost: at a sales visit that orders (S3's
  `conversion_base`) one of the give-aways of the lines its customer buys, else one of the brand-wide ones, ∝ 1 ÷
  `unit_cost`, × the mean `qty`.

Each region's gain ÷ cost, its return (gain − cost) ÷ cost, the spread (`S5_BEST` ÷ `S5_WORST`) and the Spearman of
the returns against `pos_skill` (average ranks, `s5.sql`'s); for the null world (S5 off: `u` flat, every manager's
skill 1 and no factor, nobody skilled) the same.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from hartland_sim.checks.stories import S5_BEST, S5_LIFE_DAYS, S5_WORST
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.loop.field import UNREACHED, kind_of
from hartland_sim.loop.pos import Material, Terms, best_line, fits
from hartland_sim.world import World

Array = np.ndarray[Any, Any]


@dataclass(frozen=True)
class Region:
    """A region's expected day (module docstring), CZK."""

    displays: float  # out on an average day
    gain: float
    display_cost: float
    giveaway_cost: float

    @property
    def cost(self) -> float:
        return self.display_cost + self.giveaway_cost

    @property
    def per_cost(self) -> float:
        """Gain ÷ cost, `s5_best_worst.sql`'s side."""
        return self.gain / self.cost

    @property
    def pos_return(self) -> float:
        """(Gain − cost) ÷ cost, `s5.sql`'s."""
        return (self.gain - self.cost) / self.cost


def chances(weights: Array, out: float) -> Array:
    """Each store's chance of holding one of `out` displays drawn ∝ `weights` without replacement (module docstring)."""
    out = min(out, len(weights) - 1e-6)
    low, high = 0.0, 1.0
    while (1 - np.exp(-high * weights)).sum() < out:
        high *= 2
    for _ in range(200):
        mid = (low + high) / 2
        low, high = (mid, high) if (1 - np.exp(-mid * weights)).sum() < out else (low, mid)
    return np.asarray(1 - np.exp(-high * weights))


def handout_cost(giveaways: tuple[Material, ...], lines: set[str]) -> float:
    """A give-away's expected unit cost among those of `lines` (the brand-wide ones without), ∝ 1 ÷ `unit_cost`."""
    pool = [m for m in giveaways if m.line in lines] or [m for m in giveaways if m.line is None]
    return len(pool) / sum(1 / m.unit_cost for m in pool)


def expected(world: World, config: WorldConfig, stories: StoriesConfig) -> dict[str, Region]:
    """Each region's expected day from the opening `world` (module docstring), in `region.csv`'s order; `stories`
    enabled is the story, `stories.null()` the null world."""
    demand = world.demand
    if demand is None:
        raise ValueError("the World has no demand state (setup makes it)")
    terms = Terms.of(config, stories, demand.lines)
    s5 = stories.story("S5")
    skills = {r: float(s5.parameters["pos_skill"][r]) if s5.enabled else 1.0 for r in terms.regions}
    factors = {r: float(s5.parameters["budget_factor"].get(r, 1.0)) if s5.enabled else 1.0 for r in terms.regions}
    first = config.history.warm_up_from
    _, _, price = demand.catalogue.shares(first, len(demand.lines))
    points = {int(point): p for p, point in enumerate(demand.point_ids)}
    giveaways = tuple(m for m in terms.materials.values() if m.kind == "giveaway")
    spec, field = config.field.pos_handouts, config.field
    low, high = field.pos_loans.months
    term_days = (low + high) / 2 * 365.25 / 12
    conversion = float(stories.story("S3").parameters["conversion_base"])
    by_region: dict[str, list[tuple[float, float, float]]] = {r: [] for r in terms.regions}  # (u, m, cost)
    calls: dict[str, list[tuple[float, float]]] = {r: [] for r in terms.regions}  # (calls a day, a give-away's cost)
    for tourplan_id in world.tourplan_of.values():
        t = world.tourplans[tourplan_id]
        region = terms.region_of[t.area]
        customer = world.customers[t.customer_id]
        kind = kind_of(customer, first)
        p = points[t.delivery_point_id]
        carried = demand.carried[demand.point_customer[p]]
        bought = {line.code for i, line in enumerate(demand.lines) if carried[i]}
        dealt = 1 - sum(field.outcomes[kind][name] for name in UNREACHED)
        a_day = dealt * field.adherence * (1 + field.ad_hoc_share) / (7 * t.cycle_weeks)
        unit = handout_cost(giveaways, set())
        if kind == "sales":
            unit = conversion * handout_cost(giveaways, bought) + (1 - conversion) * unit
        calls[region].append((a_day, unit * sum(spec.qty) / 2))
        store = demand.spots[p].store
        if customer.channel != "small_retail" or kind != "sales" or store is None:
            continue
        line = demand.lines[best_line(demand, p, terms.brands)].code
        display = fits(terms, line)[0]
        lines = [i for i, held in enumerate(demand.lines) if held.code in terms.displays[display] and carried[i]]
        revenue = float((demand.base[p, lines] * price[demand.point_group[p], lines]).sum())
        by_region[region].append((terms.grid.uplift(store), revenue, terms.materials[display].unit_cost))
    out = {}
    for region in terms.regions:
        held = np.array(by_region[region])
        u, m, cost = held[:, 0], held[:, 1], held[:, 2]
        weights = u ** (skills[region] * terms.choice_exponent)
        mean_cost = float((weights * cost).sum() / weights.sum())
        displays = float(config.field.pos_budget[region]) * factors[region] / mean_cost * term_days / (365.25 / 12)
        held_by = chances(weights, displays)
        share = spec.share * (1 - terms.giveaway_cut if terms.skilled(skills[region]) else factors[region])
        giveaway = sum(a_day * unit for a_day, unit in calls[region]) * share
        out[region] = Region(
            float(held_by.sum()),
            float((held_by * u * m).sum()),
            float((held_by * cost).sum() / S5_LIFE_DAYS),
            float(giveaway),
        )
    return out


def spread(found: Mapping[str, Region]) -> float:
    """`s5_best_worst.sql`'s spread on paper."""
    worst = found[S5_WORST].per_cost
    return found[S5_BEST].per_cost / worst if worst > 0 else math.nan


def rho(found: Mapping[str, Region], stories: StoriesConfig) -> float:
    """`s5.sql`'s Spearman on paper: the returns against `pos_skill`, on average ranks."""
    skill = stories.story("S5").parameters["pos_skill"]
    regions = [r for r in found if r in skill]
    returns = _ranks([found[r].pos_return for r in regions])
    skills = _ranks([float(skill[r]) for r in regions])
    return float(np.corrcoef(returns, skills)[0, 1])


def _ranks(values: list[float]) -> Array:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    k = 0
    while k < len(order):
        same = k
        while same + 1 < len(order) and values[order[same + 1]] == values[order[k]]:
            same += 1
        ranks[order[k : same + 1]] = (k + same) / 2 + 1
        k = same + 1
    return ranks
