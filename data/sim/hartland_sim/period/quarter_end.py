"""The quarter-end's G1 half (`05` §3 quarter-end, G1 S1.4 T3b; ⚑R213-A): on a quarter's last day, Q+1's standard list
with its purchase prices, and the discount budgets of Q+1. The G4 half — the stock-policy reviews — is S4.3's.

On a quarter's last day the hooks run `roll_lists`, then the month-end (whose channel lists of M+1 need Q+1's standard
list, and whose plan of M+1…M+3 is Q+1's), then `roll_budgets` (which reads that plan).

- **The standard list** (`roll_lists`): each product line has a price index in the `World` (`World.price_index`, 1.0
  for setup's Q1-2023 list); each quarter moves it × (1 + d ÷ 100), d uniform in `world.yaml calendar.price_drift_pct`,
  drawn by (Q+1's first day, the line's id) from stream `pricing_agree`, and the quarter that holds
  `calendar.coffee_jump.starts` also × (1 + its `pct` ÷ 100) on the lines of its business (KAVA). Every SKU sold in
  Q+1 is listed at its `base_price_case` × its line's index, per piece ÷ its units, its purchase price its `cogs_case`
  × the same index (`pricing.standard_list`): each quarter's prices are the base's × the index, never a rounded price
  moved again, and a SKU launched in Q+1 enters at its line's index. The list goes into the `World` and its rows are
  written under Q+1's first day, issued by Marketing's director — so 2023-12-31's SC-2024-Q1 is the first written.
- **The budgets** (`roll_budgets`, ⚑NH-7): each rep holding an area on Q+1's first day and each KAM gets a budget of
  Q+1 of `pricing.budgets.{rep,kam}_discount_pct_of_planned_gross` × the planned gross of what it holds — its area's,
  its accounts' (the month-end's plan of Q+1 at Q+1's standard prices, `World.planned_gross`) — setup's rule with the
  plan in place of `opening_quarter_gross`. A KAM with no account gets 0, as a rep of an empty area does. The budgets
  are the `World`'s (`World.budgets`, used by the grants); `discount_budget` is rendered when the build ends.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from hartland_sim.dates import quarter_end, quarter_start
from hartland_sim.ids import row_id
from hartland_sim.loop.pricing import standard_list, standard_rows
from hartland_sim.loop.warmup import Output
from hartland_sim.period import Closing
from hartland_sim.rng import rng
from hartland_sim.setup.org import MARKETING
from hartland_sim.world import Budget, PriceList, World

ISSUER = tuple(MARKETING.split(":"))  # Marketing's director (role, org unit): who issues a standard list


def _first(day: date) -> date:
    if day != quarter_end(day):
        raise ValueError(f"{day} is not a quarter's last day")
    return day + timedelta(days=1)


def issuer(world: World) -> int:
    """Marketing's director's employee_id."""
    (found,) = [e for e, actor in world.actors.items() if (actor.role, actor.org_unit) == ISSUER]
    return found


def drift(closing: Closing, first: date, line: str) -> float:
    """The % line `line`'s index moves by into the quarter from `first` (module docstring)."""
    low, high = closing.config.calendar.price_drift_pct
    return float(rng("pricing_agree", "drift", first, row_id("product_line", code=line)).uniform(low, high))


def roll_lists(world: World, closing: Closing, day: date, output: Output) -> PriceList:
    """Q+1's standard list, into the `World` and written (module docstring)."""
    first = _first(day)
    jump = closing.config.calendar.coffee_jump
    index = dict(world.price_index)
    for line in closing.rules.lines:
        index[line] = index.get(line, 1.0) * (1 + drift(closing, first, line) / 100)
        if first == quarter_start(jump.starts) and closing.business[line] == jump.business:
            index[line] *= 1 + jump.pct / 100
    world.price_index = index
    standard = world.standard_list = standard_list(closing.skus, first, index)
    head, items = standard_rows(standard, issuer(world))
    output.append(first, "standard_price_list", [head])
    output.append(first, "standard_price", items)
    # S4.3: the stock-policy reviews.
    return standard


def roll_budgets(world: World, closing: Closing, day: date) -> dict[int, Budget]:
    """Q+1's budgets by employee_id, into the `World` (module docstring)."""
    first = _first(day)
    shares = closing.config.pricing.budgets
    planned = world.planned_gross
    totals: dict[int, float] = defaultdict(float)
    pct: dict[int, float] = {}
    for duty in world.duties:
        if not duty.on(first):
            continue
        if duty.kind == "rep_area":
            totals[duty.employee_id] += planned.get(duty.target, 0.0)
            pct[duty.employee_id] = shares.rep_discount_pct_of_planned_gross
        elif duty.kind == "kam_customer":
            totals[duty.employee_id] += planned.get(duty.target, 0.0)
            pct[duty.employee_id] = shares.kam_discount_pct_of_planned_gross
    for employee, actor in world.actors.items():
        if actor.role == "kam" and employee not in pct and (actor.left_on is None or actor.left_on >= first):
            totals[employee] = 0.0
            pct[employee] = shares.kam_discount_pct_of_planned_gross
    out = {employee: Budget(round(totals[employee] * pct[employee] / 100, 2)) for employee in pct}
    for employee, budget in out.items():
        world.budgets[employee, first] = budget
    return out
