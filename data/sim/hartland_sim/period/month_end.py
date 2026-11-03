"""The month-end's G1 half (`05` §3 month-end, G1 S1.4 T3): on a month's last day, after its orders, the next month's
conditions, the customers' phases, the rolling plan and the channel lists. The G4 half — the promotion results, the
campaign costs, the allocations, the GL — is S4.3's.

A row of a later period is written under the day it becomes valid (`Output.append(valid_from, …)`), so the warm-up's
last month-end, 2023-12-31, writes January 2024's conditions, plan rows and channel lists, and nothing valid before
2024-01-01 is written.

- **The conditions** (`customer_condition`, S1.3's tiers): each tiered (small-retail) customer active on M+1's first
  day gets M+1's `Condition` from what its month M booked (`World.bought`): its net revenue, the product lines it paid
  for, whether coffee and culinary both. It is set on the customer and written.
- **The phases** (R-4, `world.yaml policies.lifecycle`): each tiered customer active on M+1's first day is phased from
  its last `lookback_months` months: VIP when their net revenue reached `vip_min_net_revenue`, regular when it ordered
  in `regular_min_order_months` of them, else occasional. A change opens an episode from M+1's first day in the
  `World` (`lifecycle_phase` is rendered when the build ends, ⚑R201-A); potential and lost are the acquisition's.
- **The plan** (`sales_plan`, stream `plan`): for M+1…M+3 the demand model's expectation — each point of a customer
  active today, its base on the lines it buys and its group lists on the month's first day, × S1's season of the
  month by channel (no weather, promotions or discounts) — summed to the key account (big retail, gas KA), the rep
  area (small retail) or the channel (the others), × a planning error uniform in ± `policies.sales_plan.error_pct`
  of its months ahead, drawn per row by its id. `planned_net_revenue` is its cases at the standard list in force
  (`price_case`, the SKUs by their shares of the point's group). `version` is how near the month is: 1 three months
  ahead, 3 one month ahead (the month-ends of M−2, M−1, M each plan M+1 once). On a quarter's last day M+1…M+3 are
  Q+1, priced at Q+1's list (the quarter-end rolls it first), and each budget holder's planned gross of Q+1 — a rep
  area's, a KAM account's, its points' gross × their rows' error — is kept for the quarter-end's budgets
  (`World.planned_gross`).
- **The channel lists** (S1.3): each channel's list of M+1 from the standard list in force on M+1's first day, into
  the `World`, its rows written.

`World.bought` keeps the months the next month-end reads; the older ones are dropped.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np

from hartland_sim.dates import add_months, month_end, quarter_end
from hartland_sim.ids import row_id
from hartland_sim.io.reference import Row
from hartland_sim.loop.demand import CHANNELS
from hartland_sim.loop.pricing import TIERED, channel_list, channel_rows, condition, condition_row
from hartland_sim.loop.warmup import Output
from hartland_sim.period import Closing
from hartland_sim.rng import uniforms
from hartland_sim.setup.rows import row
from hartland_sim.world import Bought, Customer, Episode, World

REASONS = {
    "vip": "Obrat za posledních {months} měsíců nad hranicí VIP",
    "regular": "Objednávky alespoň v {min_months} z posledních {months} měsíců",
    "occasional": "Objednávky v méně než {min_months} z posledních {months} měsíců",
}
Array = np.ndarray[Any, Any]


def close(world: World, closing: Closing, day: date, output: Output) -> None:
    """The month-end of `day`'s month (module docstring), on its last day."""
    if day != month_end(day):
        raise ValueError(f"{day} is not a month's last day")
    first = day + timedelta(days=1)
    conditions(world, closing, day, output)
    phases(world, closing, day)
    plan(world, closing, day, output)
    lists(world, closing, first, output)
    keep = add_months(first, 1 - closing.config.policies.lifecycle.lookback_months)  # the next one reads from here
    world.bought = {month: bought for month, bought in world.bought.items() if month >= keep}
    # S4.3: the promotion results, the campaign costs, the allocations onto the lines, the GL.


def _tiered(world: World, first: date) -> list[Customer]:
    return [c for c in world.customers.values() if c.channel in TIERED and c.active_on(first)]


def _lines(closing: Closing, held: Bought) -> list[str]:
    return [code for code, i in closing.rules.lines.items() if held.lines >> i & 1]


def conditions(world: World, closing: Closing, day: date, output: Output) -> list[Row]:
    """M+1's `customer_condition` rows, its `Condition` set on each customer (module docstring)."""
    first = day + timedelta(days=1)
    month = world.bought.get(day.replace(day=1), {})
    tiers = closing.config.pricing.tiers
    rows = []
    for customer in _tiered(world, first):
        held = month.get(customer.customer_id, Bought())
        lines = _lines(closing, held)
        both = len({closing.business[line] for line in lines}) > 1
        customer.condition = condition(tiers, first, held.net_revenue, len(lines), both)
        rows.append(condition_row(customer.customer_id, customer.condition))
    output.append(first, "customer_condition", rows)
    return rows


def phase(closing: Closing, months: list[Bought]) -> str:
    """The phase of a customer whose last months booked `months` (module docstring)."""
    rule = closing.config.policies.lifecycle
    if sum(held.net_revenue for held in months) >= rule.vip_min_net_revenue:
        return "vip"
    if sum(held.orders > 0 for held in months) >= rule.regular_min_order_months:
        return "regular"
    return "occasional"


def phases(world: World, closing: Closing, day: date) -> list[tuple[int, Episode]]:
    """The episodes opened from M+1's first day, each added to its customer (module docstring)."""
    first = day + timedelta(days=1)
    rule = closing.config.policies.lifecycle
    window = [add_months(first, -k) for k in range(1, rule.lookback_months + 1)]
    opened = []
    for customer in _tiered(world, first):
        held = customer.phase_on(day)
        if held in ("potential", "lost"):
            continue
        months = [world.bought.get(month, {}).get(customer.customer_id, Bought()) for month in window]
        target = phase(closing, months)
        if target != held:
            reason = REASONS[target].format(months=rule.lookback_months, min_months=rule.regular_min_order_months)
            episode = Episode(target, first, reason)
            customer.episodes.append(episode)
            opened.append((customer.customer_id, episode))
    return opened


def expectation(world: World, closing: Closing, day: date, month: date) -> tuple[Array, Array]:
    """(P, L) the cases each point is expected to order in `month` at flat terms, and their gross at the standard
    list in force (module docstring)."""
    demand = world.demand
    standard = world.standard_list
    assert demand is not None and standard is not None
    goods = demand.catalogue
    today = day.toordinal()
    active = ((demand.customer_since <= today) & (today <= demand.customer_until))[demand.point_customer]
    share, listed, _ = goods.shares(month, len(demand.lines))
    buys = listed[demand.point_group] & demand.carried[demand.point_customer] & active[:, None]
    days = (month_end(month) - month).days + 1
    season = np.mean([closing.seasons.of_day(month + timedelta(days=k)) for k in range(days)], axis=0)
    cases = demand.base * buys * season[demand.point_channel] * days
    prices = np.array([standard.prices[int(s)].case if int(s) in standard.prices else 0.0 for s in goods.sku_ids])
    price = _sums(goods.line, (share * prices).T, len(demand.lines)).T  # (G, L): a line's mean case price by group
    return cases, cases * price[demand.point_group]


def _sums(index: Array, weights: Array, n: int) -> Array:
    """(n, m) the rows of `weights` (k, m) summed by their `index` (k,)."""
    return np.stack([np.bincount(index, weights=weights[:, j], minlength=n) for j in range(weights.shape[1])], axis=1)


def plan(world: World, closing: Closing, day: date, output: Output) -> list[Row]:
    """The `sales_plan` rows of M+1…M+3; on a quarter's last day, Q+1's planned gross by budget holder (module
    docstring)."""
    demand = world.demand
    assert demand is not None
    errors = closing.config.policies.sales_plan.error_pct
    first = day + timedelta(days=1)
    n_targets, n_lines = len(closing.targets), len(demand.lines)
    holders = np.zeros((len(closing.holders), n_lines))
    held = closing.point_holder >= 0
    rows = []
    for ahead, error in enumerate(errors, start=1):
        month = add_months(first, ahead - 1)
        version = len(errors) - ahead + 1
        cases, gross = expectation(world, closing, day, month)
        summed = _sums(closing.point_target, cases, n_targets)
        valued = _sums(closing.point_target, gross, n_targets)
        ids = np.zeros((n_targets, n_lines), dtype=np.int64)
        for t, (channel, customer, area) in enumerate(closing.targets):
            for i, line in enumerate(demand.lines):
                ids[t, i] = row_id(
                    "sales_plan",
                    month=month,
                    channel=channel,
                    product_line=line.code,
                    customer=customer or "",
                    area=area or "",
                    version=version,
                )
        factor = 1 + (2 * uniforms("plan", ids=ids) - 1) * error / 100
        month_rows = []
        for t, (channel, customer, area) in enumerate(closing.targets):
            for i, line in enumerate(demand.lines):
                if summed[t, i] <= 0:
                    continue
                month_rows.append(
                    row(
                        "sales_plan",
                        sales_plan_id=int(ids[t, i]),
                        month=month,
                        channel=channel,
                        product_line_id=row_id("product_line", code=line.code),
                        customer_id=row_id("customer", code=customer) if customer else None,
                        area_id=row_id("area", code=area) if area else None,
                        planned_cases=round(float(summed[t, i] * factor[t, i]), 3),
                        planned_net_revenue=round(float(valued[t, i] * factor[t, i]), 2),
                        version=version,
                    )
                )
        output.append(month, "sales_plan", month_rows)
        rows += month_rows
        if day == quarter_end(day):
            weighted = gross[held] * factor[closing.point_target[held]]  # each point's gross × its row's error
            holders += _sums(closing.point_holder[held], weighted, len(closing.holders))
    if day == quarter_end(day):
        world.planned_gross = {
            holder: round(float(total), 2) for holder, total in zip(closing.holders, holders.sum(axis=1), strict=True)
        }
    return rows


def lists(world: World, closing: Closing, first: date, output: Output) -> None:
    """Each channel's list of the month from `first` (module docstring)."""
    standard = world.standard_list
    if standard is None:
        raise ValueError("the World has no standard list (setup makes it)")
    heads, items = [], []
    for channel in CHANNELS:
        made = channel_list(standard, closing.skus, closing.config, channel, first)
        world.channel_lists[channel] = made
        head, prices = channel_rows(made)
        heads.append(head)
        items += prices
    output.append(first, "channel_price_list", heads)
    output.append(first, "channel_price", items)
