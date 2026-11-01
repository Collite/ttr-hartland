"""Setup step 7 (`05` §2): the opening balances, and writing the opening to a run folder.

**Stock.** Each SKU on sale on the warm-up's first day opens at both DCs at its policy's midpoint, `(min + max) / 2`
days of cover, and nothing is reserved or in transit; there are no open orders, shipments or invoices. A day of cover
needs a rate, and the demand model is S1.2's: the opening rate is the opening plan's — the same
`pricing.budgets.opening_quarter_gross` the opening discount budgets read — for every customer active on that day,
its quarter's gross spread over its delivery points' DCs, then over the SKUs its channel lists (by importance + 1),
in cases at the SKU's list price for the channel. The warm-up year's reorder runs (G3) bring the stock to the plan's
own rate before the first written day. The opening `stock_day` rows are dated the day before the warm-up.

**The state tables** (`STATE_TABLES`, ⚑R201-A): `employee`, `assignment`, `customer`, `lifecycle_phase` and
`discount_budget` hold state the loop changes — a churned customer's `until`, a closed phase episode or duty, a
budget's `used_amount`. They are rendered from the `World` when a build ends (`state_rows`), never appended, and a
prolong renders them from the checkpoint alone. The opening is a build of zero days: setup makes no rows for them.

**Writing** (`write`): setup's other tables and the state tables, through the CSV writer in `load_order`,
`_sim_meta` (the inputs' hashes, the day before the warm-up as `last_day_done`) and the opening checkpoint. A run
folder is written once.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

import hartland_sim
from hartland_sim.config import INVENTORY_FILE, STORIES_DIR, WORLD_FILE, StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.io.reference import REFERENCE_DIR
from hartland_sim.setup.customers import Customers, customer_row, phases
from hartland_sim.setup.org import assignment, employee_row
from hartland_sim.setup.policies import budget_row
from hartland_sim.setup.portfolio import Portfolio, listed
from hartland_sim.setup.reference import Geography
from hartland_sim.setup.rows import Tables, row, tables
from hartland_sim.world import Checkpoint, Meta, Stock, World, digest

STATE_TABLES = ("employee", "assignment", "customer", "lifecycle_phase", "discount_budget")


def rates(
    world: WorldConfig, portfolio: Portfolio, made: Customers, geography: Geography, day: date
) -> dict[tuple[int, int], float]:
    """Cases a day per (sku_id, dc_site_id) at the opening (module docstring)."""
    gross = world.pricing.budgets.opening_quarter_gross
    quarter_days = (_next_quarter(day) - day).days
    by_key: dict[tuple[str, str], float] = defaultdict(float)  # (listing key, DC) → gross a day
    for customer in made.customers:
        if customer.channel not in gross or not customer.active_on(day):
            continue
        key = "special_retail" if customer.sub_channel == "special_retail" else customer.channel
        places = made.places[customer.code]
        for place in places:
            by_key[key, geography.dc_for(place.okres)] += gross[customer.channel] / quarter_days / len(places)
    live = [sku for sku in portfolio.skus if sku.launched_on <= day <= sku.last_day]
    factors = {str(channel): factor for channel, factor in world.pricing.channel_factor.items()}
    listings = {str(channel): listing for channel, listing in world.channels.listings.items()}
    out: dict[tuple[int, int], float] = defaultdict(float)
    for (key, dc), daily in sorted(by_key.items()):
        skus = [sku for sku in live if listed(listings[key], sku)]
        weight = sum(sku.importance + 1 for sku in skus)
        for sku in skus:
            price = sku.base_price_case * factors[key]
            out[sku.sku_id, row_id("site", code=dc)] += daily * (sku.importance + 1) / weight / price
    return dict(out)


def _next_quarter(day: date) -> date:
    first = date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)
    return date(first.year + (first.month + 3) // 13, (first.month + 2) % 12 + 1, 1)


def stock(
    out: Tables,
    world: WorldConfig,
    portfolio: Portfolio,
    covers: Mapping[tuple[int, int], tuple[float, float]],
    made: Customers,
    geography: Geography,
) -> dict[tuple[int, int], Stock]:
    """The opening `stock_day` rows into `out`; the `World`'s stock."""
    first = world.history.warm_up_from
    opening = first - timedelta(days=1)
    rate = rates(world, portfolio, made, geography, first)
    state = {}
    for sku in portfolio.skus:
        if not sku.launched_on <= first <= sku.last_day:
            continue
        for dc in world.sites.dcs:
            key = (sku.sku_id, row_id("site", code=dc))
            low, high = covers[key]
            days = (low + high) / 2
            cases = float(math.ceil(rate.get(key, 0.0) * days))
            state[key] = Stock(cases)
            out["stock_day"].append(
                row(
                    "stock_day",
                    sku_id=sku.sku_id,
                    dc_site_id=key[1],
                    day=opening,
                    on_hand_cases=cases,
                    reserved_cases=0.0,
                    in_transit_cases=0.0,
                    days_cover=days if cases else None,
                    below_min=False,
                    above_max=False,
                )
            )
    return state


def meta(world: WorldConfig, stories: StoriesConfig, last_day_done: date) -> Meta:
    """`_sim_meta` for a build of the committed inputs (`world.Meta`)."""
    inputs = [
        WORLD_FILE,
        INVENTORY_FILE,
        *sorted(STORIES_DIR.glob("*.yaml")),
        *sorted(path for path in REFERENCE_DIR.rglob("*") if path.is_file()),
    ]
    return Meta(
        master_seed=world.master_seed,
        stories_sha256=stories.sha256,
        world_sha256=hashlib.sha256(WORLD_FILE.read_bytes()).hexdigest(),
        last_day_done=last_day_done,
        build_id=digest(inputs, str(world.master_seed), hartland_sim.__version__)[:16],
    )


def state_rows(world_state: World, world: WorldConfig) -> Tables:
    """The state tables' rows, rendered from `world_state` in its own order (module docstring)."""
    out = tables()
    codes = {employee: actor.code for employee, actor in world_state.actors.items()}
    out["employee"] = [employee_row(actor) for actor in world_state.actors.values()]
    out["assignment"] = [assignment(duty, codes) for duty in world_state.duties]
    out["customer"] = [customer_row(customer, world) for customer in world_state.customers.values()]
    out["lifecycle_phase"] = phases(world_state.customers.values())
    out["discount_budget"] = [budget_row(*key, budget) for key, budget in world_state.budgets.items()]
    return out


def write(world_state: World, setup_rows: Tables, run: Path, world: WorldConfig, stories: StoriesConfig) -> Meta:
    """The opening into `run`: setup's tables, the state tables rendered from `world_state`, `_sim_meta`, the opening
    checkpoint. `FileExistsError` when `run` already holds a CSV; `ValueError` when `setup_rows` holds a state table's
    rows, which only the `World` may give."""
    held = [name for name in STATE_TABLES if setup_rows.get(name)]
    if held:
        raise ValueError(f"{held} are rendered from the World, not written from setup's rows")
    if run.exists() and any(run.glob("*.csv")):
        raise FileExistsError(f"{run} already holds a build")
    with CsvWriter(run) as writer:
        writer.write_all({**setup_rows, **state_rows(world_state, world)})
    opening = meta(world, stories, world_state.day)
    opening.save(run)
    Checkpoint.of(world_state).save(run)
    return opening
