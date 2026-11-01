"""Setup, once per build (`05` §2): steps 1–7 in order, from the committed inputs — `world.yaml`, `stories/`,
`data/reference/` — into the opening `World` and the rows of the setup-time tables.

The opening is the state the warm-up starts from: `World.day` is the day before `history.warm_up_from`. Every draw
is made under `world.yaml`'s master seed, from a named stream (contracts §2), so two runs give the same world and the
same rows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.reference import REFERENCE_DIR
from hartland_sim.rng import using_master_seed
from hartland_sim.setup import customers, policies, portfolio, reference
from hartland_sim.setup import org as org_setup
from hartland_sim.setup.customers import Customers
from hartland_sim.setup.market import Market
from hartland_sim.setup.org import SALES, Org
from hartland_sim.setup.portfolio import Portfolio
from hartland_sim.setup.reference import Geography
from hartland_sim.setup.rows import Tables, tables
from hartland_sim.world import World


@dataclass
class Setup:
    world: World
    tables: Tables
    geography: Geography
    org: Org
    portfolio: Portfolio
    covers: dict[tuple[int, int], tuple[float, float]]  # (sku_id, dc_site_id) → the policy's (min, max) days
    customers: Customers


def run(
    world: WorldConfig | None = None,
    stories: StoriesConfig | None = None,
    root: Path = REFERENCE_DIR,
    market: Market | None = None,
) -> Setup:
    """The opening world and the setup tables' rows. `market` saves rebuilding the OSM market (`setup.market`)."""
    world = world or WorldConfig.load()
    stories = stories or StoriesConfig.load()
    with using_master_seed(world.master_seed):
        out = tables()
        geography = reference.build(out, world, root)
        org = org_setup.build(out, world, stories, geography, root)
        products = portfolio.build(out, world, stories, root)
        covers = policies.stock_policies(out, world, products)
        policies.rules(out, world, owner=org.actors[SALES].employee_id)
        policies.pos(out, world, geography, root)
        policies.fleet(out, world, root)
        made = customers.build(out, world, stories, geography, org, market)
        state = World(world.history.warm_up_from - timedelta(days=1))
        state.actors = {actor.employee_id: actor for actor in org.actors.values()}
        state.duties = list(org.duties)
        state.customers = {customer.customer_id: customer for customer in made.customers}
        state.budgets = policies.budgets(out, world, planned(state, made))
    return Setup(state, out, geography, org, products, covers, made)


def planned(state: World, made: Customers) -> list[tuple[int, str, int]]:
    """Each rep's and KAM's (employee_id, channel, customers active) on the opening quarter's first day — the
    base of the opening discount budgets: a rep's area's small-retail customers, a KAM's accounts by channel."""
    first = state.day + timedelta(days=1)
    by_code = {customer.code: customer for customer in made.customers}
    in_area = Counter(c.area for c in made.customers if c.channel == "small_retail" and c.active_on(first))
    out = []
    for duty in state.duties:
        if not duty.on(first):
            continue
        if duty.kind == "rep_area":
            out.append((duty.employee_id, "small_retail", in_area[duty.target]))
        elif duty.kind == "kam_customer" and by_code[duty.target].active_on(first):
            out.append((duty.employee_id, by_code[duty.target].channel, 1))
    return out
