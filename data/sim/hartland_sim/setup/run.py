"""Setup, once per build (`05` §2): steps 1–7 in order, from the committed inputs — `world.yaml`, `stories/`,
`data/reference/` — into the opening `World` and the rows of the setup-time tables.

The opening is the state the warm-up starts from: `World.day` is the day before `history.warm_up_from`. Every draw
is made under `world.yaml`'s master seed, from a named stream (contracts §2), so two runs give the same world and the
same rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.reference import REFERENCE_DIR
from hartland_sim.rng import using_master_seed
from hartland_sim.setup import org as org_setup
from hartland_sim.setup import policies, portfolio, reference
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


def run(
    world: WorldConfig | None = None,
    stories: StoriesConfig | None = None,
    root: Path = REFERENCE_DIR,
) -> Setup:
    """The opening world and the setup tables' rows."""
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
        state = World(world.history.warm_up_from - timedelta(days=1))
        state.actors = {actor.employee_id: actor for actor in org.actors.values()}
        state.duties = list(org.duties)
    return Setup(state, out, geography, org, products, covers)
