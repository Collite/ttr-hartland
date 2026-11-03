"""The period ends (architecture §2; `05` §4): `month_end` · `quarter_end` · `year_end`. G1 S1.4 · G4 S4.3.

`Closing` is what they read beside the `World`, built once a run and never pickled, as the loop's other run terms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.reference import read
from hartland_sim.loop.demand import KEY_ACCOUNTS, Seasons
from hartland_sim.loop.pricing import KAM_CHANNELS, TIERED, Rules
from hartland_sim.setup.portfolio import Sku
from hartland_sim.world import World

Array = np.ndarray[Any, Any]


@dataclass(frozen=True)
class Closing:
    """The period ends' run terms: `world.yaml`, the catalogue's SKUs, S1's seasons, the pricing rules, each product
    line's business, and each delivery point's targets — its `sales_plan` row's (`targets`: (channel, key account's
    code, rep area's code)) and its budget holder's (`holders`: a rep area's code, a KAM account's; `None` for a point
    no budget plans). The points and their customers' channels and areas never change in a run."""

    config: WorldConfig
    skus: tuple[Sku, ...]
    seasons: Seasons
    rules: Rules
    business: Mapping[str, str]  # product line code → its business (KAVA · KULINAR)
    targets: tuple[tuple[str, str | None, str | None], ...]
    point_target: Array  # (P,) index into `targets`
    holders: tuple[str, ...]
    point_holder: Array  # (P,) index into `holders`, -1 for none

    @classmethod
    def of(
        cls, world: World, config: WorldConfig, stories: StoriesConfig, skus: Sequence[Sku], rules: Rules
    ) -> Closing:
        demand = world.demand
        if demand is None:
            raise ValueError("the World has no demand state (setup makes it)")
        brands = {brand["code"]: brand["business"] for brand in read("brand")}
        business = {line["code"]: brands[line["brand"]] for line in read("product_line")}
        customers = list(world.customers.values())
        targets: dict[tuple[str, str | None, str | None], int] = {}
        holders: dict[str, int] = {}
        point_target, point_holder = [], []
        for c in demand.point_customer.tolist():
            customer = customers[c]
            if customer.channel in KEY_ACCOUNTS:
                target: tuple[str, str | None, str | None] = (customer.channel, customer.code, None)
            elif customer.channel in TIERED:
                target = (customer.channel, None, customer.area)
            else:
                target = (customer.channel, None, None)
            point_target.append(targets.setdefault(target, len(targets)))
            holder = customer.area if customer.channel in TIERED else None
            if customer.channel in KAM_CHANNELS:
                holder = customer.code
            point_holder.append(holders.setdefault(holder, len(holders)) if holder else -1)
        return cls(
            config,
            tuple(skus),
            Seasons.of(stories, demand.lines),
            rules,
            business,
            tuple(targets),
            np.array(point_target, dtype=np.int64),
            tuple(holders),
            np.array(point_holder, dtype=np.int64),
        )
