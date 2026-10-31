"""The HoReCa sample (G0 S0.3 T5; `05` §2 step 5, `01` §4): from `osm-horeca.csv`, 600 customers and 2 000 potential
customers, stratified by okres population × affluence.

A place is eligible when it is a restaurant, café, bar (a pub counts as a bar) or canteen — fast food is not taken —
has a name, carries no brand (a chain café or a burger franchise is not Hartland's HoReCa customer), its name reads
as Czech and it lies in an okres. The Czech name is the sample's preference: the sample takes 2 600 of about
25 000 eligible places, so it loses nothing by it. A place mapped twice (the same name within `SAME_NAME_M`, as in
the store set) is one place, the first in OSM order. Each okres's quota is its share of population × affluence
(largest remainder, capped at what the okres has; the excess goes to the others). Within an okres the places are
taken in `osm_order` — fixed by the extract, not by the world's seed — the first ones as customers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hartland_sim.checks.czech import is_czech
from hartland_sim.setup.classifier import osm_order
from hartland_sim.setup.classifier.chains import fold
from hartland_sim.setup.classifier.place import SAME_NAME_M, mapped_twice

KINDS = {"restaurant": "restaurant", "cafe": "cafe", "bar": "bar", "pub": "bar", "canteen": "canteen"}
CUSTOMERS = 600
POTENTIAL = 2000


@dataclass(frozen=True)
class Pick:
    osm_id: str
    okres: str
    kind: str  # restaurant · cafe · bar · canteen
    role: str  # customer · potential


def eligible(place: Mapping[str, Any]) -> bool:
    return place["amenity"] in KINDS and bool(place["name"]) and not place["brand"] and is_czech(place["name"])


def quotas(weights: Mapping[str, float], total: int, cap: Mapping[str, int]) -> dict[str, int]:
    """`total` split over the keys in proportion to `weights` (largest remainder), none above its `cap`; what a capped
    key cannot take goes to the others, round by round. A key of weight 0 takes nothing."""
    given = dict.fromkeys(weights, 0)
    remaining = min(total, sum(cap[key] for key in weights if weights[key] > 0))
    while remaining:
        open_keys = sorted(key for key in weights if given[key] < cap[key] and weights[key] > 0)
        if not open_keys:
            raise ValueError(f"no key can take the remaining {remaining}")
        weight = sum(weights[key] for key in open_keys)
        shares = {key: remaining * weights[key] / weight for key in open_keys}
        whole = {key: int(shares[key]) for key in open_keys}
        by_remainder = sorted(open_keys, key=lambda key: (whole[key] - shares[key], key))
        for key in by_remainder[: remaining - sum(whole.values())]:
            whole[key] += 1
        for key in open_keys:
            take = min(whole[key], cap[key] - given[key])
            given[key] += take
            remaining -= take
    return given


def sample(
    places: Sequence[Mapping[str, Any]],
    okres_of: Sequence[str | None],
    weight: Mapping[str, float],
    *,
    customers: int = CUSTOMERS,
    potential: int = POTENTIAL,
) -> list[Pick]:
    """The sample: `customers` + `potential` picks over the okresy of `weight` (population × affluence)."""
    candidates = [
        (place, okres) for place, okres in zip(places, okres_of, strict=True) if okres is not None and eligible(place)
    ]
    points = np.array([(place["lon"], place["lat"]) for place, _ in candidates], dtype=float).reshape(-1, 2)
    twice = mapped_twice([fold(place["name"]) for place, _ in candidates], points, SAME_NAME_M)
    by_okres: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for i, (place, okres) in enumerate(candidates):
        if i not in twice:
            by_okres[okres].append(place)
    for in_okres in by_okres.values():
        in_okres.sort(key=lambda place: osm_order(place["osm_id"]))
    available = {okres: len(by_okres.get(okres, [])) for okres in weight}
    total = quotas(weight, customers + potential, available)
    as_customers = quotas(weight, customers, total)
    picks = []
    for okres in sorted(weight):
        for rank, place in enumerate(by_okres.get(okres, [])[: total[okres]]):
            role = "customer" if rank < as_customers[okres] else "potential"
            picks.append(Pick(place["osm_id"], okres, KINDS[place["amenity"]], role))
    return picks
