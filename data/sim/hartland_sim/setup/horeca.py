"""The HoReCa sample (G0 S0.3 T5; `05` §2 step 5, `01` §4): from `osm-horeca.csv`, 600 customers and 2 000 potential
customers, stratified by okres population × affluence.

A place is eligible when it is a restaurant, café, bar (a pub counts as a bar) or canteen — fast food is not taken —
has a name, carries no brand (a chain café or a burger franchise is not Hartland's HoReCa customer), its name reads
as Czech (NH-D8) and it lies in an okres. Each okres's quota is its share of population × affluence (largest
remainder, capped at what the okres has; the excess goes to the others). Within an okres the places are taken in the
order of a hash of their OSM id — fixed by the extract, not by the world's seed — the first ones as customers.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hartland_sim.checks.czech import assert_czech

KINDS = {"restaurant": "restaurant", "cafe": "cafe", "bar": "bar", "pub": "bar", "canteen": "canteen"}
CUSTOMERS = 600
POTENTIAL = 2000


@dataclass(frozen=True)
class Pick:
    osm_id: str
    okres: str
    kind: str  # restaurant · cafe · bar · canteen
    role: str  # customer · potential


def order(osm_id: str) -> str:
    """The fixed order of the sample within an okres."""
    return hashlib.blake2b(osm_id.encode(), digest_size=8).hexdigest()


def eligible(place: Mapping[str, str]) -> bool:
    if place["amenity"] not in KINDS or not place["name"] or place["brand"]:
        return False
    try:
        assert_czech(place["name"])
    except AssertionError:
        return False
    return True


def quotas(weights: Mapping[str, float], total: int, cap: Mapping[str, int]) -> dict[str, int]:
    """`total` split over the keys in proportion to `weights` (largest remainder), none above its `cap`; what a capped
    key cannot take goes to the others, round by round."""
    given = dict.fromkeys(weights, 0)
    remaining = min(total, sum(cap.values()))
    while remaining:
        open_keys = sorted(key for key in weights if given[key] < cap[key] and weights[key] > 0)
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
    places: Sequence[Mapping[str, str]],
    okres_of: Sequence[str | None],
    weight: Mapping[str, float],
    *,
    customers: int = CUSTOMERS,
    potential: int = POTENTIAL,
) -> list[Pick]:
    """The sample: `customers` + `potential` picks over the okresy of `weight` (population × affluence)."""
    by_okres: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for place, okres in zip(places, okres_of, strict=True):
        if okres is not None and eligible(place):
            by_okres[okres].append(place)
    for candidates in by_okres.values():
        candidates.sort(key=lambda place: order(place["osm_id"]))
    available = {okres: len(by_okres.get(okres, [])) for okres in weight}
    total = quotas(weight, customers + potential, available)
    as_customers = quotas(weight, customers, total)
    picks = []
    for okres in sorted(weight):
        for rank, place in enumerate(by_okres.get(okres, [])[: total[okres]]):
            role = "customer" if rank < as_customers[okres] else "potential"
            picks.append(Pick(place["osm_id"], okres, KINDS[place["amenity"]], role))
    return picks
