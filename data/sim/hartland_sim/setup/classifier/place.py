"""Street type, store class and size class (G0 S0.3 T4; `03` §3, `01` §7, `04` S4).

**Street type** (`04` S4's clusters) comes from what is around the store in the same extract (`osm-context.csv`), by
the first rule that holds:

| street type | rule |
|---|---|
| `petrol` | a station (`amenity=fuel`) or a station chain's shop |
| `shopping_centre` | within `MALL_M` of a mall (`shop=mall`) |
| `transport_hub` | within `HUB_M` of a railway or metro station or a bus station (a halt is not a hub) |
| `village` | its nearest settlement point is a village, and no town, city, suburb or quarter point is within `TOWN_M` |
| `high_street` | within `CENTRE_M` of a town's or city's centre point, or `QUARTER_M` of a suburb's or quarter's |
| `residential` | the rest |

A suburb or quarter point is urban by OSM's definition, so the quarter rule needs no city around it: inside Prague,
Brno, Ostrava or Plzeň the nearest settlement point is often an old village that OSM keeps as `place=village`.

**Store class** (`03` §3) from the chain and the `shop` tag: `hyper` (Kaufland, Globus, Makro, a hypermarket by
name), `discount` (Lidl, Penny, Norma), `super` (any other `shop=supermarket`), `petrol`, `village` (a small shop in
a village), `convenience` (`shop=convenience`), `minimarket` (grocery, general, greengrocer, deli). A hypermarket or
a discounter is a destination, not a village shop: in a village it counts as `residential` (`settle`). **Size class**
(`01` §7: small / medium / large) follows the store class; the extract carries no building area.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from hartland_sim.io.reference import REFERENCE_DIR, Row, read
from hartland_sim.setup.classifier.chains import Chain, fold
from hartland_sim.setup.classifier.okres import COS_LAT, METRES_PER_DEGREE, Points

STREET_TYPES = ("high_street", "residential", "village", "transport_hub", "shopping_centre", "petrol")
STORE_CLASSES = ("hyper", "super", "discount", "convenience", "minimarket", "village", "petrol")
SIZE_OF = {
    "hyper": "L",
    "super": "M",
    "discount": "M",
    "convenience": "S",
    "minimarket": "S",
    "village": "S",
    "petrol": "S",
}

MALL_M = 150.0
HUB_M = 200.0
CENTRE_M = {"city": 900.0, "town": 350.0}
QUARTER_M = 250.0
TOWN_M = 2000.0  # a village point closer than this to a town is the town's outskirts
SAME_NAME_M = 30.0  # the same name this close is one place mapped twice

HYPER_CHAINS = frozenset({"KAUFLAND", "GLOBUS", "MAKRO"})
DISCOUNT_CHAINS = frozenset({"LIDL", "PENNY", "NORMA"})


@dataclass(frozen=True)
class Context:
    malls: Points
    hubs: Points
    settlements: Points  # city, town, village centre points
    settlement_kind: NDArray[np.str_]
    settlement_names: Sequence[str]  # the town a store without an address is in (`setup.market`)
    towns: Points  # the urban points: towns, cities and their suburbs and quarters
    quarters: Points  # suburbs and quarters

    @classmethod
    def load(cls, root: Path = REFERENCE_DIR) -> Context:
        """From `osm-context.csv` under `root`."""
        rows = read("osm-context", root)

        def points(selected: list[Row]) -> Points:
            return np.array([(row["lon"], row["lat"]) for row in selected], dtype=float).reshape(-1, 2)

        settlements = [row for row in rows if row["place"] in ("city", "town", "village")]
        return cls(
            malls=points([row for row in rows if row["shop"] == "mall"]),
            hubs=points([row for row in rows if row["railway"] == "station" or row["amenity"] == "bus_station"]),
            settlements=points(settlements),
            settlement_kind=np.array([row["place"] for row in settlements]),
            settlement_names=[row["name"] for row in settlements],
            towns=points([row for row in rows if row["place"] in ("city", "town", "suburb", "quarter")]),
            quarters=points([row for row in rows if row["place"] in ("suburb", "quarter")]),
        )


def nearest(targets: Points, points: Points, chunk: int = 500) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Each point's distance in metres to the nearest target (a local flat projection), and that target's index."""
    scale = np.array([COS_LAT * METRES_PER_DEGREE, METRES_PER_DEGREE])
    if not len(targets):
        return np.full(len(points), np.inf), np.zeros(len(points), dtype=np.intp)
    tx, ty = (targets * scale).T
    distance = np.empty(len(points))
    index = np.empty(len(points), dtype=np.intp)
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk] * scale
        d2 = (p[:, :1] - tx) ** 2 + (p[:, 1:] - ty) ** 2
        best = d2.argmin(axis=1)
        index[start : start + chunk] = best
        distance[start : start + chunk] = np.sqrt(d2[np.arange(len(p)), best])
    return distance, index


class Streets(NamedTuple):
    types: list[str]  # the street type of each point
    settlement: NDArray[np.intp]  # its nearest settlement point (an index into `Context.settlements`)


def street_types(points: Points, station: NDArray[np.bool_], context: Context) -> Streets:
    """The street type of each point (lon, lat); `station`: it is a station or a station chain's shop."""
    mall, _ = nearest(context.malls, points)
    hub, _ = nearest(context.hubs, points)
    settlement, which = nearest(context.settlements, points)
    quarter, _ = nearest(context.quarters, points)
    town, _ = nearest(context.towns, points)
    kind = context.settlement_kind[which]
    centre = np.array([CENTRE_M.get(str(k), 0.0) for k in kind])
    types = np.full(len(points), "residential", dtype=object)
    types[(settlement <= centre) | (quarter <= QUARTER_M)] = "high_street"
    types[(kind == "village") & (town > TOWN_M)] = "village"
    types[hub <= HUB_M] = "transport_hub"
    types[mall <= MALL_M] = "shopping_centre"
    types[station] = "petrol"
    return Streets([str(t) for t in types], which)


def mapped_twice(keys: Sequence[str | None], points: Points, radius: float) -> set[int]:
    """The objects mapped twice: an object whose key (a folded name, a chain) an earlier object within `radius` metres
    shares. The first in order stays; an empty key is never a duplicate."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        if key:
            groups[key].append(i)
    twice: set[int] = set()
    for members in groups.values():
        for position, i in enumerate(members[1:], start=1):
            earlier = [j for j in members[:position] if j not in twice]
            if nearest(points[earlier], points[i : i + 1])[0][0] <= radius:
                twice.add(i)
    return twice


def settle(street_type: str, store_class: str) -> str:
    """The street type once the store class is known: a hypermarket or a discounter in a village is out-of-town."""
    return "residential" if street_type == "village" and store_class in ("hyper", "discount") else street_type


def store_class(*, shop: str, amenity: str, name: str, chain: Chain | None, street_type: str) -> str:
    if amenity == "fuel" or (chain is not None and chain.is_station):
        return "petrol"
    code = chain.code if chain else ""
    if code in HYPER_CHAINS or "hypermarket" in fold(name):
        return "hyper"
    if code in DISCOUNT_CHAINS:
        return "discount"
    if shop == "supermarket":
        return "super"
    if street_type == "village":
        return "village"
    return "convenience" if shop == "convenience" else "minimarket"
