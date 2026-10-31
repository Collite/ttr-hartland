"""The market (G0 S0.3 T6; `05` §2 step 2): the OSM extract → the `store` and `chain` rows of `03` §3, and the store
set report. Run `uv run python -m hartland_sim.setup.market` for the report.

From `osm-stores.csv`, in order:
1. **the okres** of each object (`classifier.okres`); one outside every leaf is dropped;
2. **the chain** (`classifier.chains`);
3. **no shop**: a station tagged `shop=no` without a brand is dropped (it has no shelf to sell from); one with a
   brand stays, as the extract's brand rule takes it, and the report counts those;
4. **names**: an OSM name is a proper noun and stays verbatim (⚑NH-2), whatever the Czech guard says of it: the
   Vietnamese minimarkets (`Potraviny An Duc`) and a station's `MOL, Praha 6 - Evropská (in)` are real Czech
   stores. NH-D8 guards the world's own words around a name; a renderer passes the name as `allow` (⚑R188-A). The
   report counts the names the guard would flag. A name the market makes up (an unnamed store's) reads as Czech;
5. **duplicates**: a station chain's shop or café mapped beside its own pumps (within `STATION_SHOP_M`; the pumps are
   the chain's by brand or by operator) is the station's shop, not a second store; the same chain (or brand of an
   independent network, such as Robin Oil) within `SAME_CHAIN_M`, or an independent's same name within `SAME_NAME_M`,
   is one store mapped twice (`place.mapped_twice`). The first in OSM order stays;
6. **the place** (`classifier.place`): street type, store class (`settle`d), size class;
7. **the hours** (`classifier.hours`); untagged or unreadable hours take those of a tagged store of the same class
   and chain (independents together), or of the class alone where that pool has fewer than `HOURS_POOL` stores,
   chosen by `osm_order` (`01` §7: "assigned by a seeded classifier where tags are missing");
8. **the rest of the row**: the name (OSM's; else the chain's; else the class's Czech word and the town), the town
   (Praha for a Prague district; else `addr:city`; else the nearest settlement), the street, the area and region.

Then the HoReCa sample (`setup.horeca`) joins as `store_class = horeca`, from the places that are not already a
store (an object tagged both a shop and a café is the shop). Station split (NH-D5): an OMV or Shell
station belongs to a key-account chain; every other station is `petrol` with its own chain or none, a small-retail
customer at G1. The rows are keyed by `osm_id`, the business key; their `bigint` ids come with `ids.py` (G1 S1.1).
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hartland_sim.checks.czech import is_czech
from hartland_sim.io.reference import REFERENCE_DIR, Row, read
from hartland_sim.setup.affluence import load as affluence
from hartland_sim.setup.classifier import chains, okres, osm_order, place
from hartland_sim.setup.classifier.chains import Chain
from hartland_sim.setup.classifier.hours import Week, parse
from hartland_sim.setup.horeca import Pick, sample

STATION_SHOP_M = 150.0
SAME_CHAIN_M = 50.0
SAME_NAME_M = place.SAME_NAME_M
HOURS_POOL = 30  # the fewest tagged stores a class-and-chain pool draws from; below it, the class's
INDEPENDENT = "independent"
PRAHA = "CZ0100"

# The name of an unnamed independent: the class's Czech word and the town.
UNNAMED = {
    "super": "Supermarket",
    "convenience": "Potraviny",
    "minimarket": "Potraviny",
    "village": "Smíšené zboží",
    "petrol": "Čerpací stanice",
    "hyper": "Hypermarket",
    "discount": "Prodejna",
}

STORE_COLUMNS = (
    "osm_id",
    "name",
    "brand",
    "chain",
    "store_class",
    "size_class",
    "street_type",
    "weekly_hours",
    "sunday_open",
    "lat",
    "lon",
    "street",
    "town",
    "postcode",
    "okres",
    "area",
    "region",
    "opened_on",
    "closed_on",
)


@dataclass
class Report:
    read: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    kept: Counter[str] = field(default_factory=Counter)  # stores kept that a reader might expect dropped, by why
    snapped: int = 0
    hours_drawn: Counter[str] = field(default_factory=Counter)  # by the pool: "class and chain", "class"
    by_class_region: Counter[tuple[str, str]] = field(default_factory=Counter)

    def lines(self) -> list[str]:
        regions = sorted({region for _, region in self.by_class_region})
        classes = [c for c in (*place.STORE_CLASSES, "horeca") if any(k[0] == c for k in self.by_class_region)]
        out = [
            f"read {self.read} OSM objects; dropped {dict(self.dropped)}; kept {dict(self.kept)}; "
            f"{self.snapped} snapped to an okres; hours drawn {dict(self.hours_drawn)}"
        ]
        out.append("class        " + " ".join(f"{region:>6}" for region in regions) + "   total")
        for store_class in classes:
            counts = [self.by_class_region[store_class, region] for region in regions]
            out.append(f"{store_class:<12} " + " ".join(f"{n:>6}" for n in counts) + f" {sum(counts):>7}")
        return out


@dataclass
class Market:
    stores: list[Row]
    chains: list[Row]
    horeca: list[Pick]
    report: Report


def build(root: Path = REFERENCE_DIR) -> Market:
    report = Report()
    objects = read("osm-stores", root)
    report.read = len(objects)
    leaves = okres.load(root / "osm" / "okres-boundaries.geojson")
    context = place.Context.load(root)
    area_of = {row["okres"]: row["area"] for row in read("area_okres", root)}
    region_of = {row["code"]: row["region"] for row in read("area", root)}
    okres_names = {row["code"]: row["name"] for row in read("okres", root)}

    points = _points(objects)
    assignment = okres.assign(leaves, points)
    report.snapped += assignment.snapped
    kept: list[tuple[Row, Chain | None, str]] = []
    for row, code in zip(objects, assignment.codes, strict=True):
        chain = chains.match(brand=row["brand"], name=row["name"], operator=row["operator"], fuel=_fuel(row))
        if code is None:
            report.dropped["outside"] += 1
        elif _no_shop(row) and not row["brand"]:
            report.dropped["no shop"] += 1
        else:
            kept.append((row, chain, code))
    kept = _without_duplicates(kept, report)
    report.kept["station shop=no, by its brand"] = sum(_no_shop(row) for row, _, _ in kept)
    report.kept["OSM name kept verbatim (⚑NH-2)"] = sum(
        bool(row["name"]) and not is_czech(row["name"]) for row, _, _ in kept
    )

    rows = [row for row, _, _ in kept]
    points = _points(rows)
    station = np.array([_fuel(row) or (chain is not None and chain.is_station) for row, chain, _ in kept], dtype=bool)
    streets = place.street_types(points, station, context)
    classes = []
    for (row, chain, _), street_type in zip(kept, streets.types, strict=True):
        store_class = place.store_class(
            shop=row["shop"], amenity=row["amenity"], name=row["name"], chain=chain, street_type=street_type
        )
        classes.append(store_class)
    street_types = [
        place.settle(street_type, store_class) for street_type, store_class in zip(streets.types, classes, strict=True)
    ]
    weeks = _weeks(
        [row["opening_hours"] for row in rows],
        [row["osm_id"] for row in rows],
        classes,
        [chain.code if chain else INDEPENDENT for _, chain, _ in kept],
        report,
    )

    stores = []
    for i, (row, chain, code) in enumerate(kept):
        town = _town(code, row["city"], context.settlement_names[streets.settlement[i]], okres_names)
        store = _store_row(row, chain, code, classes[i], street_types[i], weeks[i], town, area_of, region_of)
        stores.append(store)
        report.by_class_region[store["store_class"], store["region"]] += 1

    store_ids = {store["osm_id"] for store in stores}
    picks, horeca_rows = _horeca(root, store_ids, leaves, context, okres_names, area_of, region_of, report)
    stores += horeca_rows
    used = {store["chain"] for store in stores if store["chain"]}
    chain_rows: list[Row] = [
        {"code": c.code, "name": c.name, "kind": c.kind, "is_key_account": c.is_key_account}
        for c in chains.CHAINS
        if c.code in used
    ]
    return Market(stores, chain_rows, picks, report)


def _points(rows: Sequence[Row]) -> okres.Points:
    return np.array([(row["lon"], row["lat"]) for row in rows], dtype=float).reshape(-1, 2)


def _fuel(row: Row) -> bool:
    return bool(row["amenity"] == "fuel")


def _no_shop(row: Row) -> bool:
    """A station that says it has no shop (`shop=no`)."""
    return _fuel(row) and row["shop"] == "no"


def _without_duplicates(
    kept: list[tuple[Row, Chain | None, str]], report: Report
) -> list[tuple[Row, Chain | None, str]]:
    """Drop a station chain's shop beside its own pumps, and an object mapped twice (same chain, brand or name). The
    pumps are a chain's by their brand or by their operator: Robin Oil's pumps are no EuroOil station, but the shop
    that Čepro runs beside them is theirs."""
    points = _points([row for row, _, _ in kept])
    stations: dict[str, list[int]] = defaultdict(list)
    for i, (row, chain, _) in enumerate(kept):
        if _fuel(row):
            run_by = chains.match(operator=row["operator"], fuel=True)
            for code in {c.code for c in (chain, run_by) if c is not None}:
                stations[code].append(i)
    drop: set[int] = set()
    for i, (row, chain, _) in enumerate(kept):
        if chain is not None and chain.is_station and not _fuel(row) and stations[chain.code]:
            distance, _ = place.nearest(points[stations[chain.code]], points[i : i + 1])
            if distance[0] <= STATION_SHOP_M:
                drop.add(i)
                report.dropped["station's own shop"] += 1
    by_chain = [_network(row, chain) if i not in drop else None for i, (row, chain, _) in enumerate(kept)]
    by_name = [
        chains.fold(row["name"]) if chain is None and i not in drop else None for i, (row, chain, _) in enumerate(kept)
    ]
    twice = place.mapped_twice(by_chain, points, SAME_CHAIN_M) | place.mapped_twice(by_name, points, SAME_NAME_M)
    report.dropped["mapped twice"] += len(twice)
    return [item for i, item in enumerate(kept) if i not in drop | twice]


def _network(row: Row, chain: Chain | None) -> str | None:
    """The network an object belongs to, for the duplicate rule: its chain, or a brand that names no chain (an
    independent network, such as Robin Oil); a station and a shop of one network are apart."""
    network = chain.code if chain is not None else chains.fold(row["brand"])
    return network + ("/fuel" if _fuel(row) else "") if network else None


def _weeks(
    values: Sequence[str], osm_ids: Sequence[str], classes: Sequence[str], groups: Sequence[str], report: Report
) -> list[Week]:
    """Each store's week: its own tagged hours, or those of a tagged store drawn by `osm_order` from its class and
    `groups` entry (a chain, or the independents) — from its class alone when that pool is under `HOURS_POOL`."""
    parsed = [parse(value) for value in values]
    by_class: dict[str, list[Week]] = defaultdict(list)
    by_group: dict[tuple[str, str], list[Week]] = defaultdict(list)
    for week, store_class, group in zip(parsed, classes, groups, strict=True):
        if week is not None:
            by_class[store_class].append(week)
            by_group[store_class, group].append(week)
    weeks = []
    for week, osm_id, store_class, group in zip(parsed, osm_ids, classes, groups, strict=True):
        if week is None:
            pool, which = by_group[store_class, group], "class and chain"
            if len(pool) < HOURS_POOL:
                pool, which = by_class[store_class] or [w for ws in by_class.values() for w in ws], "class"
            if not pool:
                raise ValueError("no readable opening_hours to draw from")
            week = pool[osm_order(osm_id) % len(pool)]
            report.hours_drawn[which] += 1
        weeks.append(week)
    return weeks


def _store_row(
    row: Row,
    chain: Chain | None,
    code: str,
    store_class: str,
    street_type: str,
    week: Week,
    town: str,
    area_of: Mapping[str, str],
    region_of: Mapping[str, str],
) -> Row:
    name = row["name"] or (chain.name if chain else f"{UNNAMED.get(store_class, 'Prodejna')} {town}")
    street = row["street"] or row["place_name"]
    if street and row["housenumber"]:
        street = f"{street} {row['housenumber']}"
    return {
        "osm_id": row["osm_id"],
        "name": name,
        "brand": row["brand"] or None,
        "chain": chain.code if chain else None,
        "store_class": store_class,
        "size_class": place.SIZE_OF.get(store_class, "S"),
        "street_type": street_type,
        "weekly_hours": week.weekly_hours,
        "sunday_open": week.sunday_open,
        "lat": row["lat"],
        "lon": row["lon"],
        "street": street or None,
        "town": town,
        "postcode": row["postcode"] or None,
        "okres": code,
        "area": area_of[code],
        "region": region_of[area_of[code]],
        "opened_on": None,
        "closed_on": None,
    }


def _horeca(
    root: Path,
    store_ids: set[str],
    leaves: Sequence[okres.Leaf],
    context: place.Context,
    okres_names: Mapping[str, str],
    area_of: Mapping[str, str],
    region_of: Mapping[str, str],
    report: Report,
) -> tuple[list[Pick], list[Row]]:
    """The HoReCa sample's rows, from the places that are not already a store (`store_ids`, the OSM ids)."""
    places = [row for row in read("osm-horeca", root) if row["osm_id"] not in store_ids]
    points = _points(places)
    assignment = okres.assign(leaves, points)
    population = {row["code"]: row["population"] for row in read("okres", root)}
    index = affluence(root)
    weight = {code: population[code] * index[code] for code in area_of}
    picks = sample(places, assignment.codes, weight)
    chosen = {pick.osm_id: pick for pick in picks}
    by_id = {row["osm_id"]: (i, row) for i, row in enumerate(places) if row["osm_id"] in chosen}
    ordered = [by_id[pick.osm_id] for pick in picks]
    picked_points = points[[i for i, _ in ordered]]
    streets = place.street_types(picked_points, np.zeros(len(ordered), dtype=bool), context)
    weeks = _weeks(
        [row["opening_hours"] for _, row in ordered],
        [p.osm_id for p in picks],
        ["horeca"] * len(picks),
        [INDEPENDENT] * len(picks),
        report,
    )
    rows = []
    for k, (pick, (_, row)) in enumerate(zip(picks, ordered, strict=True)):
        town = _town(pick.okres, row["city"], context.settlement_names[streets.settlement[k]], okres_names)
        store = _store_row(
            row | {"shop": "", "operator": ""},
            None,
            pick.okres,
            "horeca",
            streets.types[k],
            weeks[k],
            town,
            area_of,
            region_of,
        )
        rows.append(store)
        report.by_class_region["horeca", store["region"]] += 1
    return picks, rows


def _town(code: str, city: str, nearest: str, okres_names: Mapping[str, str]) -> str:
    """Praha for a Prague district; the city for a city okres (`Brno-město` → Brno) — its edge is a village's;
    else the address's town, or the nearest settlement."""
    if code.startswith(PRAHA):
        return "Praha"
    if okres_names[code].endswith("-město"):
        return okres_names[code].removesuffix("-město")
    return city or nearest


def main() -> int:
    market = build()
    for line in market.report.lines():
        print(line)
    print(f"stores {len(market.stores)} · chains {len(market.chains)} · HoReCa picks {len(market.horeca)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
