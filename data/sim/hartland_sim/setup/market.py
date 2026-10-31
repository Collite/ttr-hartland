"""The market (G0 S0.3 T6; `05` §2 step 2): the OSM extract → the `store` and `chain` rows of `03` §3, and the store
set report. Run `uv run python -m hartland_sim.setup.market` for the report.

From `osm-stores.csv`, in order:
1. **the okres** of each object (`classifier.okres`); one outside every leaf is dropped;
2. **the chain** (`classifier.chains`);
3. **NH-D8**: an object whose name fails the Czech guard is dropped — the OSM names are kept verbatim (⚑NH-2), so
   the world takes the store only when its name reads as Czech;
4. **duplicates**: a station chain's shop or café mapped beside its own pumps (within `STATION_SHOP_M`) is the
   station's shop, not a second store; the same chain within `SAME_CHAIN_M`, or the same name within `SAME_NAME_M`,
   is one store mapped twice. The first in OSM order stays;
5. **the place** (`classifier.place`): street type, store class (`settle`d), size class;
6. **the hours** (`classifier.hours`); untagged or unreadable hours take those of a tagged store of the same class,
   chosen by the OSM id's hash (`01` §7: "assigned by a seeded classifier where tags are missing");
7. **the rest of the row**: the name (OSM's; else the chain's; else the class's Czech word and the town), the town
   (Praha for a Prague district; else `addr:city`; else the nearest settlement), the street, the area and region.

Then the HoReCa sample (`setup.horeca`) joins as `store_class = horeca`. Station split (NH-D5): an OMV or Shell
station belongs to a key-account chain; every other station is `petrol` with its own chain or none, a small-retail
customer at G1. The rows are keyed by `osm_id`, the business key; their `bigint` ids come with `ids.py` (G1 S1.1).
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hartland_sim.checks.czech import assert_czech
from hartland_sim.io.reference import REFERENCE_DIR, Row, read
from hartland_sim.setup.affluence import load as affluence
from hartland_sim.setup.classifier import chains, okres, place
from hartland_sim.setup.classifier.chains import Chain
from hartland_sim.setup.classifier.hours import Week, parse
from hartland_sim.setup.horeca import Pick, sample

STATION_SHOP_M = 150.0
SAME_CHAIN_M = 50.0
SAME_NAME_M = 30.0
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
    snapped: int = 0
    hours_drawn: int = 0
    by_class_region: Counter[tuple[str, str]] = field(default_factory=Counter)

    def lines(self) -> list[str]:
        regions = sorted({region for _, region in self.by_class_region})
        classes = [c for c in (*place.STORE_CLASSES, "horeca") if any(k[0] == c for k in self.by_class_region)]
        out = [
            f"read {self.read} OSM objects; dropped {dict(self.dropped)}; {self.snapped} snapped to an okres; "
            f"{self.hours_drawn} stores' hours drawn from their class"
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
    context = place.Context.load(root / "osm-context.csv")
    settlement_names = _settlement_names(root / "osm-context.csv")
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
        elif row["name"] and not _czech(row["name"]):
            report.dropped["name not Czech"] += 1
        else:
            kept.append((row, chain, code))
    kept = _without_duplicates(kept, report)

    rows = [row for row, _, _ in kept]
    points = _points(rows)
    station = np.array([_fuel(row) or (chain is not None and chain.is_station) for row, chain, _ in kept], dtype=bool)
    streets = place.street_types(points, station, context)
    _, nearest_settlement = place.nearest(context.settlements, points)
    classes = []
    for (row, chain, _), street_type in zip(kept, streets, strict=True):
        store_class = place.store_class(
            shop=row["shop"], amenity=row["amenity"], name=row["name"], chain=chain, street_type=street_type
        )
        classes.append(store_class)
    streets = [
        place.settle(street_type, store_class) for street_type, store_class in zip(streets, classes, strict=True)
    ]
    weeks = _weeks([row["opening_hours"] for row in rows], [row["osm_id"] for row in rows], classes, report)

    stores = []
    for i, (row, chain, code) in enumerate(kept):
        town = _town(code, row["city"], settlement_names[int(nearest_settlement[i])], okres_names)
        store = _store_row(row, chain, code, classes[i], streets[i], weeks[i], town, area_of, region_of)
        stores.append(store)
        report.by_class_region[store["store_class"], store["region"]] += 1

    picks, horeca_rows = _horeca(root, leaves, context, settlement_names, okres_names, area_of, region_of, report)
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


def _czech(name: str) -> bool:
    try:
        assert_czech(name)
    except AssertionError:
        return False
    return True


def _without_duplicates(
    kept: list[tuple[Row, Chain | None, str]], report: Report
) -> list[tuple[Row, Chain | None, str]]:
    """Drop a station chain's shop beside its own pumps, and an object mapped twice (same chain or same name)."""
    points = _points([row for row, _, _ in kept])
    stations: dict[str, list[int]] = defaultdict(list)
    for i, (row, chain, _) in enumerate(kept):
        if _fuel(row) and chain is not None:
            stations[chain.code].append(i)
    drop: set[int] = set()
    for i, (row, chain, _) in enumerate(kept):
        if chain is not None and chain.is_station and not _fuel(row) and stations[chain.code]:
            distance, _ = place.nearest(points[stations[chain.code]], points[i : i + 1])
            if distance[0] <= STATION_SHOP_M:
                drop.add(i)
                report.dropped["station's own shop"] += 1
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (row, chain, _) in enumerate(kept):
        if i in drop:
            continue
        if chain is not None:
            groups["chain", chain.code + ("/fuel" if _fuel(row) else "")].append(i)
        elif row["name"]:
            groups["name", chains.fold(row["name"])].append(i)
    for (kind, _), members in groups.items():
        radius = SAME_CHAIN_M if kind == "chain" else SAME_NAME_M
        for position, i in enumerate(members[1:], start=1):
            earlier = [j for j in members[:position] if j not in drop]
            if earlier and place.nearest(points[earlier], points[i : i + 1])[0][0] <= radius:
                drop.add(i)
                report.dropped["mapped twice"] += 1
    return [item for i, item in enumerate(kept) if i not in drop]


def _weeks(values: Sequence[str], osm_ids: Sequence[str], classes: Sequence[str], report: Report) -> list[Week]:
    parsed = [parse(value) for value in values]
    tagged: dict[str, list[Week]] = defaultdict(list)
    for week, store_class in zip(parsed, classes, strict=True):
        if week is not None:
            tagged[store_class].append(week)
    weeks = []
    for week, osm_id, store_class in zip(parsed, osm_ids, classes, strict=True):
        if week is None:
            pool = tagged[store_class] or [w for ws in tagged.values() for w in ws]
            week = pool[int(hashlib.blake2b(osm_id.encode(), digest_size=8).hexdigest(), 16) % len(pool)]
            report.hours_drawn += 1
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
    leaves: Sequence[okres.Leaf],
    context: place.Context,
    settlement_names: Sequence[str],
    okres_names: Mapping[str, str],
    area_of: Mapping[str, str],
    region_of: Mapping[str, str],
    report: Report,
) -> tuple[list[Pick], list[Row]]:
    places = read("osm-horeca", root)
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
    _, nearest_settlement = place.nearest(context.settlements, picked_points)
    weeks = _weeks(
        [row["opening_hours"] for _, row in ordered], [p.osm_id for p in picks], ["horeca"] * len(picks), report
    )
    rows = []
    for k, (pick, (_, row)) in enumerate(zip(picks, ordered, strict=True)):
        town = _town(pick.okres, row["city"], settlement_names[int(nearest_settlement[k])], okres_names)
        store = _store_row(
            row | {"shop": "", "operator": ""},
            None,
            pick.okres,
            "horeca",
            streets[k],
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


def _settlement_names(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as file:
        return [row["name"] for row in csv.DictReader(file) if row["place"] in ("city", "town", "village")]


def main() -> int:
    market = build()
    for line in market.report.lines():
        print(line)
    print(f"stores {len(market.stores)} · chains {len(market.chains)} · HoReCa picks {len(market.horeca)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
