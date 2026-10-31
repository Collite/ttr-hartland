"""`just osm-extract` (`05` §8): the Overpass queries of `data/reference/osm/` → the committed OSM snapshot.

| query | → | what |
|---|---|---|
| `stores.overpassql` | `osm-stores.csv` | the grocery store set: food shops and fuel stations (⚑NH-2) |
| `horeca.overpassql` | `osm-horeca.csv` | restaurants, cafés, bars, pubs, fast food, canteens (the HoReCa sample) |
| `context.overpassql` | `osm-context.csv` | stations, malls, settlement points (what `street_type` is read from) |
| `okres-boundaries.overpassql` | `osm/okres-boundaries.geojson` | the 98 leaves of `okres.csv` as polygons |

plus `osm/osm-extract.meta`: when, from which endpoint, OSM's own timestamp, the row counts. Overpass's CSV is
tab-separated and unquoted; it is rewritten as the convention's CSV (`io.reference`), each cell stripped and NFC. The
boundaries are assembled from their member ways and simplified (Douglas–Peucker, `TOLERANCE`), so the file stays
small; `setup.classifier.okres` absorbs the slivers that leaves between neighbours. Data © OpenStreetMap contributors,
ODbL 1.0 (`LICENSE-DATA.md`).

Run once, on a machine that reaches Overpass; the result is committed. `scripts/osm_extract_pbf.sh` is the fallback
from Geofabrik's `.osm.pbf` (same CSV columns).
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from hartland_sim.io.reference import REFERENCE_DIR

ENDPOINT = "https://overpass-api.de/api/interpreter"
USER_AGENT = "ttr-hartland-osm-extract/1 (+https://github.com/Collite/ttr-hartland)"
ATTEMPTS = 6
FIRST_WAIT_S = 15.0
RETRY_STATUS = frozenset({429, 502, 503, 504})

TOLERANCE = 0.0002  # degrees, ≈ 15–22 m in Czechia
PRAHA_DISTRICT = re.compile(r"SO Praha (\d+)")

# Overpass's column names → the CSV's.
RENAME = {
    "::lat": "lat",
    "::lon": "lon",
    "addr:street": "street",
    "addr:place": "place_name",
    "addr:housenumber": "housenumber",
    "addr:city": "city",
    "addr:postcode": "postcode",
}


@dataclass(frozen=True)
class Extract:
    query: str  # in data/reference/osm/
    output: str  # relative to data/reference/
    keep: Callable[[str, Mapping[str, str]], bool]  # the query's filter, for the `.osm.pbf` fallback: (osm type, tags)


STORE_SHOPS = frozenset({"supermarket", "convenience", "grocery", "general", "greengrocer", "deli"})
HORECA = frozenset({"restaurant", "cafe", "bar", "pub", "fast_food", "canteen"})
PLACES = frozenset({"city", "town", "village", "suburb", "quarter"})


def _is_store(kind: str, tags: Mapping[str, str]) -> bool:
    return (
        tags.get("shop") in STORE_SHOPS
        or (tags.get("amenity") == "fuel" and bool(tags.get("shop") or tags.get("brand")))
        or (tags.get("shop") == "wholesale" and tags.get("brand", "").lower() == "makro")
    )


def _is_horeca(kind: str, tags: Mapping[str, str]) -> bool:
    return tags.get("amenity") in HORECA


def _is_context(kind: str, tags: Mapping[str, str]) -> bool:
    return (
        tags.get("railway") in ("station", "halt")
        or tags.get("amenity") == "bus_station"
        or tags.get("shop") == "mall"
        or (kind == "node" and tags.get("place") in PLACES)
    )


def _is_leaf(kind: str, tags: Mapping[str, str]) -> bool:
    return kind == "relation" and tags.get("boundary") == "administrative" and leaf_code(tags) is not None


EXTRACTS = (
    Extract("stores.overpassql", "osm-stores.csv", _is_store),
    Extract("horeca.overpassql", "osm-horeca.csv", _is_horeca),
    Extract("context.overpassql", "osm-context.csv", _is_context),
    Extract("okres-boundaries.overpassql", "osm/okres-boundaries.geojson", _is_leaf),
)

Post = Callable[[str], bytes]


class OverpassError(RuntimeError):
    """Overpass answered with an error, or kept refusing after `ATTEMPTS` tries."""


def post(query: str, endpoint: str = ENDPOINT, *, attempts: int = ATTEMPTS, wait: float = FIRST_WAIT_S) -> bytes:
    """POST one query; retries a busy server (429, 502–504, a timeout) with doubling waits."""
    data = urllib.parse.urlencode({"data": query}).encode()
    request = urllib.request.Request(endpoint, data=data, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                body: bytes = response.read()
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUS or attempt == attempts:
                raise OverpassError(f"{endpoint}: HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts:
                raise OverpassError(f"{endpoint}: {error}") from error
        else:
            if b"runtime error" in body[-2000:] or b"runtime error" in body[:2000]:
                raise OverpassError(f"{endpoint}: {body[:300].decode(errors='replace')}")
            return body
        print(f"osm-extract: busy, retrying in {wait:.0f} s ({attempt}/{attempts})", file=sys.stderr)
        time.sleep(wait)
        wait *= 2
    raise AssertionError("unreachable")


def query_columns(query: str) -> list[str]:
    """The CSV columns a `[out:csv(…)]` query writes: `osm_id`, then its fields renamed (`addr:street` → `street`)."""
    header = re.search(r"\[out:csv\((.*?)\)\]", query, re.DOTALL)
    if header is None:
        raise ValueError("not an [out:csv(…)] query")
    fields = [field.strip().strip('"') for field in header.group(1).split(",")]
    if fields[:2] != ["::type", "::id"]:
        raise ValueError(f"the fields start {fields[:2]}, not ::type, ::id")
    return ["osm_id"] + [RENAME.get(field, field) for field in fields[2:]]


def clean(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def tsv_rows(text: str) -> tuple[list[str], list[dict[str, str]], int]:
    """Overpass's `[out:csv]` text → the CSV's columns, its rows, and how many lines were dropped (the wrong number of
    cells). Overpass quotes a value with a comma or a quote in it, as CSV does, with a tab between the values.
    `::type` + `::id` become one `osm_id`, as OSM writes it: `node/123`."""
    lines = csv.reader(io.StringIO(text, newline=""), delimiter="\t")
    header = next(lines, [])
    if header[:2] != ["@type", "@id"]:
        raise OverpassError(f"unexpected header {header[:4]}")
    columns = ["osm_id"] + [RENAME.get(name.replace("@", "::"), name) for name in header[2:]]
    rows, dropped = [], 0
    for cells in lines:
        if len(cells) != len(header):
            dropped += 1
            continue
        row = dict(zip(columns[1:], (clean(cell) for cell in cells[2:]), strict=True))
        rows.append({"osm_id": f"{cells[0]}/{cells[1]}"} | row)
    rows.sort(key=lambda row: _osm_order(row["osm_id"]))
    return columns, rows, dropped


def _osm_order(osm_id: str) -> tuple[str, int]:
    kind, _, number = osm_id.partition("/")
    return kind, int(number)


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ── the boundaries ──────────────────────────────────────────────────────────


Ring = list[tuple[float, float]]  # (lon, lat), closed: first == last


def leaf_code(tags: Mapping[str, str]) -> str | None:
    """The `okres.csv` code a boundary relation stands for: an okres's LAU 1 `ref`, or `CZ0100-NN` for SO Praha NN."""
    district = PRAHA_DISTRICT.fullmatch(tags.get("name", ""))
    if district:
        return f"CZ0100-{int(district.group(1)):02d}"
    ref = tags.get("ref", "")
    return ref if ref.startswith("CZ0") and ref != "CZ0100" else None


def assemble(ways: Sequence[Sequence[tuple[float, float]]]) -> list[Ring]:
    """Join a relation's member ways end to end into closed rings; `OverpassError` on a ring that does not close."""
    pending = [list(way) for way in ways if len(way) >= 2]
    rings: list[Ring] = []
    while pending:
        ring = pending.pop()
        while ring[0] != ring[-1]:
            for i, way in enumerate(pending):
                if way[0] == ring[-1]:
                    ring += way[1:]
                elif way[-1] == ring[-1]:
                    ring += way[-2::-1]
                else:
                    continue
                del pending[i]
                break
            else:
                raise OverpassError(f"a ring does not close at {ring[-1]}")
        rings.append(ring)
    return rings


def simplify(ring: Ring, tolerance: float = TOLERANCE) -> Ring:
    """Douglas–Peucker on a closed ring; keeps at least four points (a triangle, closed)."""
    points = np.asarray(ring, dtype=float)
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    far = int(np.argmax(np.hypot(*(points - points[0]).T)))  # split the ring at its farthest point from the start
    keep[far] = True
    stack = [(0, far), (far, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        start, end = points[first], points[last]
        span = end - start
        inner = points[first + 1 : last] - start
        length = float(np.hypot(*span))
        if length == 0:
            distance = np.hypot(*inner.T)
        else:
            distance = np.abs(span[0] * inner[:, 1] - span[1] * inner[:, 0]) / length
        index = int(np.argmax(distance))
        if distance[index] > tolerance:
            middle = first + 1 + index
            keep[middle] = True
            stack += [(first, middle), (middle, last)]
    kept = [ring[i] for i in np.flatnonzero(keep)]
    return kept if len(kept) >= 4 else list(ring)


def boundaries(overpass: Mapping[str, Any], tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Overpass's `out geom` JSON of the boundary relations → a GeoJSON FeatureCollection, one MultiPolygon per leaf
    (`code`, `name`), its rings simplified. Holes go with the outer ring that holds them; the okres classifier reads
    the rings even–odd, so a hole is outside."""
    features: list[dict[str, Any]] = []
    for element in overpass["elements"]:
        code = leaf_code(element.get("tags", {}))
        if element.get("type") != "relation" or code is None:
            continue
        by_role: dict[str, list[list[tuple[float, float]]]] = {"outer": [], "inner": []}
        for member in element["members"]:
            if member["type"] == "way" and member.get("role") in by_role and "geometry" in member:
                by_role[member["role"]].append([(point["lon"], point["lat"]) for point in member["geometry"]])
        outers = [_rounded(simplify(ring, tolerance)) for ring in assemble(by_role["outer"])]
        polygons: list[list[Ring]] = [[outer] for outer in outers]
        for inner in assemble(by_role["inner"]):
            holder = next((polygon for polygon in polygons if _holds(polygon[0], inner[0])), None)
            if holder is not None:
                holder.append(_rounded(simplify(inner, tolerance)))
        features.append(
            {
                "type": "Feature",
                "properties": {"code": code, "name": element["tags"]["name"], "osm_id": f"relation/{element['id']}"},
                "geometry": {"type": "MultiPolygon", "coordinates": polygons},
            }
        )
    features.sort(key=lambda feature: feature["properties"]["code"])
    return {"type": "FeatureCollection", "features": features}


def _holds(ring: Ring, point: tuple[float, float]) -> bool:
    """Whether `point` is inside `ring` (a ray cast; for placing a hole only)."""
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:], strict=False):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _rounded(ring: Ring) -> Ring:
    return [(round(lon, 6), round(lat, 6)) for lon, lat in ring]


# ── the fallback: Geofabrik's .osm.pbf through osmium ──────────────────────

OSMIUM_TYPES = {"n": "node", "w": "way", "r": "relation", "a": "area"}


def pbf_rows(lines: Iterable[str], extract: Extract, columns: Sequence[str]) -> list[dict[str, str]]:
    """`osmium export -f geojsonseq --add-unique-id=type_id` features → the rows Overpass would give: the extract's
    filter, the centre of the bounding box (Overpass's `out center`), the same columns."""
    tag_of = {column: tag for tag, column in RENAME.items()}
    rows = []
    for line in lines:
        line = line.strip().lstrip("\x1e")
        if not line:
            continue
        feature = json.loads(line)
        kind = OSMIUM_TYPES[feature["id"][0]]
        tags = feature.get("properties") or {}
        if not extract.keep(kind, tags):
            continue
        lon, lat = _centre(feature["geometry"])
        row = {"osm_id": f"{kind}/{feature['id'][1:]}", "lat": f"{lat:.7f}", "lon": f"{lon:.7f}"}
        row |= {column: clean(str(tags.get(tag_of.get(column, column), ""))) for column in columns[3:]}
        rows.append(row)
    rows.sort(key=lambda row: _osm_order(row["osm_id"]))
    return rows


def _centre(geometry: Mapping[str, Any]) -> tuple[float, float]:
    points = np.asarray(list(_points(geometry["coordinates"])), dtype=float)
    low, high = points.min(axis=0), points.max(axis=0)
    return float((low[0] + high[0]) / 2), float((low[1] + high[1]) / 2)


def _points(coordinates: Any) -> Iterable[tuple[float, float]]:
    if isinstance(coordinates[0], (int, float)):
        yield coordinates[0], coordinates[1]
        return
    for part in coordinates:
        yield from _points(part)


def pbf_boundaries(lines: Iterable[str], tolerance: float = TOLERANCE) -> dict[str, Any]:
    """osmium's boundary areas → the same FeatureCollection as `boundaries()` (osmium has assembled the rings)."""
    features: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip().lstrip("\x1e")
        if not line:
            continue
        feature = json.loads(line)
        tags = feature.get("properties") or {}
        code = leaf_code(tags)
        if feature["id"][0] != "r" or tags.get("boundary") != "administrative" or code is None:
            continue
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        coordinates = [
            [_rounded(simplify([tuple(p) for p in ring], tolerance)) for ring in polygon] for polygon in polygons
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {"code": code, "name": tags["name"], "osm_id": f"relation/{feature['id'][1:]}"},
                "geometry": {"type": "MultiPolygon", "coordinates": coordinates},
            }
        )
    features.sort(key=lambda feature: feature["properties"]["code"])
    return {"type": "FeatureCollection", "features": features}


# ── the run ─────────────────────────────────────────────────────────────────


def run(
    root: Path = REFERENCE_DIR,
    *,
    endpoint: str = ENDPOINT,
    only: Sequence[str] = (),
    fetch: Post | None = None,
    pbf_dir: Path | None = None,
) -> dict[str, int]:
    """Every extract (or the `only` ones, by output name) → its file under `root`; then the meta file. Returns the row
    (or feature) count per output. With `pbf_dir`, reads `scripts/osm_extract_pbf.sh`'s `<stem>.geojsonseq` files
    there instead of asking Overpass."""
    send: Post = fetch or (lambda query: post(query, endpoint))
    counts: dict[str, int] = {}
    osm_base = None
    for extract in EXTRACTS:
        if only and extract.output not in only:
            continue
        query = (root / "osm" / extract.query).read_text(encoding="utf-8")
        target = root / extract.output
        if pbf_dir is not None:
            counts[extract.output] = _from_pbf(
                pbf_dir / f"{Path(extract.query).stem}.geojsonseq", extract, query, target
            )
            continue
        body = send(query).decode("utf-8")
        if extract.output.endswith(".geojson"):
            answer = json.loads(body)
            osm_base = answer.get("osm3s", {}).get("timestamp_osm_base")
            collection = boundaries(answer)
            target.write_text(json.dumps(collection, ensure_ascii=False, separators=(",", ":")) + "\n", "utf-8")
            counts[extract.output] = len(collection["features"])
        else:
            columns, rows, dropped = tsv_rows(body)
            if columns != query_columns(query):
                raise OverpassError(f"{extract.output}: the answer's columns {columns} are not the query's")
            write_csv(target, columns, rows)
            counts[extract.output] = len(rows)
            if dropped:
                print(f"osm-extract: {extract.output}: {dropped} lines dropped (a tab in a value)", file=sys.stderr)
        print(f"osm-extract: {extract.output}: {counts[extract.output]}", file=sys.stderr)
    source = f"{pbf_dir} (osmium, Geofabrik)" if pbf_dir is not None else endpoint
    write_meta(root / "osm" / "osm-extract.meta", endpoint=source, osm_base=osm_base, counts=counts)
    return counts


def _from_pbf(source: Path, extract: Extract, query: str, target: Path) -> int:
    with source.open(encoding="utf-8") as lines:
        if target.suffix == ".geojson":
            collection = pbf_boundaries(lines)
            target.write_text(json.dumps(collection, ensure_ascii=False, separators=(",", ":")) + "\n", "utf-8")
            return len(collection["features"])
        columns = query_columns(query)
        rows = pbf_rows(lines, extract, columns)
    write_csv(target, columns, rows)
    return len(rows)


def write_meta(path: Path, *, endpoint: str, osm_base: str | None, counts: Mapping[str, int]) -> None:
    lines = [
        f"extracted: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"endpoint: {endpoint}",
        f"osm_base: {osm_base or 'unknown'}",
        "licence: ODbL 1.0, © OpenStreetMap contributors",
        "counts:",
        *(f"  {name}: {count}" for name, count in counts.items()),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
