"""G0 S0.3 T1 — the Overpass queries and `just osm-extract`, against captured responses (no network in tests)."""

from __future__ import annotations

import csv
import http.client
import json
import shutil
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.io.osm_extract import (
    BASE_QUERY,
    EXTRACTS,
    META,
    OverpassError,
    Provenance,
    _is_store,
    assemble,
    boundaries,
    leaf_code,
    okres_leaves,
    osmium_object,
    pbf_boundaries,
    pbf_rows,
    post,
    query_columns,
    read_meta,
    run,
    simplify,
    tsv_rows,
)
from hartland_sim.io.osm_names import ENDPOINT, OUTPUTS

STORES_HEADER = (
    "@type\t@id\t@lat\t@lon\tname\tbrand\toperator\tshop\tamenity\taddr:street\taddr:place\taddr:housenumber\t"
    "addr:city\taddr:postcode\topening_hours"
)
STORES_TSV = (
    "\n".join(
        [
            STORES_HEADER,
            "way\t42\t50.0763000\t14.4280000\tAlbert\tAlbert\t\tsupermarket\t\tVinohradská\t\t1/2\tPraha\t120 00\t"
            "Mo-Su 07:00-22:00",
            "node\t7\t49.1950000\t16.6080000\t Žabka \tŽabka\t\tconvenience\t\t\t\t\tBrno\t\t",
            "node\t9\t49.0\t17.0\tBenzina\tOrlen\t\t\tfuel\t\t\t\t\t\t",
            "node\t11\t49.5\t15.5\tSmíšené\tzboží\t\t\tgeneral\t\t\tLhota\t12\tLhota\t\t",  # an unquoted tab: dropped
            "node\t12\t50.1\t15.0\tPotraviny U Koni\u0301c\u030cku\t\t\tconvenience\t\t\t\t\t\t\t",  # NFD
            'node\t13\t49.9\t14.0\t"Potraviny ""U Lípy"""\tCOOP TIP\t"COOP Hořovice, družstvo"\tsupermarket'
            "\t\t\t\t\t\t\t",
            'node\t14\t49.8\t14.1\t"Smíšené\tzboží"\t\t\tgeneral\t\t\t\t\t\t\t',  # quoted, so the tab is kept
        ]
    )
    + "\n"
)


def test_the_stores_answer_becomes_the_documented_columns() -> None:
    columns, rows, dropped = tsv_rows(STORES_TSV)
    assert columns == [
        "osm_id",
        "lat",
        "lon",
        "name",
        "brand",
        "operator",
        "shop",
        "amenity",
        "street",
        "place_name",
        "housenumber",
        "city",
        "postcode",
        "opening_hours",
    ]
    assert [row["osm_id"] for row in rows] == ["node/7", "node/9", "node/12", "node/13", "node/14", "way/42"]
    assert dropped == 1
    assert rows[0]["name"] == "Žabka"  # stripped
    assert rows[2]["name"] == "Potraviny U Kon\u00ed\u010dku"  # NFC
    assert (rows[3]["name"], rows[3]["operator"]) == ('Potraviny "U Lípy"', "COOP Hořovice, družstvo")
    assert rows[4]["name"] == "Smíšené\tzboží"
    assert rows[5]["street"] == "Vinohradská" and rows[5]["postcode"] == "120 00"


def test_an_unexpected_answer_is_an_error() -> None:
    with pytest.raises(OverpassError, match="unexpected header"):
        tsv_rows("<html>busy</html>\n")


def square(x: float, y: float, size: float) -> list[tuple[float, float]]:
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]


def test_ways_join_into_rings_whatever_their_direction() -> None:
    ring = square(14.0, 50.0, 1.0)
    halves = [ring[:3], ring[2:][::-1]]  # the second way runs backwards
    [joined] = assemble(halves)
    assert joined[0] == joined[-1]
    assert sorted(set(joined)) == sorted(set(ring))


def test_a_ring_that_does_not_close_is_an_error() -> None:
    with pytest.raises(OverpassError, match="does not close"):
        assemble([[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]])


def test_simplify_drops_the_points_on_a_line_and_keeps_the_corners() -> None:
    ring = [(14.0 + i / 100, 50.0) for i in range(101)] + [(15.0, 51.0), (14.0, 51.0), (14.0, 50.0)]
    assert simplify(ring) == [(14.0, 50.0), (15.0, 50.0), (15.0, 51.0), (14.0, 51.0), (14.0, 50.0)]


@pytest.mark.parametrize(
    ("tags", "code"),
    [
        ({"name": "okres Benešov", "ref": "CZ0201", "admin_level": "5"}, "CZ0201"),
        ({"name": "okres Praha-západ", "ref": "CZ020A", "admin_level": "5"}, "CZ020A"),
        ({"name": "SO Praha 7", "admin_level": "6"}, "CZ0100-07"),
        ({"name": "SO Praha 22", "admin_level": "6"}, "CZ0100-22"),
        ({"name": "Hlavní město Praha", "ref": "CZ0100", "admin_level": "5"}, None),
        ({"name": "Landkreis Cham", "ref": "DE", "admin_level": "6"}, None),
        ({"name": "Středočeský kraj", "ref": "CZ020", "admin_level": "6"}, None),  # a kraj's ref is no okres
        ({"name": "Středočeský kraj", "ref": "CZ020", "admin_level": "4"}, None),
        ({"name": "okres Benešov", "ref": "CZ0201"}, None),  # the level is the query's: 5 or 6
        ({"name": "SO Praha 7", "admin_level": "7"}, None),
    ],
)
def test_a_relation_names_its_leaf(tags: dict[str, str], code: str | None) -> None:
    assert leaf_code(tags) == code


Points = list[tuple[float, float]]


def relation(rid: int, tags: dict[str, str], outers: Sequence[Points], inners: Sequence[Points] = ()) -> Any:
    def way(points: Points, role: str) -> dict[str, Any]:
        return {"type": "way", "role": role, "geometry": [{"lon": x, "lat": y} for x, y in points]}

    members = [way(outer, "outer") for outer in outers] + [way(inner, "inner") for inner in inners]
    return {"type": "relation", "id": rid, "tags": tags, "members": members}


BASE = "2026-10-31T12:00:00Z"


def overpass_boundaries() -> dict[str, Any]:
    benesov = {"name": "okres Benešov", "ref": "CZ0201", "admin_level": "5"}
    return {
        "osm3s": {"timestamp_osm_base": BASE},
        "elements": [
            relation(2, {"name": "SO Praha 1", "admin_level": "6"}, [square(14.4, 50.08, 0.02)]),
            relation(1, benesov, [square(14.5, 49.6, 0.4)], [square(14.6, 49.7, 0.1)]),
            relation(3, {"name": "Hlavní město Praha", "ref": "CZ0100", "admin_level": "5"}, [square(14.2, 49.9, 0.6)]),
        ],
    }


def leaf_relations(leaves: set[str]) -> dict[str, Any]:
    """Overpass's answer with a small square for each leaf, at the levels the boundary query asks for."""
    elements = []
    for n, code in enumerate(sorted(leaves)):
        if code.startswith("CZ0100-"):
            tags = {"name": f"SO Praha {int(code[-2:])}", "admin_level": "6"}
        else:
            tags = {"name": f"okres {code}", "ref": code, "admin_level": "5"}
        elements.append(relation(n + 1, tags, [square(12.5 + n * 0.06, 49.0, 0.05)]))
    return {"osm3s": {"timestamp_osm_base": BASE}, "elements": elements}


def test_the_boundaries_become_one_multipolygon_per_leaf() -> None:
    collection = boundaries(overpass_boundaries())
    assert [feature["properties"]["code"] for feature in collection["features"]] == ["CZ0100-01", "CZ0201"]
    benesov = collection["features"][1]
    assert benesov["properties"] == {"code": "CZ0201", "name": "okres Benešov", "osm_id": "relation/1"}
    [[outer, hole]] = benesov["geometry"]["coordinates"]
    assert len(outer) == len(hole) == 5


def test_a_hole_goes_with_the_outer_ring_that_holds_it() -> None:
    tags = {"name": "okres Benešov", "ref": "CZ0201", "admin_level": "5"}
    answer = {
        "elements": [relation(1, tags, [square(14.0, 49.0, 0.1), square(15.0, 49.0, 0.4)], [square(15.1, 49.1, 0.1)])]
    }
    polygons = boundaries(answer)["features"][0]["geometry"]["coordinates"]
    assert sorted((min(polygon[0]), len(polygon)) for polygon in polygons) == [((14.0, 49.0), 1), ((15.0, 49.0), 2)]


def test_a_hole_in_no_outer_ring_is_an_error() -> None:
    tags = {"name": "okres Benešov", "ref": "CZ0201", "admin_level": "5"}
    answer = {"elements": [relation(1, tags, [square(14.0, 49.0, 0.1)], [square(16.0, 49.0, 0.1)])]}
    with pytest.raises(OverpassError, match="relation/1: a hole .* lies in no outer ring"):
        boundaries(answer)


Send = Callable[[str], bytes]


def fake_overpass(queries: list[str], ref: Path, *, fail_at: int = 0, leaves: set[str] | None = None) -> Send:
    """Each CSV query gets four rows under its own header, the base query OSM's timestamp, the boundary query a square
    for each of `leaves` (default: `okres.csv`'s). `fail_at`: the n-th query (from 1) raises, as a refusing server."""

    def send(query: str) -> bytes:
        queries.append(query)
        if len(queries) == fail_at:
            raise OverpassError("HTTP 504")
        if query == BASE_QUERY:
            return json.dumps({"osm3s": {"timestamp_osm_base": BASE}, "elements": []}).encode()
        if "out geom" in query:
            return json.dumps(leaf_relations(okres_leaves(ref) if leaves is None else leaves)).encode()
        fields = query.split("[out:csv(")[1].split(")]")[0].replace("\n", " ").split(",")
        header = [field.strip().strip('"').replace("::", "@") for field in fields]
        rows = [["node", str(n), "50.0", "14.0", "Lhota"] + [""] * (len(header) - 5) for n in range(1, 5)]
        return "".join("\t".join(row) + "\n" for row in [header, *rows]).encode()

    return send


@pytest.fixture
def snapshot(tmp_path: Path, ref: Path) -> Path:
    """A copy of the committed snapshot with what a run reads: the queries, the meta, `okres.csv`."""
    shutil.copytree(ref / "osm", tmp_path / "osm")
    (tmp_path / "schemas").mkdir()
    shutil.copy(ref / "okres.csv", tmp_path)
    shutil.copy(ref / "schemas" / "okres.schema.json", tmp_path / "schemas")
    for output in OUTPUTS[:3]:
        shutil.copy(ref / output, tmp_path / output)
    return tmp_path


def files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_the_verb_writes_every_file_and_the_meta(snapshot: Path, ref: Path) -> None:
    queries: list[str] = []
    counts = run(snapshot, fetch=fake_overpass(queries, ref))
    assert counts == {
        "osm-stores.csv": 4,
        "osm-horeca.csv": 4,
        "osm-context.csv": 4,
        "osm/okres-boundaries.geojson": 98,
    }
    assert len(queries) == len(EXTRACTS) + 3  # each CSV query asks for OSM's timestamp first
    with (snapshot / "osm-stores.csv").open(encoding="utf-8", newline="") as stores:
        assert next(csv.reader(stores))[:4] == ["osm_id", "lat", "lon", "name"]
    meta = read_meta(snapshot / META)
    assert list(meta) == list(OUTPUTS)
    assert all((p.osm_base, p.source) == (BASE, ENDPOINT) for p in meta.values())
    assert meta["osm-stores.csv"].count == 4
    assert not list(snapshot.rglob("*.tmp"))


def test_a_partial_run_keeps_the_other_outputs_lines(snapshot: Path, ref: Path) -> None:
    before = read_meta(snapshot / META)
    run(snapshot, only=["osm-horeca.csv"], fetch=fake_overpass([], ref))
    after = read_meta(snapshot / META)
    assert after["osm-horeca.csv"].count == 4
    assert {k: v for k, v in after.items() if k != "osm-horeca.csv"} == {
        k: v for k, v in before.items() if k != "osm-horeca.csv"
    }
    assert len(after) == 4


@pytest.mark.parametrize("fail_at", [2, 4, 7])
def test_a_failed_run_changes_nothing(snapshot: Path, ref: Path, fail_at: int) -> None:
    before = files(snapshot)
    with pytest.raises(OverpassError):
        run(snapshot, fetch=fake_overpass([], ref, fail_at=fail_at))
    assert files(snapshot) == before


def test_the_boundaries_must_be_okres_csvs_leaves(snapshot: Path, ref: Path) -> None:
    leaves = (okres_leaves(ref) - {"CZ0201"}) | {"CZ0999"}
    before = files(snapshot)
    with pytest.raises(OverpassError, match=r"boundaries: missing \['CZ0201'\], extra \['CZ0999'\], twice \[\]"):
        run(snapshot, fetch=fake_overpass([], ref, leaves=leaves))
    assert files(snapshot) == before


def test_the_committed_meta_records_every_output(ref: Path) -> None:
    meta = read_meta(ref / META)
    assert list(meta) == list(OUTPUTS)
    assert all(p.source == ENDPOINT and p.extracted.startswith("2026-10-31T") for p in meta.values())
    assert meta["osm/okres-boundaries.geojson"] == Provenance(
        98, "2026-10-31T14:05:13Z", "2026-10-31T14:03:51Z", ENDPOINT
    )


def test_the_extracts_are_the_cli_outputs() -> None:
    assert tuple(extract.output for extract in EXTRACTS) == OUTPUTS


@pytest.mark.parametrize(
    ("tags", "store"),
    [
        ({"amenity": "fuel", "shop": "convenience"}, True),
        ({"amenity": "fuel", "shop": "no"}, False),  # OSM's "this station has no shop"
        ({"amenity": "fuel", "shop": "no", "brand": "MOL"}, True),  # a brand still counts (05 §8)
        ({"amenity": "fuel"}, False),
        ({"shop": "wholesale", "brand": "Makro"}, True),
        ({"shop": "kiosk"}, False),
    ],
)
def test_which_objects_are_stores(tags: dict[str, str], store: bool) -> None:
    assert _is_store("node", tags) is store


class _Answer:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


@pytest.mark.parametrize(
    "dropped", [http.client.IncompleteRead(b"node\t1"), ConnectionResetError("reset by peer"), TimeoutError()]
)
def test_a_dropped_download_is_retried(monkeypatch: pytest.MonkeyPatch, dropped: Exception) -> None:
    answers: list[Exception | bytes] = [dropped, b"@type\t@id\n"]

    def urlopen(request: urllib.request.Request, timeout: float) -> _Answer:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _Answer(answer)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    assert post("query", attempts=2) == b"@type\t@id\n"


def test_every_query_asks_czechia_and_carries_the_odbl_line(ref: Path) -> None:
    for extract in EXTRACTS:
        query = (ref / "osm" / extract.query).read_text(encoding="utf-8")
        assert 'area["ISO3166-1"="CZ"]->.cz;' in query, extract.query
        assert "© OpenStreetMap contributors, ODbL 1.0" in query, extract.query


def test_each_csv_query_names_its_columns(ref: Path) -> None:
    stores = (ref / "osm" / "stores.overpassql").read_text(encoding="utf-8")
    assert query_columns(stores) == tsv_rows(STORES_TSV)[0]


def osmium_line(osm: str, geometry: dict[str, Any], **tags: str) -> str:
    return "\x1e" + json.dumps({"type": "Feature", "id": osm, "properties": tags, "geometry": geometry})


@pytest.mark.parametrize(
    ("unique_id", "osm"),
    [("n7", ("node", 7)), ("w42", ("way", 42)), ("a84", ("way", 42)), ("a5", ("relation", 2))],
)
def test_osmiums_ids_name_the_osm_object(unique_id: str, osm: tuple[str, int]) -> None:
    """libosmium's area ids, as `osmium export --add-unique-id=type_id` 1.18 writes them: way 42 as `w42` (the line)
    and `a84` (the area), relation 2 as `a5`."""
    assert osmium_object(unique_id) == osm


def test_the_pbf_fallback_gives_the_same_rows(ref: Path) -> None:
    """Overpass's filter (a fuel station counts with a shop or a brand) and its `out center` (the bounding box's);
    a closed way that osmium writes twice, as a line and as an area, is one row."""
    [stores] = [extract for extract in EXTRACTS if extract.output == "osm-stores.csv"]
    columns = query_columns((ref / "osm" / stores.query).read_text(encoding="utf-8"))
    ring = [[14.42, 50.07], [14.44, 50.07], [14.44, 50.08], [14.42, 50.07]]
    albert = {"name": "Albert", "shop": "supermarket", "addr:street": "Vinohradská"}
    lines = [
        osmium_line("n7", {"type": "Point", "coordinates": [16.608, 49.195]}, name="Žabka", shop="convenience"),
        osmium_line("w42", {"type": "LineString", "coordinates": ring}, **albert),
        osmium_line("a84", {"type": "MultiPolygon", "coordinates": [[ring]]}, **albert),
        osmium_line("n9", {"type": "Point", "coordinates": [17.0, 49.0]}, amenity="fuel"),  # neither shop nor brand
        osmium_line("n10", {"type": "Point", "coordinates": [17.1, 49.1]}, amenity="fuel", brand="MOL"),
        osmium_line("n12", {"type": "Point", "coordinates": [17.1, 49.1]}, amenity="fuel", shop="no"),
        osmium_line("n11", {"type": "Point", "coordinates": [17.2, 49.2]}, shop="kiosk"),
        osmium_line("a7", {"type": "MultiPolygon", "coordinates": [[ring]]}, name="Billa", shop="supermarket"),
    ]
    rows = pbf_rows(lines, stores, columns)
    assert [row["osm_id"] for row in rows] == ["node/7", "node/10", "relation/3", "way/42"]
    assert (rows[3]["lat"], rows[3]["lon"], rows[3]["street"]) == ("50.0750000", "14.4300000", "Vinohradská")
    assert list(rows[0]) == columns


def test_the_pbf_fallback_gives_the_same_boundaries() -> None:
    """osmium writes a boundary relation as an area with an odd id (relation 2 → `a5`); only the leaves' levels."""
    praha = {"type": "MultiPolygon", "coordinates": [[square(14.4, 50.08, 0.02)]]}
    admin = {"boundary": "administrative"}
    lines = [
        osmium_line("a5", praha, **admin, admin_level="6", name="SO Praha 1"),
        osmium_line("a7", praha, **admin, admin_level="6", name="SO ORP Kladno"),
        osmium_line("a9", praha, **admin, admin_level="6", ref="CZ020", name="Středočeský kraj"),
        osmium_line("a10", praha, **admin, admin_level="6", name="SO Praha 2"),  # way 5's area
        osmium_line("w5", {"type": "LineString", "coordinates": square(14.4, 50.08, 0.02)}, **admin, name="SO Praha 2"),
    ]
    collection = pbf_boundaries(lines)
    [feature] = collection["features"]
    assert feature["properties"] == {"code": "CZ0100-01", "name": "SO Praha 1", "osm_id": "relation/2"}


def test_the_pbf_run_records_the_pbf_in_the_meta(snapshot: Path, ref: Path) -> None:
    pbf = snapshot / "pbf"
    pbf.mkdir()
    point = {"type": "Point", "coordinates": [14.42, 50.07]}
    for extract in EXTRACTS[:3]:
        (pbf / f"{Path(extract.query).stem}.geojsonseq").write_text("", encoding="utf-8")
    (pbf / "stores.geojsonseq").write_text(osmium_line("n7", point, shop="convenience") + "\n", encoding="utf-8")
    leaves = leaf_relations(okres_leaves(ref))["elements"]
    (pbf / "okres-boundaries.geojsonseq").write_text(
        "".join(
            osmium_line(
                f"a{2 * leaf['id'] + 1}",
                {
                    "type": "MultiPolygon",
                    "coordinates": [[[[p["lon"], p["lat"]] for p in leaf["members"][0]["geometry"]]]],
                },
                boundary="administrative",
                **leaf["tags"],
            )
            + "\n"
            for leaf in leaves
        ),
        encoding="utf-8",
    )
    counts = run(snapshot, pbf_dir=pbf, pbf_source="cz.osm.pbf (osmium)", pbf_osm_base="2026-10-30T20:21:00Z")
    assert counts["osm-stores.csv"] == 1 and counts["osm/okres-boundaries.geojson"] == 98
    meta = read_meta(snapshot / META)
    assert {(p.source, p.osm_base) for p in meta.values()} == {("cz.osm.pbf (osmium)", "2026-10-30T20:21:00Z")}
    with pytest.raises(ValueError, match="--pbf-source"):
        run(snapshot, pbf_dir=pbf)
