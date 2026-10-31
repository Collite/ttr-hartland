"""G0 S0.3 T1 — the Overpass queries and `just osm-extract`, against captured responses (no network in tests)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.io.osm_extract import (
    EXTRACTS,
    OverpassError,
    assemble,
    boundaries,
    leaf_code,
    pbf_boundaries,
    pbf_rows,
    query_columns,
    run,
    simplify,
    tsv_rows,
)

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
        ({"name": "okres Benešov", "ref": "CZ0201"}, "CZ0201"),
        ({"name": "okres Praha-západ", "ref": "CZ020A"}, "CZ020A"),
        ({"name": "SO Praha 7"}, "CZ0100-07"),
        ({"name": "SO Praha 22"}, "CZ0100-22"),
        ({"name": "Hlavní město Praha", "ref": "CZ0100"}, None),
        ({"name": "Landkreis Cham", "ref": "DE"}, None),
    ],
)
def test_a_relation_names_its_leaf(tags: dict[str, str], code: str | None) -> None:
    assert leaf_code(tags) == code


def relation(rid: int, tags: dict[str, str], outer: list[tuple[float, float]], inner: list[tuple[float, float]]) -> Any:
    def way(points: list[tuple[float, float]], role: str) -> dict[str, Any]:
        return {"type": "way", "role": role, "geometry": [{"lon": x, "lat": y} for x, y in points]}

    members = [way(outer, "outer")] + ([way(inner, "inner")] if inner else [])
    return {"type": "relation", "id": rid, "tags": tags, "members": members}


def overpass_boundaries() -> dict[str, Any]:
    return {
        "osm3s": {"timestamp_osm_base": "2026-10-31T12:00:00Z"},
        "elements": [
            relation(2, {"name": "SO Praha 1", "admin_level": "6"}, square(14.4, 50.08, 0.02), []),
            relation(1, {"name": "okres Benešov", "ref": "CZ0201"}, square(14.5, 49.6, 0.4), square(14.6, 49.7, 0.1)),
            relation(3, {"name": "Hlavní město Praha", "ref": "CZ0100"}, square(14.2, 49.9, 0.6), []),
        ],
    }


def test_the_boundaries_become_one_multipolygon_per_leaf() -> None:
    collection = boundaries(overpass_boundaries())
    assert [feature["properties"]["code"] for feature in collection["features"]] == ["CZ0100-01", "CZ0201"]
    benesov = collection["features"][1]
    assert benesov["properties"] == {"code": "CZ0201", "name": "okres Benešov", "osm_id": "relation/1"}
    [[outer, hole]] = benesov["geometry"]["coordinates"]
    assert len(outer) == len(hole) == 5


def fake_overpass(queries: list[str]) -> Any:
    """Each CSV query gets four rows under its own header; the boundary query gets `overpass_boundaries()`."""

    def send(query: str) -> bytes:
        queries.append(query)
        if "out geom" in query:
            return json.dumps(overpass_boundaries()).encode()
        fields = query.split("[out:csv(")[1].split(")]")[0].replace("\n", " ").split(",")
        header = [field.strip().strip('"').replace("::", "@") for field in fields]
        rows = [["node", str(n), "50.0", "14.0", "Lhota"] + [""] * (len(header) - 5) for n in range(1, 5)]
        return "".join("\t".join(row) + "\n" for row in [header, *rows]).encode()

    return send


def test_the_verb_writes_every_file_and_the_meta(tmp_path: Path, ref: Path) -> None:
    shutil.copytree(ref / "osm", tmp_path / "osm", ignore=shutil.ignore_patterns("*.geojson", "*.meta"))
    queries: list[str] = []
    counts = run(tmp_path, fetch=fake_overpass(queries))
    assert counts == {
        "osm-stores.csv": 4,
        "osm-horeca.csv": 4,
        "osm-context.csv": 4,
        "osm/okres-boundaries.geojson": 2,
    }
    assert len(queries) == len(EXTRACTS)
    with (tmp_path / "osm-stores.csv").open(encoding="utf-8", newline="") as stores:
        assert next(csv.reader(stores))[:4] == ["osm_id", "lat", "lon", "name"]
    meta = (tmp_path / "osm" / "osm-extract.meta").read_text(encoding="utf-8")
    assert "osm_base: 2026-10-31T12:00:00Z" in meta
    assert "  osm-stores.csv: 4" in meta


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


def test_the_pbf_fallback_gives_the_same_rows(ref: Path) -> None:
    """Overpass's filter (a fuel station counts with a shop or a brand) and its `out center` (the bounding box's)."""
    [stores] = [extract for extract in EXTRACTS if extract.output == "osm-stores.csv"]
    columns = query_columns((ref / "osm" / stores.query).read_text(encoding="utf-8"))
    lines = [
        osmium_line("n7", {"type": "Point", "coordinates": [16.608, 49.195]}, name="Žabka", shop="convenience"),
        osmium_line(
            "w42",
            {"type": "Polygon", "coordinates": [[[14.42, 50.07], [14.44, 50.07], [14.44, 50.08], [14.42, 50.07]]]},
            name="Albert",
            shop="supermarket",
            **{"addr:street": "Vinohradská"},
        ),
        osmium_line("n9", {"type": "Point", "coordinates": [17.0, 49.0]}, amenity="fuel"),  # neither shop nor brand
        osmium_line("n10", {"type": "Point", "coordinates": [17.1, 49.1]}, amenity="fuel", brand="MOL"),
        osmium_line("n11", {"type": "Point", "coordinates": [17.2, 49.2]}, shop="kiosk"),
    ]
    rows = pbf_rows(lines, stores, columns)
    assert [row["osm_id"] for row in rows] == ["node/7", "node/10", "way/42"]
    assert (rows[2]["lat"], rows[2]["lon"], rows[2]["street"]) == ("50.0750000", "14.4300000", "Vinohradská")
    assert list(rows[0]) == columns


def test_the_pbf_fallback_gives_the_same_boundaries() -> None:
    praha = {"type": "Polygon", "coordinates": [square(14.4, 50.08, 0.02)]}
    lines = [
        osmium_line("r2", praha, boundary="administrative", admin_level="6", name="SO Praha 1"),
        osmium_line("r3", praha, boundary="administrative", admin_level="6", name="SO ORP Kladno"),
        osmium_line("w5", praha, boundary="administrative", name="SO Praha 2"),
    ]
    collection = pbf_boundaries(lines)
    assert [feature["properties"]["code"] for feature in collection["features"]] == ["CZ0100-01"]
