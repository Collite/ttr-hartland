"""G0 S0.3 T6 — the market: the OSM extract → `store` + `chain` (`03` §3), and the store set report."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from hartland_sim.checks.czech import assert_czech
from hartland_sim.io.reference import read
from hartland_sim.setup.classifier import place
from hartland_sim.setup.market import STORE_COLUMNS, Market, build


@pytest.fixture(scope="module")
def market(ref: Path) -> Market:
    return build(ref)


def test_the_store_set(market: Market) -> None:
    """The certification line of the full extract (STATUS): ~10 k grocery stores, ~1.9 k stations, 2 600 HoReCa."""
    kinds = Counter(
        "horeca" if s["store_class"] == "horeca" else "station" if s["store_class"] == "petrol" else "grocery"
        for s in market.stores
    )
    assert 9_000 <= kinds["grocery"] <= 14_000
    assert 1_500 <= kinds["station"] <= 2_800
    assert kinds["horeca"] == 2_600
    assert len({s["osm_id"] for s in market.stores}) == len(market.stores)


def test_every_store_has_an_okres_an_area_and_a_region(market: Market, ref: Path, leaves: list[str]) -> None:
    areas = {row["code"]: row["region"] for row in read("area", ref)}
    for store in market.stores:
        assert store["okres"] in leaves and areas[store["area"]] == store["region"], store
        assert list(store) == list(STORE_COLUMNS)


def test_the_codes_are_the_tables(market: Market) -> None:
    assert {s["store_class"] for s in market.stores} == {*place.STORE_CLASSES, "horeca"}
    assert {s["street_type"] for s in market.stores} == set(place.STREET_TYPES)
    assert {s["size_class"] for s in market.stores} == {"S", "M", "L"}
    assert all(0 < s["weekly_hours"] <= 168 for s in market.stores)


def test_every_name_in_the_world_reads_as_czech(market: Market) -> None:
    """NH-D8 where the OSM names enter the world; the generated ones (`Potraviny Lhota`) included."""
    for store in market.stores:
        assert_czech(store["name"], what=store["osm_id"])
    assert market.report.dropped["name not Czech"] > 0


def test_the_station_split(market: Market) -> None:
    """NH-D5: OMV and Shell stations belong to key-account chains; every other station is small retail."""
    key_accounts = {chain["code"] for chain in market.chains if chain["is_key_account"]}
    stations = [s for s in market.stores if s["store_class"] == "petrol"]
    assert {s["chain"] for s in stations if s["chain"] in key_accounts} == {"OMV", "SHELL"}
    assert sum(s["chain"] in ("OMV", "SHELL") for s in stations) > 250


def test_the_chains(market: Market) -> None:
    assert sorted(c["code"] for c in market.chains if c["is_key_account"]) == sorted(
        ["TESCO", "ALBERT", "LIDL", "KAUFLAND", "PENNY", "BILLA", "GLOBUS", "COOP", "MAKRO", "OMV", "SHELL"]
    )
    used = Counter(s["chain"] for s in market.stores if s["chain"])
    assert all(used[c["code"]] > 0 for c in market.chains)
    assert used["MAKRO"] >= 10  # the cash-and-carry stores


def test_a_stations_own_shop_is_not_a_second_store(market: Market) -> None:
    assert market.report.dropped["station's own shop"] > 100
    assert market.report.dropped["mapped twice"] > 0


@pytest.mark.parametrize(
    ("osm_id", "expected"),
    [
        ("node/334399644", ("Albert", "ALBERT", "super", "shopping_centre", "Praha", "CZ0100-11")),  # Westfield Chodov
        ("node/33576718", ("Kaufland", "KAUFLAND", "hyper", "residential", "Mělník", "CZ0206")),
        ("node/976473599", ("Makro", "MAKRO", "hyper", "residential", "Praha", "CZ0100-13")),
        ("node/33381345", ("OMV KV, Pobřežní", "OMV", "petrol", "petrol", "Karlovy Vary", "CZ0412")),
        ("node/302691345", ("COOP", "COOP", "village", "village", "Smilovy Hory", "CZ0317")),
        ("node/251873868", ("Smíšené zboží Libouchec", None, "village", "village", "Libouchec", "CZ0427")),
    ],
)
def test_hand_checked_stores(market: Market, osm_id: str, expected: tuple[object, ...]) -> None:
    [store] = [s for s in market.stores if s["osm_id"] == osm_id]
    columns = ("name", "chain", "store_class", "street_type", "town", "okres")
    assert tuple(store[column] for column in columns) == expected


def test_the_report_counts_class_by_region(market: Market) -> None:
    lines = market.report.lines()
    assert lines[1].split()[:7] == ["class", "JMV", "PHA", "SJC", "SMS", "VCH", "ZSC"]
    assert lines[-1].split()[0] == "horeca" and lines[-1].split()[-1] == "2600"
    assert sum(market.report.by_class_region.values()) == len(market.stores)
