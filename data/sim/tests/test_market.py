"""G0 S0.3 T6 — the market: the OSM extract → `store` + `chain` (`03` §3), and the store set report."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from hartland_sim.checks.czech import assert_czech
from hartland_sim.io.reference import read
from hartland_sim.setup.classifier import place
from hartland_sim.setup.market import INDEPENDENT, STORE_COLUMNS, Market, Report, _weeks, build

# The eligible HoReCa places that are grocery stores too (a shop tagged a café or a pub as well).
IN_BOTH_EXTRACTS = (
    "node/820168227",  # Cukrárna u Zdubů
    "node/2385315906",  # Podbukovina
    "node/2503668956",  # Koloniál U Zámku
    "node/3243798662",  # Černomorka
    "node/3243798663",  # La Bottega Bistroteka
    "node/3448220174",  # La Cucina Tusarova
    "node/7504671755",  # U Týnky
    "node/10065096948",  # Dobrotéka
    "way/1245375292",  # U draka
)


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


def test_every_generated_name_reads_as_czech(market: Market, ref: Path) -> None:
    """NH-D8 on the names the market makes up for an unnamed store (`Potraviny Lhota`, a chain's name)."""
    unnamed = {row["osm_id"] for row in read("osm-stores", ref) if not row["name"]}
    generated = [store for store in market.stores if store["osm_id"] in unnamed]
    assert len(generated) > 1000
    for store in generated:
        assert_czech(store["name"], what=store["osm_id"])


def test_an_osm_name_is_a_proper_noun(market: Market) -> None:
    """⚑NH-2: no store is dropped for its name; the report counts the names the Czech guard would flag."""
    names = {store["name"] for store in market.stores}
    assert {"Potraviny An Duc", "MOL, Praha 6 - Evropská (in)", "MOL, Praha 6 - Evropská (out)"} <= names
    assert "name not Czech" not in market.report.dropped
    assert market.report.kept["OSM name kept verbatim (⚑NH-2)"] >= 20


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


def test_a_place_in_both_extracts_is_the_grocery_store(market: Market) -> None:
    grocery = {s["osm_id"] for s in market.stores if s["store_class"] != "horeca"}
    horeca = {s["osm_id"] for s in market.stores if s["store_class"] == "horeca"}
    assert set(IN_BOTH_EXTRACTS) <= grocery
    assert not grocery & horeca


def test_a_station_without_a_shop_is_no_store(market: Market, ref: Path) -> None:
    """`shop=no`: the station says it has no shop. With a brand it stays (the extract's brand rule), counted."""
    no_shop = {r["osm_id"]: r for r in read("osm-stores", ref) if r["amenity"] == "fuel" and r["shop"] == "no"}
    stores = {s["osm_id"] for s in market.stores}
    assert all(row["brand"] for osm_id, row in no_shop.items() if osm_id in stores)
    assert market.report.dropped["no shop"] == sum(not row["brand"] for row in no_shop.values()) == 10
    assert market.report.kept["station shop=no, by its brand"] == len(stores & set(no_shop)) > 0


def test_an_independent_brand_is_not_its_operators_chain(market: Market) -> None:
    """Robin Oil's stations are run by Čepro, whose pattern is EuroOil's; the brand decides. The shops Čepro runs
    beside them (`Senza Area`) are still the stations' own, not a second store."""
    robin_oil = [s for s in market.stores if s["brand"] == "Robin Oil"]
    assert len(robin_oil) >= 25 and all(s["chain"] is None for s in robin_oil)
    assert not [s for s in market.stores if s["name"] == "Senza Area"]


def test_an_independent_networks_station_mapped_twice_is_one_store(market: Market) -> None:
    """The same brand within `SAME_CHAIN_M`, as for a chain: Tesco Plzeň's pumps are two nodes 40 m apart."""
    stores = {s["osm_id"] for s in market.stores}
    assert "node/7529712872" in stores and "node/12222064768" not in stores


def test_a_store_beside_a_quarter_centre_is_on_the_high_street(market: Market, ref: Path) -> None:
    """Unless a station, a mall or a hub claims it first (`place.street_types`' order)."""
    quarters = place.Context.load(ref).quarters
    distance, _ = place.nearest(quarters, np.array([(s["lon"], s["lat"]) for s in market.stores], dtype=float))
    beside = [s for s, d in zip(market.stores, distance, strict=True) if d <= place.QUARTER_M]
    assert len(beside) > 300
    assert {s["street_type"] for s in beside} == {"high_street", "petrol", "shopping_centre", "transport_hub"}


def test_untagged_hours_come_from_the_class_and_chain_where_it_can(market: Market) -> None:
    assert market.report.hours_drawn["class and chain"] > market.report.hours_drawn["class"] > 0


def test_hours_need_a_tagged_store_to_draw_from() -> None:
    with pytest.raises(ValueError, match="no readable opening_hours"):
        _weeks([""], ["node/1"], ["super"], [INDEPENDENT], Report())


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
