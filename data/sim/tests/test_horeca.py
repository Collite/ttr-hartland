"""G0 S0.3 T5 — the HoReCa sample: 600 customers + 2 000 potential, stratified by okres population × affluence."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hartland_sim.io.reference import read
from hartland_sim.setup.affluence import load as affluence
from hartland_sim.setup.classifier.chains import fold
from hartland_sim.setup.classifier.okres import assign, load
from hartland_sim.setup.classifier.place import SAME_NAME_M, mapped_twice
from hartland_sim.setup.horeca import Pick, eligible, quotas, sample


def test_quotas_follow_the_weights() -> None:
    assert quotas({"a": 1, "b": 1, "c": 2}, 8, {"a": 9, "b": 9, "c": 9}) == {"a": 2, "b": 2, "c": 4}


def test_a_capped_quota_goes_to_the_others() -> None:
    assert quotas({"a": 1, "b": 1, "c": 2}, 8, {"a": 1, "b": 9, "c": 9}) == {"a": 1, "b": 2, "c": 5}
    assert sum(quotas({"a": 3, "b": 1}, 10, {"a": 2, "b": 3}).values()) == 5  # no more than there is


def test_a_key_without_weight_takes_nothing() -> None:
    assert quotas({"a": 1.0, "b": 0.0}, 5, {"a": 2, "b": 3}) == {"a": 2, "b": 0}
    assert quotas({"a": 0.0}, 5, {"a": 3}) == {"a": 0}


def test_the_largest_remainder_rounds() -> None:
    assert quotas({"a": 1, "b": 1, "c": 1}, 10, {"a": 9, "b": 9, "c": 9}) == {"a": 4, "b": 3, "c": 3}


@pytest.mark.parametrize(
    ("place", "ok"),
    [
        ({"amenity": "restaurant", "name": "Restaurace U Lípy", "brand": ""}, True),
        ({"amenity": "pub", "name": "Hospoda Na Růžku", "brand": ""}, True),
        ({"amenity": "fast_food", "name": "Bistro", "brand": ""}, False),
        ({"amenity": "cafe", "name": "Starbucks", "brand": "Starbucks"}, False),
        ({"amenity": "cafe", "name": "", "brand": ""}, False),
        ({"amenity": "bar", "name": "The Irish Bar", "brand": ""}, False),
    ],
)
def test_who_is_eligible(place: dict[str, str], ok: bool) -> None:
    assert eligible(place | {"osm_id": "node/1"}) is ok


def test_a_place_mapped_twice_is_one_place() -> None:
    """The store set's rule: the same name within 30 m is one place, the first in OSM order."""
    pub = {"amenity": "pub", "name": "Hospoda", "brand": "", "lon": 14.0}
    places = [
        pub | {"osm_id": "node/1", "lat": 50.0},
        pub | {"osm_id": "node/2", "lat": 50.00005},  # 5.6 m away: the same pub
        pub | {"osm_id": "node/3", "lat": 50.01},  # 1.1 km away: another
    ]
    picks = sample(places, ["CZ0201"] * 3, {"CZ0201": 1.0}, customers=1, potential=5)
    assert sorted(pick.osm_id for pick in picks) == ["node/1", "node/3"]


@pytest.fixture(scope="module")
def picks(ref: Path, leaves: list[str], okres: dict[str, dict[str, Any]]) -> list[Pick]:
    places = read("osm-horeca", ref)
    points = np.array([(place["lon"], place["lat"]) for place in places])
    codes = assign(load(ref / "osm" / "okres-boundaries.geojson"), points).codes
    index = affluence(ref)
    return sample(places, codes, {code: okres[code]["population"] * index[code] for code in leaves})


def test_six_hundred_customers_and_two_thousand_potential(picks: list[Pick]) -> None:
    assert Counter(pick.role for pick in picks) == {"customer": 600, "potential": 2000}
    assert len({pick.osm_id for pick in picks}) == 2600
    assert {pick.kind for pick in picks} == {"restaurant", "cafe", "bar", "canteen"}


def test_no_two_picks_are_one_place(picks: list[Pick], ref: Path) -> None:
    """The three places OSM maps twice (`Kavárna Zelenohorská pošta` 0 m, `Hospůdka na hřišti` 2.7 m, `Hospoda` 7.3 m
    apart) are picked once each at most; no two picks share a folded name within 30 m."""
    chosen = {pick.osm_id for pick in picks}
    for pair in [
        ("node/13981534504", "node/13981543901"),
        ("node/2763021045", "node/13970694535"),
        ("node/1447481429", "node/2314812994"),
    ]:
        assert not set(pair) <= chosen, pair
    rows = [row for row in read("osm-horeca", ref) if row["osm_id"] in chosen]
    points = np.array([(row["lon"], row["lat"]) for row in rows])
    assert not mapped_twice([fold(row["name"]) for row in rows], points, SAME_NAME_M)


def test_every_kraj_is_present_and_prague_leads(picks: list[Pick], okres: dict[str, dict[str, Any]]) -> None:
    by_kraj = Counter(okres[pick.okres]["kraj"] for pick in picks)
    assert len(by_kraj) == 14
    assert by_kraj.most_common(1)[0][0] == okres["CZ0100"]["kraj"]  # the most people, the richest


def test_the_sample_is_the_extracts_not_the_seeds(picks: list[Pick], ref: Path, leaves: list[str]) -> None:
    """Fixed by the OSM ids: a second draw gives the same sample."""
    places = read("osm-horeca", ref)
    points = np.array([(place["lon"], place["lat"]) for place in places])
    codes = assign(load(ref / "osm" / "okres-boundaries.geojson"), points).codes
    again = sample(places, codes, dict.fromkeys(leaves, 1.0))
    assert len(again) == 2600 and again != picks  # other weights, another sample …
    assert sample(places, codes, dict.fromkeys(leaves, 1.0)) == again  # … but the same one each time
