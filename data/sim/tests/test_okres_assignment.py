"""G0 S0.3 T2 — the okres of a point: point-in-polygon against the committed boundaries."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from hartland_sim.setup.classifier.okres import Leaf, assign, load

# Town centres (lat, lon), checked by hand against the map, → their okres.csv leaf.
TOWNS = {
    "Benešov": (49.7817, 14.6868, "CZ0201"),
    "Kladno": (50.1475, 14.1030, "CZ0203"),
    "Kolín": (50.0280, 15.2005, "CZ0204"),
    "Jirny": (50.1155, 14.6995, "CZ0209"),
    "České Budějovice": (48.9745, 14.4743, "CZ0311"),
    "Plzeň": (49.7475, 13.3776, "CZ0323"),
    "Cheb": (50.0796, 12.3739, "CZ0411"),
    "Karlovy Vary": (50.2318, 12.8720, "CZ0412"),
    "Ústí nad Labem": (50.6611, 14.0320, "CZ0427"),
    "Liberec": (50.7697, 15.0585, "CZ0513"),
    "Hradec Králové": (50.2092, 15.8328, "CZ0521"),
    "Chrudim": (49.9511, 15.7953, "CZ0531"),
    "Pardubice": (50.0382, 15.7790, "CZ0532"),
    "Jihlava": (49.3961, 15.5912, "CZ0632"),
    "Brno": (49.1951, 16.6081, "CZ0642"),
    "Modřice": (49.1280, 16.6140, "CZ0643"),
    "Hodonín": (48.8489, 17.1324, "CZ0645"),
    "Jeseník": (50.2294, 17.2047, "CZ0711"),
    "Olomouc": (49.5938, 17.2509, "CZ0712"),
    "Zlín": (49.2265, 17.6707, "CZ0724"),
    "Bruntál": (49.9884, 17.4647, "CZ0801"),
    "Ostrava": (49.8354, 18.2925, "CZ0806"),
    "Staroměstské náměstí": (50.0875, 14.4213, "CZ0100-01"),
    "Pankrác": (50.0510, 14.4392, "CZ0100-04"),
    "Vítězné náměstí": (50.1007, 14.3954, "CZ0100-06"),
    "Vršovice": (50.0687, 14.4530, "CZ0100-10"),
    "Stodůlky": (50.0465, 14.3070, "CZ0100-13"),
    "Uhříněves": (50.0310, 14.6050, "CZ0100-22"),
}
ABROAD = {"Wien": (48.2082, 16.3738), "Dresden": (51.0504, 13.7373)}


@pytest.fixture(scope="module")
def boundaries(ref: Path) -> list[Leaf]:
    return load(ref / "osm" / "okres-boundaries.geojson")


def test_the_boundaries_are_the_leaves_of_okres_csv(boundaries: list[Leaf], leaves: list[str]) -> None:
    assert sorted(leaf.code for leaf in boundaries) == sorted(leaves)


def test_hand_checked_towns_land_in_their_okres(boundaries: list[Leaf]) -> None:
    points = np.array([(lon, lat) for lat, lon, _ in TOWNS.values()])
    result = assign(boundaries, points)
    assert dict(zip(TOWNS, result.codes, strict=True)) == {town: code for town, (_, _, code) in TOWNS.items()}


def test_a_point_abroad_is_dropped_and_counted(boundaries: list[Leaf]) -> None:
    points = np.array([(lon, lat) for lat, lon in ABROAD.values()] + [(14.4213, 50.0875)])
    result = assign(boundaries, points)
    assert result.codes == [None, None, "CZ0100-01"]
    assert (result.outside, result.snapped) == (2, 0)


def square(x: float, y: float, size: float) -> np.ndarray:
    return np.array([(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)], dtype=float)


def test_a_point_in_a_sliver_takes_the_nearest_leaf() -> None:
    """Two neighbours simplified apart by ≈ 70 m: a point in the gap is snapped, one 2 km out is not."""
    west, east = Leaf("W", [square(14.0, 50.0, 0.1)]), Leaf("E", [square(14.101, 50.0, 0.1)])
    points = np.array([(14.1004, 50.05), (14.1007, 50.05), (14.25, 50.05)])
    result = assign([west, east], points)
    assert result.codes == ["W", "E", None]
    assert (result.snapped, result.outside) == (2, 1)


def test_a_hole_is_outside_its_leaf() -> None:
    ring = Leaf("R", [square(14.0, 50.0, 0.4), square(14.1, 50.1, 0.2)])
    enclave = Leaf("H", [square(14.1, 50.1, 0.2)])
    result = assign([ring, enclave], np.array([(14.05, 50.05), (14.2, 50.2)]))
    assert result.codes == ["R", "H"]


def test_every_committed_store_has_an_okres_and_every_leaf_stores(ref: Path, boundaries: list[Leaf]) -> None:
    with (ref / "osm-stores.csv").open(encoding="utf-8", newline="") as stores:
        points = np.array([(float(row["lon"]), float(row["lat"])) for row in csv.DictReader(stores)])
    result = assign(boundaries, points)
    assert result.outside == 0
    assert result.snapped <= len(points) // 1000
    assert len(Counter(result.codes)) == len(boundaries)
