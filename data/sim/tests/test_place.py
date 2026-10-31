"""G0 S0.3 T4 — street type, store class, size class and the weekly hours; goldens on a 500-row sample."""

from __future__ import annotations

import csv
import io
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from hartland_sim.io.reference import read
from hartland_sim.setup.classifier import chains, place
from hartland_sim.setup.classifier.hours import Week, parse
from hartland_sim.setup.classifier.okres import Points

GOLDEN = Path(__file__).parent / "golden" / "classifier-sample.csv"
SAMPLE_EVERY = 24
SAMPLE_SIZE = 500


@pytest.mark.parametrize(
    ("value", "week"),
    [
        ("Mo-Fr 07:00-18:00; Sa 07:00-12:00", (60.0, False)),
        ("Mo-Su 06:00-22:00", (112.0, True)),
        ("24/7", (168.0, True)),
        ("Mo-Su 00:00-24:00", (168.0, True)),
        ("Mo-Fr 07:00-12:00,13:00-17:00", (45.0, False)),
        ("Mo,We,Fr 08:00-12:00", (12.0, False)),
        ("Mo-Sa 07:00-20:00; Su 08:00-18:00", (88.0, True)),
        ("Mo-Fr 06:30-18:00; Sa,Su 07:00-12:00", (67.5, True)),
        ("Mo-Fr 07:00-17:00; Sa 07:00-11:00; Su off", (54.0, False)),
        ("Mo-Fr 07:00-18:00; Sa 07:00-12:00; Su closed", (60.0, False)),
        ("Mo-Su 07:00-22:00; PH off", (105.0, True)),
        ("Mo-Su,PH 07:00-22:00", (105.0, True)),
        ("Mo-Fr 07:00-18:00, Sa 08:00-12:00; Su,PH off", (59.0, False)),
        ("Mo-Fr 07:00-19:00; Sa, Su 08:00-18:00", (80.0, True)),
        ("Fr-Mo 10:00-14:00", (16.0, True)),
        ("Mo-Su 22:00-06:00", (56.0, True)),
        ("Mo-Su 07:00-22:00; Su 08:00-12:00", (94.0, True)),
        ("08:00-20:00", (84.0, True)),
        ("PH,Mo-Su 08:30-20:30", (84.0, True)),
        ("PH off; Mo-Fr 07:00-15:00", (40.0, False)),
    ],
)
def test_opening_hours(value: str, week: tuple[float, bool]) -> None:
    assert parse(value) == Week(*week)


@pytest.mark.parametrize(
    "value",
    ["", "Mar-Oct 06:00-21:00", "sunrise-sunset", 'Mo-Fr 07:00-18:00 "po telefonu"', "Mo", "Mo-Fr 25:00-26:00"]
    + ["off", "closed", "Mo off", "Mo-Su,PH off"],
)
def test_unreadable_hours_are_unknown(value: str) -> None:
    assert parse(value) is None


def pts(*xy: tuple[float, float]) -> Points:
    return np.array(xy, dtype=float).reshape(-1, 2)


def context() -> place.Context:
    """A city centre at (14.0, 50.0) with a quarter, a mall and a station; a town; a village 20 km out."""
    return place.Context(
        malls=pts((14.05, 50.0)),
        hubs=pts((14.0, 50.03), (14.05, 50.0)),
        settlements=pts((14.0, 50.0), (14.3, 50.0), (14.6, 50.0)),
        settlement_kind=np.array(["city", "village", "town"]),
        settlement_names=["Hartov", "Lhota", "Městečko"],
        towns=pts((14.0, 50.0), (14.6, 50.0), (14.0, 49.95)),
        quarters=pts((14.0, 49.95)),
    )


M = 1 / 111_320  # a metre of latitude, in degrees


@pytest.mark.parametrize(
    ("lon", "lat", "station", "street_type"),
    [
        (14.0, 50.0 + 500 * M, False, "high_street"),  # 500 m from a city centre
        (14.0, 50.0 + 1500 * M, False, "residential"),
        (14.0, 49.95 + 200 * M, False, "high_street"),  # 200 m from a city quarter
        (14.05, 50.0 + 100 * M, False, "shopping_centre"),  # a mall and a station: the mall
        (14.0, 50.03 + 150 * M, False, "transport_hub"),
        (14.0, 50.03 + 150 * M, True, "petrol"),
        (14.3, 50.0 + 300 * M, False, "village"),
        (14.6, 50.0 + 3000 * M, False, "residential"),  # a town's outskirts
    ],
)
def test_street_type(lon: float, lat: float, station: bool, street_type: str) -> None:
    assert place.street_types(np.array([(lon, lat)]), np.array([station]), context()).types == [street_type]


def test_a_quarter_centre_is_urban_whatever_settlement_is_nearest() -> None:
    """Inside a city, the nearest settlement point is often an old village OSM keeps as `place=village`."""
    old_village = place.Context(
        malls=pts(),
        hubs=pts(),
        settlements=pts((14.0, 50.0), (14.3, 49.95 + 150 * M)),
        settlement_kind=np.array(["city", "village"]),
        settlement_names=["Hartov", "Lhota"],
        towns=pts((14.0, 50.0), (14.3, 49.95)),
        quarters=pts((14.3, 49.95)),
    )
    streets = place.street_types(pts((14.3, 49.95 + 100 * M)), np.array([False]), old_village)
    assert streets.types == ["high_street"]
    assert old_village.settlement_names[streets.settlement[0]] == "Lhota"


@pytest.mark.parametrize(
    ("shop", "amenity", "name", "brand", "street_type", "store_class"),
    [
        ("supermarket", "", "Kaufland", "Kaufland", "residential", "hyper"),
        ("supermarket", "", "Albert Hypermarket", "Albert", "residential", "hyper"),
        ("wholesale", "", "Makro", "Makro", "residential", "hyper"),
        ("supermarket", "", "Lidl", "Lidl", "village", "discount"),
        ("supermarket", "", "Billa", "Billa", "high_street", "super"),
        ("supermarket", "", "Potraviny U Lípy", "", "village", "super"),
        ("convenience", "", "COOP Jednota", "", "village", "village"),
        ("convenience", "", "Žabka", "Žabka", "transport_hub", "convenience"),
        ("greengrocer", "", "Ovoce a zelenina", "", "residential", "minimarket"),
        ("", "fuel", "Benzina", "Orlen", "petrol", "petrol"),
        ("convenience", "cafe", "Stop Cafe", "Stop Cafe", "petrol", "petrol"),
    ],
)
def test_store_class(shop: str, amenity: str, name: str, brand: str, street_type: str, store_class: str) -> None:
    chain = chains.match(brand=brand, name=name, fuel=amenity == "fuel")
    assert place.store_class(shop=shop, amenity=amenity, name=name, chain=chain, street_type=street_type) == store_class


def test_a_destination_store_in_a_village_is_out_of_town() -> None:
    assert place.settle("village", "hyper") == place.settle("village", "discount") == "residential"
    assert place.settle("village", "super") == "village"


def classify(ref: Path) -> list[dict[str, str]]:
    """The classifier on the whole extract (the context needs every neighbour), one row per object."""
    objects = read("osm-stores", ref)
    found = [
        chains.match(brand=o["brand"], name=o["name"], operator=o["operator"], fuel=o["amenity"] == "fuel")
        for o in objects
    ]
    points = np.array([(o["lon"], o["lat"]) for o in objects])
    station = np.array(
        [o["amenity"] == "fuel" or (c is not None and c.is_station) for o, c in zip(objects, found, strict=True)]
    )
    streets = place.street_types(points, station, place.Context.load(ref)).types
    rows = []
    for o, c, street in zip(objects, found, streets, strict=True):
        store_class = place.store_class(
            shop=o["shop"], amenity=o["amenity"], name=o["name"], chain=c, street_type=street
        )
        week = parse(o["opening_hours"])
        rows.append(
            {
                "osm_id": o["osm_id"],
                "chain": c.code if c else "",
                "street_type": place.settle(street, store_class),
                "store_class": store_class,
                "size_class": place.SIZE_OF[store_class],
                "weekly_hours": "" if week is None else f"{week.weekly_hours:g}",
                "sunday_open": "" if week is None else str(week.sunday_open).lower(),
            }
        )
    return rows


@pytest.fixture(scope="module")
def classified(ref: Path) -> list[dict[str, str]]:
    return classify(ref)


def test_the_500_row_sample_matches_its_golden(classified: list[dict[str, str]]) -> None:
    sample = classified[::SAMPLE_EVERY][:SAMPLE_SIZE]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, list(sample[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(sample)
    if os.environ.get("HARTLAND_REGENERATE_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(buffer.getvalue(), encoding="utf-8")
    regenerate = f"HARTLAND_REGENERATE_GOLDEN=1 uv run pytest {Path(__file__).name} -k golden, then read the diff"
    assert buffer.getvalue() == GOLDEN.read_text(encoding="utf-8"), regenerate


def test_every_store_gets_a_value_and_every_class_is_present(classified: list[dict[str, str]]) -> None:
    """`01` §7 names the classes, not their shares: each of the six street types and seven store classes holds
    between 1 % (the hypermarkets are 1.5 %) and 40 % of the stores."""
    for column, values in (("street_type", place.STREET_TYPES), ("store_class", place.STORE_CLASSES)):
        shares = Counter(row[column] for row in classified)
        assert set(shares) == set(values), column
        assert all(0.01 <= n / len(classified) <= 0.40 for n in shares.values()), (column, shares)
    assert Counter(row["size_class"] for row in classified).keys() == {"S", "M", "L"}
