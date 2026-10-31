"""G0 S0.2 T3 — kraj · okres · the wage tables → the affluence index (⚑NH-17)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.io.reference import read
from hartland_sim.setup.affluence import affluence_index, load


def test_14_kraje_77_okresy_and_22_prague_districts(
    ref: Path, okres: dict[str, dict[str, Any]], prague_districts: list[str]
) -> None:
    assert len(read("kraj", ref)) == 14
    assert len([row for row in okres.values() if row["parent_okres"] is None]) == 77
    assert [okres[code]["name"] for code in prague_districts] == [f"Praha {n}" for n in range(1, 23)]
    assert sum(okres[code]["population"] for code in prague_districts) == okres["CZ0100"]["population"]


def test_the_population_is_the_countrys(okres: dict[str, dict[str, Any]]) -> None:
    total = sum(row["population"] for row in okres.values() if row["parent_okres"] is None)
    assert 10_500_000 < total < 11_200_000


def test_the_index_weighted_by_population_is_about_one(
    ref: Path, okres: dict[str, dict[str, Any]], leaves: list[str]
) -> None:
    index = load(ref)
    people = sum(okres[code]["population"] for code in leaves)
    mean = sum(index[code] * okres[code]["population"] for code in leaves) / people
    assert mean == pytest.approx(1.0, abs=0.05)


def test_the_index_ends(ref: Path) -> None:
    index = load(ref)
    assert all(0.8 <= value <= 1.3 for value in index.values())
    assert index["CZ0100-01"] == 1.3  # Praha 1
    assert index["CZ0801"] <= 0.82  # Bruntál
    assert index["CZ0711"] <= 0.82  # Jeseník


def test_the_index_clamps() -> None:
    kraj_of = {"rich": "K", "poor": "K", "mid": "K"}
    index = affluence_index(kraj_of, {"CZ": 40_000, "K": 40_000}, {"rich": 1.5, "poor": 0.5, "mid": 1.1})
    assert index == {"rich": 1.3, "poor": 0.8, "mid": 1.1}


def test_the_spread_keeps_each_krajs_figure(ref: Path, okres: dict[str, dict[str, Any]], leaves: list[str]) -> None:
    """ČSÚ's kraj wage is the population-weighted mean of its okresy: the spread moves okresy, not kraje."""
    spread = {row["okres"]: row["wage_vs_kraj"] for row in read("wage_by_okres", ref)}
    weighted: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for code in leaves:
        weighted[okres[code]["kraj"]][0] += spread[code] * okres[code]["population"]
        weighted[okres[code]["kraj"]][1] += okres[code]["population"]
    for kraj, (sum_, people) in weighted.items():
        assert sum_ / people == pytest.approx(1.0, abs=0.01), kraj


def test_the_wage_table_covers_every_kraj_and_the_country(ref: Path) -> None:
    kraje = {row["code"] for row in read("kraj", ref)}
    assert {row["code"] for row in read("wage_by_kraj", ref)} == kraje | {"CZ"}
