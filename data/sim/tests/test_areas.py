"""G0 S0.2 T4 — regions · areas · area_okres (`01` §4.3: 6 regions, 50 areas, 10 of them in Prague)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.io.reference import read

PRAHA = "CZ0100"


@pytest.fixture(scope="module")
def ref(repo_root: Path) -> Path:
    return repo_root / "data" / "reference"


@pytest.fixture(scope="module")
def okres(ref: Path) -> dict[str, dict[str, Any]]:
    return {row["code"]: row for row in read("okres", ref)}


@pytest.fixture(scope="module")
def leaves(okres: dict[str, dict[str, Any]]) -> list[str]:
    """The okresy a customer can sit in: every okres but Praha, which its 22 districts replace."""
    parents = {row["parent_okres"] for row in okres.values()}
    return [code for code in okres if code not in parents]


def test_six_regions_and_fifty_areas_ten_in_prague(ref: Path) -> None:
    assert [row["code"] for row in read("region", ref)] == ["PHA", "SJC", "ZSC", "VCH", "JMV", "SMS"]
    areas = read("area", ref)
    assert len(areas) == 50
    assert Counter(row["region"] for row in areas)["PHA"] == 10
    assert sorted(row["rep_slot"] for row in areas) == list(range(1, 51))


def test_every_okres_is_in_exactly_one_area(ref: Path, leaves: list[str]) -> None:
    covered = Counter(row["okres"] for row in read("area_okres", ref))
    assert set(covered) == set(leaves)
    assert set(covered.values()) == {1}


def test_prague_areas_cover_the_districts_and_nothing_else(
    ref: Path, okres: dict[str, dict[str, Any]], leaves: list[str]
) -> None:
    region = {row["code"]: row["region"] for row in read("area", ref)}
    in_prague = {row["okres"] for row in read("area_okres", ref) if region[row["area"]] == "PHA"}
    assert in_prague == {code for code in leaves if okres[code]["parent_okres"] == PRAHA}


def test_a_kraj_never_straddles_two_regions(ref: Path, okres: dict[str, dict[str, Any]]) -> None:
    region = {row["code"]: row["region"] for row in read("area", ref)}
    regions_of: dict[str, set[str]] = defaultdict(set)
    for row in read("area_okres", ref):
        regions_of[okres[row["okres"]]["kraj"]].add(region[row["area"]])
    assert {kraj: len(regions) for kraj, regions in regions_of.items()} == dict.fromkeys(regions_of, 1)
