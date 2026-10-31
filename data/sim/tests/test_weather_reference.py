"""G0 S0.2 T5 — the weather normals and the severe-day calendar (A-5)."""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

from hartland_sim.io.reference import read

REGIONS = {"PHA", "SJC", "ZSC", "VCH", "JMV", "SMS"}


# ── T5 ──────────────────────────────────────────────────────────────────────


def test_a_normal_for_every_region_and_month(ref: Path) -> None:
    normals = read("weather_normals", ref)
    assert {(row["region"], row["month"]) for row in normals} == {(r, m) for r in REGIONS for m in range(1, 13)}
    for region in REGIONS:
        temp = {row["month"]: row["temp_mean_c"] for row in normals if row["region"] == region}
        assert temp[1] < 0 < temp[4] < temp[7], region
        assert max(temp, key=lambda month: temp[month]) == 7, region


def test_the_planted_a5_days_are_in_the_history(ref: Path) -> None:
    planted = [row for row in read("severe_days", ref) if row["story"] == "A-5"]
    days = {row["day"] for row in planted}
    assert len(days) >= 10
    assert all(date(2024, 1, 1) <= day <= date(2026, 10, 31) for day in days)
    assert {row["region"] for row in planted} == REGIONS


def test_every_region_has_a_background_every_year(ref: Path) -> None:
    background = Counter((row["region"], row["day"].year) for row in read("severe_days", ref) if not row["story"])
    assert set(background) == {(r, y) for r in REGIONS for y in range(2023, 2028)}
    assert min(background.values()) >= 4
