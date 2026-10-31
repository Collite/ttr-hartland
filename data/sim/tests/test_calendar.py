"""G0 S0.2 T2 — `calendar.csv`: the Czech days off, the ISO week, the season; the file is the rules' output."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from hartland_sim.io.reference import read
from hartland_sim.setup.calendar import easter_sunday, holidays, render


@pytest.fixture(scope="module")
def calendar(repo_root: Path) -> list[dict[str, object]]:
    return read("calendar", repo_root / "data" / "reference")


def test_the_committed_file_is_the_rules_output(repo_root: Path) -> None:
    committed = (repo_root / "data" / "reference" / "calendar.csv").read_text(encoding="utf-8")
    regenerate = "uv run python -m hartland_sim.setup.calendar > data/reference/calendar.csv"
    assert committed == render(), f"the rules changed: {regenerate}"


def test_it_spans_the_warm_up_year_to_2027(calendar: list[dict[str, object]]) -> None:
    assert calendar[0]["day"] == date(2023, 1, 1)
    assert calendar[-1]["day"] == date(2027, 12, 31)
    assert len(calendar) == 1826


@pytest.mark.parametrize(
    ("year", "sunday"),
    [(2023, date(2023, 4, 9)), (2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5))]
    + [(2027, date(2027, 3, 28)), (2038, date(2038, 4, 25)), (1961, date(1961, 4, 2))],
)
def test_easter(year: int, sunday: date) -> None:
    assert easter_sunday(year) == sunday


def test_easter_monday_2026(calendar: list[dict[str, object]]) -> None:
    by_day = {row["day"]: row for row in calendar}
    monday = by_day[date(2026, 4, 6)]
    assert (monday["holiday_cs"], monday["weekday_cs"], monday["is_working_day"]) == (
        "Velikonoční pondělí",
        "pondělí",
        False,
    )
    assert by_day[date(2026, 4, 3)]["holiday_cs"] == "Velký pátek"


def test_thirteen_days_off_every_year() -> None:
    for year in range(2023, 2028):
        assert len(holidays(year)) == 13


def test_2026_has_251_working_days_give_or_take_one(calendar: list[dict[str, object]]) -> None:
    working = sum(1 for row in calendar if row["year"] == 2026 and row["is_working_day"])
    assert abs(working - 251) <= 1, working


def test_a_holiday_on_a_weekend_is_not_moved(calendar: list[dict[str, object]]) -> None:
    by_day = {row["day"]: row for row in calendar}
    assert by_day[date(2026, 12, 26)]["holiday_cs"] == "2. svátek vánoční"  # a Saturday
    assert by_day[date(2026, 12, 28)]["is_working_day"] is True


def test_iso_week_and_its_year(calendar: list[dict[str, object]]) -> None:
    by_day = {row["day"]: row for row in calendar}
    assert (by_day[date(2027, 1, 1)]["week"], by_day[date(2027, 1, 1)]["iso_year"]) == (53, 2026)
    assert (by_day[date(2024, 12, 30)]["week"], by_day[date(2024, 12, 30)]["iso_year"]) == (1, 2025)


def test_the_seasons(calendar: list[dict[str, object]]) -> None:
    season = {row["month"]: row["season"] for row in calendar}
    assert [season[m] for m in (12, 1, 2, 3, 6, 9)] == ["zima", "zima", "zima", "jaro", "léto", "podzim"]
