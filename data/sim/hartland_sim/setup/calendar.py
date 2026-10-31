"""The calendar (⚑NH-12): one row per day with the Czech days off, the ISO week and the season.

`data/reference/calendar.csv` (2023-01-01 → 2027-12-31: the warm-up year 2023, the history from 2024-01-01, room to
prolong) is these rules' output, committed and drift-tested; `prolong` past 2027 extends it with the same rules.
Regenerate: `uv run python -m hartland_sim.setup.calendar > data/reference/calendar.csv`.

The days off are the thirteen of zákon č. 245/2000 Sb. — eleven fixed, Good Friday and Easter Monday from the
Gregorian Easter (the anonymous algorithm, Meeus/Jones/Butcher). No substitute day when one falls on a weekend.
"""

from __future__ import annotations

import csv
import io
import sys
from collections.abc import Iterator
from datetime import date, timedelta

FIRST_DAY = date(2023, 1, 1)
LAST_DAY = date(2027, 12, 31)

COLUMNS = (
    "day",
    "year",
    "quarter",
    "month",
    "week",
    "iso_year",
    "weekday",
    "weekday_cs",
    "is_working_day",
    "holiday_cs",
    "season",
)

FIXED_HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 1): "Den obnovy samostatného českého státu, Nový rok",
    (5, 1): "Svátek práce",
    (5, 8): "Den vítězství",
    (7, 5): "Den slovanských věrozvěstů Cyrila a Metoděje",
    (7, 6): "Den upálení mistra Jana Husa",
    (9, 28): "Den české státnosti",
    (10, 28): "Den vzniku samostatného československého státu",
    (11, 17): "Den boje za svobodu a demokracii a Mezinárodní den studentstva",
    (12, 24): "Štědrý den",
    (12, 25): "1. svátek vánoční",
    (12, 26): "2. svátek vánoční",
}

WEEKDAYS_CS = ("pondělí", "úterý", "středa", "čtvrtek", "pátek", "sobota", "neděle")

# Meteorological seasons, by month.
SEASONS_CS = {
    month: season
    for season, months in (("zima", (12, 1, 2)), ("jaro", (3, 4, 5)), ("léto", (6, 7, 8)), ("podzim", (9, 10, 11)))
    for month in months
}


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (the anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)


def holidays(year: int) -> dict[date, str]:
    """The thirteen Czech days off of `year`, by day."""
    easter = easter_sunday(year)
    days = {date(year, month, day): name for (month, day), name in FIXED_HOLIDAYS.items()}
    days[easter - timedelta(days=2)] = "Velký pátek"
    days[easter + timedelta(days=1)] = "Velikonoční pondělí"
    return dict(sorted(days.items()))


def days(first: date = FIRST_DAY, last: date = LAST_DAY) -> Iterator[dict[str, object]]:
    """The calendar rows from `first` to `last`, both included."""
    off: dict[date, str] = {}
    for year in range(first.year, last.year + 1):
        off |= holidays(year)
    day = first
    while day <= last:
        iso = day.isocalendar()
        holiday = off.get(day)
        yield {
            "day": day,
            "year": day.year,
            "quarter": (day.month - 1) // 3 + 1,
            "month": day.month,
            "week": iso.week,
            "iso_year": iso.year,
            "weekday": iso.weekday,
            "weekday_cs": WEEKDAYS_CS[iso.weekday - 1],
            "is_working_day": iso.weekday <= 5 and holiday is None,
            "holiday_cs": holiday,
            "season": SEASONS_CS[day.month],
        }
        day += timedelta(days=1)


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def render(first: date = FIRST_DAY, last: date = LAST_DAY) -> str:
    """`calendar.csv` as committed: the reference convention (`io.reference`)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in days(first, last):
        writer.writerow(_cell(row[column]) for column in COLUMNS)
    return out.getvalue()


if __name__ == "__main__":
    sys.stdout.write(render())
