"""The daily weather of the six sales regions (`03` §1 `weather_daily`; `04` S1's weather term; A-5's severe days).

**A pure function of (region, day).** Each region's day is drawn under the master seed from stream `weather`, keyed by
the region and the day, so the series extends to any date: a prolong reads the weather a straight build read, and
the demand model can ask for any day without a state.

- **Temperature**: the month's normal (`weather_normals.csv`), drawn as a line between the middles of the months, plus
  the month's `temp_sd_c` × an anomaly that persists: a weighted sum of the last seven days' draws (weights 7…1), part
  national (key `CZ`) and part the region's, scaled to a variance of 1. A cold spell lasts about a week and covers
  the country, as winter weather does.
- **Precipitation**: a wet day with the month's `wet_days ÷ days`; on a wet day a gamma amount whose mean is
  `precip_month_mm ÷ wet_days`. `snow` = a wet day below `SNOW_BELOW_C`.
- **Severe days** (`severe_days.csv`, the planted A-5 days and a background): `severe = true`, and the day agrees with
  its kind — snow is cold and snowy, ice freezes, a storm or a flood rains hard, wind leaves the numbers alone.
"""

from __future__ import annotations

import calendar
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from hartland_sim.ids import row_id
from hartland_sim.io.reference import REFERENCE_DIR, read
from hartland_sim.rng import rng
from hartland_sim.setup.rows import Row, row

NATIONAL = "CZ"
WEIGHTS = (7, 6, 5, 4, 3, 2, 1)  # today's draw first
NATIONAL_SHARE = 0.6  # of the anomaly's variance
SNOW_BELOW_C = 0.5
GAMMA_SHAPE = 0.8
_NORM = math.sqrt(sum(w * w for w in WEIGHTS))


@dataclass(frozen=True)
class Normal:
    temp_mean_c: float
    temp_sd_c: float
    precip_month_mm: float
    wet_days: float


@dataclass(frozen=True)
class Weather:
    region: str
    day: date
    temp_mean_c: float
    precip_mm: float
    snow: bool
    severe: bool


@dataclass
class Climate:
    """The regions' normals and severe days; `day(region, day)` is the weather (module docstring)."""

    normals: Mapping[tuple[str, int], Normal]  # (region, month) → its normal
    severe: Mapping[tuple[str, date], str]  # (region, day) → the severe day's kind
    regions: tuple[str, ...]  # in weather_normals.csv's order
    _draws: dict[tuple[str, date], tuple[float, float, float]] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, root: Path = REFERENCE_DIR) -> Climate:
        normals = {
            (row["region"], int(row["month"])): Normal(
                float(row["temp_mean_c"]),
                float(row["temp_sd_c"]),
                float(row["precip_month_mm"]),
                float(row["wet_days"]),
            )
            for row in read("weather_normals", root)
        }
        severe = {(row["region"], row["day"]): row["kind"] for row in read("severe_days", root)}
        regions = tuple(dict.fromkeys(region for region, _ in normals))
        return cls(normals, severe, regions)

    def normal_temp(self, region: str, day: date) -> float:
        """The temperature normal of `day`: a line between the middles of its month and the nearer neighbour."""
        middle = _middle(day.year, day.month)
        other = _shift(day.year, day.month, 1 if day >= middle else -1)
        other_middle = _middle(*other)
        here = self.normals[region, day.month].temp_mean_c
        there = self.normals[region, other[1]].temp_mean_c
        share = abs((day - middle).days) / abs((other_middle - middle).days)
        return here + (there - here) * share

    def day(self, region: str, day: date) -> Weather:
        normal = self.normals[region, day.month]
        anomaly = 0.0
        for lag, weight in enumerate(WEIGHTS):
            past = day - timedelta(days=lag)
            national = self._draw(NATIONAL, past)[0]
            own = self._draw(region, past)[0]
            anomaly += weight * (math.sqrt(NATIONAL_SHARE) * national + math.sqrt(1 - NATIONAL_SHARE) * own)
        temp = self.normal_temp(region, day) + normal.temp_sd_c * anomaly / _NORM
        _, wet_draw, amount_draw = self._draw(region, day)
        wet = wet_draw < normal.wet_days / calendar.monthrange(day.year, day.month)[1]
        precip = amount_draw * normal.precip_month_mm / normal.wet_days if wet else 0.0
        kind = self.severe.get((region, day))
        if kind == "snow":
            temp, precip = min(temp, -1.0), max(precip, 8.0)
        elif kind == "ice":
            temp, precip = min(max(temp, -3.0), 0.0), max(precip, 3.0)
        elif kind == "storm":
            precip = max(precip, 25.0)
        elif kind == "flood":
            precip = max(precip, 60.0)
        temp, precip = round(temp, 1), round(precip, 1)
        snow = kind == "snow" or (precip > 0 and temp < SNOW_BELOW_C and kind != "ice")
        return Weather(region, day, temp, precip, snow, kind is not None)

    def rows(self, day: date) -> list[Row]:
        """`weather_daily`'s rows of `day`, one per region."""
        rows = []
        for region in self.regions:
            weather = self.day(region, day)
            rows.append(
                row(
                    "weather_daily",
                    region_id=row_id("region", code=region),
                    day=day,
                    temp_mean_c=weather.temp_mean_c,
                    precip_mm=weather.precip_mm,
                    snow=weather.snow,
                    severe=weather.severe,
                )
            )
        return rows

    def _draw(self, key: str, day: date) -> tuple[float, float, float]:
        """(standard normal, uniform, the wet day's gamma amount ÷ its mean) of `key` on `day`."""
        found = self._draws.get((key, day))
        if found is None:
            draw = rng("weather", key, day)
            found = (
                float(draw.standard_normal()),
                float(draw.random()),
                float(draw.gamma(GAMMA_SHAPE, 1 / GAMMA_SHAPE)),
            )
            self._draws[key, day] = found
        return found


def _middle(year: int, month: int) -> date:
    days = calendar.monthrange(year, month)[1]
    return date(year, month, 1) + timedelta(days=(days - 1) // 2)


def _shift(year: int, month: int, by: int) -> tuple[int, int]:
    index = year * 12 + month - 1 + by
    return index // 12, index % 12 + 1
