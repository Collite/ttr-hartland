"""The daily weather (G1 S1.2 T1): per region × day from the normals (stream `weather`), the planted severe days
flagged, any date reachable, the January normal held on average."""

from __future__ import annotations

from datetime import date, timedelta
from statistics import mean

import pytest

from hartland_sim.io.reference import read
from hartland_sim.loop.weather import Climate
from hartland_sim.rng import using_master_seed
from hartland_sim.setup.rows import columns


@pytest.fixture(scope="module")
def climate() -> Climate:
    return Climate.load()


def test_every_region_s_day_is_drawn_the_same_by_a_fresh_climate(climate: Climate) -> None:
    day = date(2024, 2, 14)
    assert [climate.day(region, day) for region in climate.regions] == [
        Climate.load().day(region, day) for region in climate.regions
    ]
    assert climate.regions == ("PHA", "SJC", "ZSC", "VCH", "JMV", "SMS")


def test_another_master_seed_draws_other_weather(climate: Climate) -> None:
    day = date(2024, 2, 14)
    with using_master_seed(1):
        other = Climate.load().day("PHA", day)
    assert other != climate.day("PHA", day)


def test_the_planted_severe_days_are_flagged_and_agree_with_their_kind(climate: Climate) -> None:
    planted = read("severe_days")
    assert planted
    for severe in planted:
        weather = climate.day(severe["region"], severe["day"])
        assert weather.severe, severe
        if severe["kind"] == "snow":
            assert weather.snow and weather.temp_mean_c <= -1.0 and weather.precip_mm >= 8.0
        elif severe["kind"] == "ice":
            assert -3.0 <= weather.temp_mean_c <= 0.0 and weather.precip_mm >= 3.0 and not weather.snow
        elif severe["kind"] == "storm":
            assert weather.precip_mm >= 25.0
        elif severe["kind"] == "flood":
            assert weather.precip_mm >= 60.0
    planted_days = {(severe["region"], severe["day"]) for severe in planted}
    day = date(2024, 1, 1)
    while ("PHA", day) in planted_days:
        day += timedelta(days=1)
    assert not climate.day("PHA", day).severe


def test_the_series_extends_to_any_date_and_a_later_reader_sees_the_same_day(climate: Climate) -> None:
    far = date(2031, 7, 1)
    straight = [climate.day("JMV", far - timedelta(days=n)) for n in range(10)]
    prolonged = Climate.load()  # a prolong's process: no memory of the build's days
    assert [prolonged.day("JMV", far - timedelta(days=n)) for n in range(10)] == straight


def test_the_january_mean_is_the_normal_within_a_degree(climate: Climate) -> None:
    normals = {(row["region"], int(row["month"])): float(row["temp_mean_c"]) for row in read("weather_normals")}
    for region in climate.regions:
        temps = [climate.day(region, date(year, 1, d)).temp_mean_c for year in range(2024, 2064) for d in range(1, 32)]
        assert mean(temps) == pytest.approx(normals[region, 1], abs=1.0), region


def test_snow_falls_only_on_a_cold_wet_day_and_the_summer_is_warm(climate: Climate) -> None:
    days = [date(2024, 1, 1) + timedelta(days=n) for n in range(731)]
    for region in climate.regions:
        for day in days:
            weather = climate.day(region, day)
            if weather.snow:
                assert weather.precip_mm > 0 and weather.temp_mean_c < 0.5
        july = [climate.day(region, date(2024, 7, d)).temp_mean_c for d in range(1, 32)]
        january = [climate.day(region, date(2024, 1, d)).temp_mean_c for d in range(1, 32)]
        assert mean(july) > mean(january) + 10


def test_the_rows_are_weather_daily_s_one_per_region(climate: Climate) -> None:
    rows = climate.rows(date(2024, 3, 1))
    assert len(rows) == 6
    assert all(tuple(row) == columns("weather_daily") for row in rows)
    assert len({row["region_id"] for row in rows}) == 6
