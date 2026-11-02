"""Season, weather and the week (G1 S1.2 T3, `04` S1): the 52-week curves, the cold weather's lift on the soups,
the channels' weeks; the null world's flat season."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig
from hartland_sim.loop.demand import CHANNELS, Seasons, _weeks, lines
from hartland_sim.loop.weather import Climate

STORIES = StoriesConfig.load()
LINES = lines()
CODES = [line.code for line in LINES]
S1 = STORIES.story("S1").parameters
SEASONS = Seasons.of(STORIES, LINES)
SMALL_RETAIL = CHANNELS.index("small_retail")


def _curve(code: str, channel: str = "small_retail") -> np.ndarray:
    return SEASONS.season[CHANNELS.index(channel), CODES.index(code), :52]


@pytest.mark.parametrize("code", list(S1["curves"]))
def test_each_line_s_curve_has_mean_one_and_its_tabled_amplitude(code: str) -> None:
    spec = S1["curves"][code]
    found = _curve(code)
    assert found.mean() == pytest.approx(1.0, abs=0.01)
    peak, trough = _weeks(spec["peak_weeks"]), _weeks(spec["trough_weeks"])
    if peak:
        ratio = found[np.array(peak) - 1].mean() / found[np.array(trough) - 1].mean()
    else:  # Decaf: flat but for a gentle winter swell
        ratio = found.max() / found.min()
    assert ratio == pytest.approx(spec["amplitude"], abs=0.01)


def test_the_tabled_amplitudes() -> None:
    def peak_over_trough(code: str) -> float:
        spec = S1["curves"][code]
        found = _curve(code)
        return float(
            found[np.array(_weeks(spec["peak_weeks"])) - 1].mean()
            / found[np.array(_weeks(spec["trough_weeks"])) - 1].mean()
        )

    assert round(peak_over_trough("CLASSIC"), 2) == 1.35
    assert round(peak_over_trough("GOLD"), 2) == round(peak_over_trough("ESPRESSO"), 2) == 1.6
    assert round(peak_over_trough("POLEVKY"), 2) == 2.2
    assert round(peak_over_trough("KELIMKY"), 2) == 1.8
    assert round(float(_curve("BEZKOFEIN").max() / _curve("BEZKOFEIN").min()), 2) == 1.1


def test_the_professional_lines_follow_their_channel() -> None:
    vending, horeca, shops = (_curve("PROFIKAVA", channel) for channel in ("vending", "horeca", "own_shops"))
    assert np.allclose(vending, 1.0)
    assert horeca[25] / horeca[5] == pytest.approx(1.2) and horeca.mean() == pytest.approx(1.0)
    assert shops[50] / shops[10] == pytest.approx(1.2)


def test_week_53_is_week_52_and_december_carries_the_gift() -> None:
    assert np.array_equal(SEASONS.season[:, :, 52], SEASONS.season[:, :, 51])
    december, november = SEASONS.of_day(date(2024, 12, 2)), SEASONS.of_day(date(2024, 11, 25))
    gold, classic = CODES.index("GOLD"), CODES.index("CLASSIC")
    week_49 = SEASONS.season[SMALL_RETAIL, gold, 48]
    assert december[SMALL_RETAIL, gold] == pytest.approx(week_49 * 1.3)
    assert december[SMALL_RETAIL, classic] == pytest.approx(SEASONS.season[SMALL_RETAIL, classic, 48])
    assert november[SMALL_RETAIL, gold] == pytest.approx(SEASONS.season[SMALL_RETAIL, gold, 47])


def test_the_soups_gain_four_percent_a_degree_below_five_on_the_weather_series() -> None:
    climate = Climate.load()
    day = date(2024, 1, 17)  # a planted ice day: at most 0 °C everywhere it struck
    temps = np.array([climate.day(region, day).temp_mean_c for region in climate.regions])
    lift = SEASONS.cold(temps)
    polevky, kelimky, gold = (CODES.index(code) for code in ("POLEVKY", "KELIMKY", "GOLD"))
    for r, temp in enumerate(temps):
        assert lift[r, polevky] == pytest.approx(1 + 0.04 * max(5 - temp, 0))
        assert lift[r, kelimky] == pytest.approx(1 + 0.03 * max(5 - temp, 0))
        assert lift[r, gold] == 1.0
    assert SEASONS.cold(np.array([6.0, 5.0]))[:, polevky].tolist() == [1.0, 1.0]


def test_each_channel_s_week() -> None:
    shops, web = SEASONS.weekday[CHANNELS.index("own_shops")], SEASONS.weekday[CHANNELS.index("web")]
    assert shops.mean() == pytest.approx(1.0) and web.mean() == pytest.approx(1.0)
    assert set(np.argsort(shops)[-2:]) == {4, 5} and set(np.argsort(web)[-2:]) == {4, 5}  # Friday, Saturday
    for channel in ("big_retail", "small_retail", "gas_ka", "vending", "horeca"):
        assert np.array_equal(SEASONS.weekday[CHANNELS.index(channel)], np.ones(7))  # the trade's shops sell daily
    trade = SEASONS.order_weekday
    assert trade.sum() == pytest.approx(1.0)
    assert set(np.argsort(trade)[-2:]) == {0, 1}  # the trade orders on Monday and Tuesday
    friday, monday = SEASONS.of_day(date(2024, 3, 8)), SEASONS.of_day(date(2024, 3, 4))
    assert friday[CHANNELS.index("web"), 0] / monday[CHANNELS.index("web"), 0] == pytest.approx(1.15 / 0.9)


def test_the_null_world_flattens_the_season_the_gift_and_the_weather() -> None:
    null = replace(STORIES, stories=tuple(replace(s, enabled=False) if s.story == "S1" else s for s in STORIES.stories))
    flat = Seasons.of(null, LINES)
    assert np.array_equal(flat.season, np.ones_like(flat.season))
    assert np.array_equal(flat.gift, np.ones(len(LINES)))
    assert np.array_equal(flat.cold(np.array([-10.0])), np.ones((1, len(LINES))))
    assert np.array_equal(flat.weekday, SEASONS.weekday)
