"""The price lists (G1 S1.3 T1, `01` §5.1): the standard list of Q1-2024 at the catalogue's base prices, each channel's
list of Jan-2024 at the standard × its factor with Marketing's hero overrides, both with their validity; setup gives the
`World` the versions in force on the warm-up's first day. The rows are the period ends' (`test_quarter_end`,
`test_month_end`, ⚑R213-A)."""

from __future__ import annotations

from datetime import date

import pytest

from hartland_sim.config import WorldConfig
from hartland_sim.dates import month_end, quarter_end, quarter_start
from hartland_sim.loop.demand import CHANNELS
from hartland_sim.loop.pricing import channel_list, standard_list
from hartland_sim.setup.run import Setup

WORLD = WorldConfig.load()
FACTORS = {str(channel): factor for channel, factor in WORLD.pricing.channel_factor.items()}
HEROES = {(hero.sku, str(hero.channel)): hero.factor for hero in WORLD.pricing.hero_overrides}


@pytest.mark.parametrize(
    ("day", "first", "last"),
    [
        (date(2024, 1, 1), date(2024, 1, 1), date(2024, 3, 31)),
        (date(2024, 5, 31), date(2024, 4, 1), date(2024, 6, 30)),
        (date(2024, 12, 31), date(2024, 10, 1), date(2024, 12, 31)),
    ],
)
def test_a_quarter_runs_from_its_first_day_to_its_last(day: date, first: date, last: date) -> None:
    assert (quarter_start(day), quarter_end(day)) == (first, last)


def test_a_month_ends_on_its_last_day() -> None:
    assert [month_end(date(2024, month, 10)).day for month in (1, 2, 4, 12)] == [31, 29, 30, 31]


def test_the_standard_list_is_the_catalogue_s_base_price_for_the_skus_of_the_quarter(opening: Setup) -> None:
    standard = standard_list(opening.portfolio.skus, date(2024, 2, 14))
    first, last = date(2024, 1, 1), date(2024, 3, 31)
    assert (standard.code, standard.valid_from, standard.valid_to) == ("SC-2024-Q1", first, last)
    sold = [sku for sku in opening.portfolio.skus if sku.launched_on <= last and sku.last_day >= first]
    assert list(standard.prices) == [sku.sku_id for sku in sold]
    for sku in sold:
        price = standard.prices[sku.sku_id]
        assert price.case == sku.base_price_case
        assert price.piece == round(sku.base_price_case / sku.units_per_case, 2)
        assert price.purchase_case == sku.cogs_case
    later = [sku for sku in opening.portfolio.skus if sku.launched_on > date(2024, 3, 31)]
    assert later, "the catalogue launches SKUs after Q1-2024 (A-2's pouch among them)"
    assert not set(standard.prices) & {sku.sku_id for sku in later}


@pytest.mark.parametrize("channel", CHANNELS)
def test_a_channel_list_is_the_standard_times_its_factor_with_the_hero_overrides(opening: Setup, channel: str) -> None:
    skus = opening.portfolio.skus
    standard = standard_list(skus, date(2024, 1, 1))
    listed = channel_list(standard, skus, WORLD, channel, date(2024, 1, 20))
    assert (listed.channel, listed.code, listed.valid_from, listed.valid_to) == (
        channel,
        None,
        date(2024, 1, 1),
        date(2024, 1, 31),
    )
    assert listed.factor == FACTORS[channel]
    assert list(listed.prices) == list(standard.prices)
    by_id = {sku.sku_id: sku for sku in skus}
    for sku_id, price in listed.prices.items():
        sku = by_id[sku_id]
        factor = HEROES.get((sku.code, channel), FACTORS[channel])
        assert price.case == round(standard.prices[sku_id].case * factor, 2)
        assert price.piece == round(price.case / sku.units_per_case, 2)
        assert price.is_override == ((sku.code, channel) in HEROES)


def test_a_channel_list_needs_the_standard_list_of_its_month(opening: Setup) -> None:
    skus = opening.portfolio.skus
    stale = opening.world.standard_list
    assert stale is not None and stale.code == "SC-2023-Q1"
    day = date(2024, 4, 2)
    with pytest.raises(ValueError, match="roll the standard list first"):
        channel_list(stale, skus, WORLD, "small_retail", day)
    launched = [sku.sku_id for sku in skus if sku.launched_on == day]
    assert len(launched) == 3
    listed = channel_list(standard_list(skus, day), skus, WORLD, "small_retail", day)
    assert set(launched) <= set(listed.prices)


def test_the_hero_overrides_reach_their_channels_only(opening: Setup) -> None:
    skus = opening.portfolio.skus
    standard = standard_list(skus, date(2024, 1, 1))
    codes = {sku.sku_id: sku.code for sku in skus}
    overridden = set()
    for channel in CHANNELS:
        listed = channel_list(standard, skus, WORLD, channel, date(2024, 1, 1))
        overridden |= {(codes[sku_id], channel) for sku_id, price in listed.prices.items() if price.is_override}
    assert overridden == set(HEROES)


def test_the_world_holds_the_versions_in_force_on_the_warm_up_s_first_day(opening: Setup) -> None:
    world = opening.world
    assert world.standard_list is not None
    assert (world.standard_list.code, world.standard_list.valid_from) == ("SC-2023-Q1", date(2023, 1, 1))
    assert list(world.channel_lists) == list(CHANNELS)
    for channel, listed in world.channel_lists.items():
        assert (listed.channel, listed.valid_from, listed.valid_to) == (channel, date(2023, 1, 1), date(2023, 1, 31))
