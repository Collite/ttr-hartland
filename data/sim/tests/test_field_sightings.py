"""Competitor sightings (G2 S2.1 T5, `loop.field.sight`): at `field.sightings.share` of the recorded visits that
found the shop (review-231 R2) a
competitor's activity on one of the store's observed lines — its product of that line (`competitor_product.csv`'s
equivalent), the kind by its shares, the price seen against the product's typical price — confirmed at its share; a
competitor is likelier in its home region, and A-6's Kavalír promotes Gold in East Bohemia from week 45 of 2025 — its
only promotions."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date, timedelta

import pytest

from hartland_sim.checks.czech import assert_czech
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import read
from hartland_sim.loop import field
from hartland_sim.loop.field import Crew, Sighting, Visit
from hartland_sim.setup.rows import columns
from hartland_sim.setup.run import Setup

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
SIGHTINGS = WORLD.field.sightings
A6 = STORIES.anomaly("A-6").parameters
DAYS = [date(2023, 1, 2) + timedelta(days=k) for k in range(42) if (date(2023, 1, 2) + timedelta(days=k)).weekday() < 5]
PRODUCTS = {row["code"]: row for row in read("competitor_product")}
LINE_OF = {row["code"]: row["product_line"] for row in read("product")}


@pytest.fixture(scope="module")
def crew(opening: Setup) -> Crew:
    return Crew.of(opening.world, WORLD, STORIES)


def _visits(opening: Setup, crew: Crew, day: date) -> list[Visit]:
    world = opening.world
    return field.visit(world, crew, day, field.calls(world, crew, day, field.plan(world, day)))


@pytest.fixture(scope="module")
def seen(opening: Setup, crew: Crew) -> list[tuple[date, list[Visit], list[Sighting]]]:
    return [
        (day, visits, field.sight(opening.world, crew, day, visits))
        for day in DAYS
        for visits in [_visits(opening, crew, day)]
    ]


def test_a_recorded_visit_sees_a_competitor_at_its_share(seen: list[tuple[date, list[Visit], list[Sighting]]]) -> None:
    inside = {v.visit_id: v for _, visits, _ in seen for v in visits if v.recorded and v.outcome not in field.SHUT}
    sightings = [s for _, _, day_sightings in seen for s in day_sightings]
    assert all(s.visit_id in inside for s in sightings)  # never a shop closed or gone
    assert len(sightings) / len(inside) == pytest.approx(SIGHTINGS.share, abs=0.006)
    assert Counter(s.visit_id for s in sightings).most_common(1)[0][1] == 1


def test_a_sighting_is_of_a_product_of_a_line_the_store_observes_at_a_plausible_price(
    seen: list[tuple[date, list[Visit], list[Sighting]]], opening: Setup
) -> None:
    kinds: Counter[str] = Counter()
    confirmed = 0
    for _, visits, sightings in seen:
        by_id = {v.visit_id: v for v in visits}
        for s in sightings:
            product = PRODUCTS[s.product]
            assert product["competitor"] == s.competitor
            assert (
                LINE_OF[product["equivalent_product"]]
                in WORLD.field.observed_lines[by_id[s.visit_id].tourplan.store_class]
            )
            low, high = SIGHTINGS.price[s.kind]
            assert low * float(product["typical_price"]) - 1 <= s.price <= high * float(product["typical_price"]) + 1
            assert s.kind != "promotion" or s.competitor not in SIGHTINGS.promotions_planted
            assert_czech(s.description)
            kinds[s.kind] += 1
            confirmed += s.confirmed
    n = sum(kinds.values())
    assert set(kinds) <= set(SIGHTINGS.kinds) and n > 100
    assert confirmed / n == pytest.approx(SIGHTINGS.confirmed, abs=0.1)


def test_a_competitor_is_likelier_in_its_home_region(
    seen: list[tuple[date, list[Visit], list[Sighting]]], crew: Crew, opening: Setup
) -> None:
    demand = opening.world.demand
    assert demand is not None
    every = replace(crew, field=replace(crew.field, sightings=replace(SIGHTINGS, share=1.0)))  # every visit sees one
    ((home, region),) = SIGHTINGS.home.items()
    at: Counter[bool] = Counter()
    of: Counter[bool] = Counter()
    for day, visits, _ in seen:
        by_id = {v.visit_id: v for v in visits}
        for s in field.sight(opening.world, every, day, visits):
            inside = demand.spots[crew.points[by_id[s.visit_id].tourplan.delivery_point_id]].region == region
            at[inside] += 1
            of[inside] += s.competitor == home
    assert at[True] > 300
    assert of[True] / at[True] > 1.5 * of[False] / at[False]


def test_a_6_kavalir_promotes_gold_in_east_bohemia_from_week_45(opening: Setup, crew: Crew) -> None:
    first, last = date.fromisoformat(str(A6["sightings_from"])), date.fromisoformat(str(A6["until"]))
    demand = opening.world.demand
    assert demand is not None
    counts = {}
    for day in (first - timedelta(days=7), first, last, last + timedelta(days=1)):
        visits = [v for d in DAYS[5:10] for v in _visits(opening, crew, d) if v.recorded]
        planted = [
            s
            for s in field.sight(opening.world, crew, day, visits)
            if s.competitor == A6["competitor"] and s.kind == "promotion"
        ]
        for s in planted:
            v = next(v for v in visits if v.visit_id == s.visit_id)
            assert demand.spots[crew.points[v.tourplan.delivery_point_id]].region == A6["region"]
            assert LINE_OF[PRODUCTS[s.product]["equivalent_product"]] == A6["line"]
            assert s.planted and s.description in field.PLANTED_SIGHTINGS
        counts[day] = len(planted)
    assert counts[first - timedelta(days=7)] == counts[last + timedelta(days=1)] == 0
    assert counts[first] >= 10 and counts[last] >= 10


def test_a_sighting_row_is_its_columns(seen: list[tuple[date, list[Visit], list[Sighting]]]) -> None:
    s = next(s for _, _, sightings in seen for s in sightings)
    row = field.sighting_row(s)
    assert tuple(row) == columns("competitor_sighting")
    assert row["competitor_sighting_id"] == row_id(
        "competitor_sighting", visit=s.visit_id, competitor=s.competitor, seq=1
    )
    assert (row["competitor_id"], row["competitor_product_id"]) == (
        row_id("competitor", code=s.competitor),
        row_id("competitor_product", code=s.product),
    )
    assert (row["activity_kind"], row["price_seen"], row["confirmed"]) == (s.kind, s.price, s.confirmed)
