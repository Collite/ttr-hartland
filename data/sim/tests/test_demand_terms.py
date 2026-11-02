"""The promotion, display, discount and competitor terms (G1 S1.2 T4; `04` S2, S5, S6, A-6, A-10), on a small
hand-built demand: an eligible and an ineligible store in a rich okres, a store in VCH, the World's running items."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig
from hartland_sim.loop.demand import (
    CHANNELS,
    Demand,
    PromotionTerms,
    Spot,
    Store,
    StoreTerms,
    Terms,
    base,
    elasticity,
    eligible,
    lifted,
    lines,
    sku_cases,
    starts,
)
from hartland_sim.setup.run import Setup
from hartland_sim.world import OPEN, Agreement, Loan, Promotion, World

STORIES = StoriesConfig.load()
LINES = lines()
CODES = [line.code for line in LINES]
GOLD, CLASSIC = CODES.index("GOLD"), CODES.index("CLASSIC")
DAY = date(2025, 3, 10)
WARM = np.full(6, 10.0)  # every region at 10 °C: no weather term


def _store(store_class: str = "convenience") -> Store:
    return Store("M", "residential", 70.0, False, store_class)


SPOTS = [
    Spot(101, 1, "small_retail", ("small_retail", None), "CZ0100", "PHA", _store(), None),  # rich: 1.3
    Spot(102, 2, "horeca", ("horeca", None), "CZ0100", "PHA", _store("horeca"), None),
    Spot(103, 3, "small_retail", ("small_retail", None), "CZ0532", "VCH", _store(), None),
    Spot(104, 4, "small_retail", ("small_retail", None), "CZ0201", "SJC", _store("petrol"), "MOL"),
]
AFFLUENCE = {"CZ0100": 1.3, "CZ0532": 0.9, "CZ0201": 1.0}


def tiny(full: Demand, spots: list[Spot], affluence: dict[str, float]) -> Demand:
    """A demand over hand-built spots, with the opening's catalogue, regions and terms; one customer a spot."""
    groups = full.catalogue.groups
    return replace(
        full,
        spots=tuple(spots),
        point_ids=np.array([spot.delivery_point_id for spot in spots], dtype=np.int64),
        point_customer=np.arange(len(spots), dtype=np.int64),
        point_channel=np.array([CHANNELS.index(spot.channel) for spot in spots], dtype=np.int64),
        point_group=np.array([groups.index(spot.group) for spot in spots], dtype=np.int64),
        point_region=np.array([full.regions.index(spot.region) for spot in spots], dtype=np.int64),
        point_affluence=np.array([affluence[spot.okres] for spot in spots]),
        customer_ids=np.array([spot.customer_id for spot in spots], dtype=np.int64),
        customer_start=starts(np.arange(len(spots), dtype=np.int64), len(spots)),
        units=base(spots, full.lines, StoreTerms.of(STORIES), affluence),
        scale=np.ones(len(spots)),
    )


@pytest.fixture(scope="module")
def demand(opening: Setup) -> Demand:
    assert opening.world.demand is not None
    return tiny(opening.world.demand, SPOTS, AFFLUENCE)


def _shares(demand: Demand, day: date = DAY) -> np.ndarray:
    share, _, _ = demand.catalogue.shares(day, len(LINES))
    return np.asarray(share[demand.point_group])


def _world(**items: object) -> World:
    world = World(DAY)
    for name, value in items.items():
        setattr(world, name, value)
    return world


def _lift(terms: Terms, world: World, day: date = DAY) -> np.ndarray:
    """The day's terms with `world`'s items ÷ those of an empty world: the items' own effect."""
    return np.asarray(terms.today(world, day, WARM).line / terms.today(_world(), day, WARM).line)


def _promotion(**changes: object) -> Promotion:
    base = Promotion(
        promotion_id=1,
        code="PR-2025-03-GOLD-TV",
        line="GOLD",
        mechanic="price_off",
        valid_from=date(2025, 3, 3),
        valid_to=date(2025, 3, 30),
        tv=True,
        eligibility=(("channel", ("big_retail", "small_retail", "gas_ka", "web")),),
        discount_pct=20.0,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def test_a_tv_price_off_of_20_on_a_premium_line_in_a_rich_okres(demand: Demand) -> None:
    terms = Terms.of(demand, STORIES)
    today = _lift(terms, _world(promotions={1: _promotion()}))
    lift = 0.40 * 1.8 * 1.3**2.5 * 1.4
    assert today[0, GOLD] == pytest.approx(1 + lift)  # 1.93 = 1.3 ^ 2.5
    assert 1.3**2.5 == pytest.approx(1.93, abs=0.005)
    assert today[1, GOLD] == 1.0  # HoReCa: not eligible
    assert today[0, CLASSIC] == 1.0
    assert today[2, GOLD] == pytest.approx(1 + 0.40 * 1.8 * 0.9**2.5 * 1.4)


def test_the_two_weeks_after_carry_minus_ten_percent(demand: Demand) -> None:
    terms = Terms.of(demand, STORIES)
    world = _world(promotions={1: _promotion()})
    after = _lift(terms, world, date(2025, 3, 31))
    assert after[0, GOLD] == pytest.approx(0.9) and after[1, GOLD] == 1.0
    assert _lift(terms, world, date(2025, 4, 13))[0, GOLD] == pytest.approx(0.9)
    assert _lift(terms, world, date(2025, 4, 14))[0, GOLD] == 1.0


def test_a_promoted_sku_lifts_itself_and_its_siblings_lose_thirty_percent_of_it(demand: Demand) -> None:
    sku = "2931000000036"  # Classic 200 g
    promotion = _promotion(line="CLASSIC", tv=False, discount_pct=15.0, target_sku=sku, code="PR-2024-10-CLASSIC-200")
    terms = Terms.of(demand, STORIES)
    today = terms.today(_world(promotions={1: promotion}), DAY, WARM)
    assert _lift(terms, _world(promotions={1: promotion}))[0, CLASSIC] == 1.0  # the line is not lifted as a whole
    (lift,) = today.skus
    assert demand.catalogue.sku_codes[lift.sku] == sku and lift.line == CLASSIC and lift.cannibalises
    assert lift.uplift[0] == pytest.approx(0.29 * 1.3**0.5)  # no TV: affluence ^ 0.5, non-premium 1.0
    assert lift.uplift[1] == 0.0
    share = _shares(demand)
    ones = np.ones((len(SPOTS), len(LINES)))
    cases, extras = lifted(ones, today.skus, share, 0.30)
    found = sku_cases(cases, extras, share, demand.catalogue.line)
    sigma = share[0, lift.sku]
    extra = sigma * lift.uplift[0]
    assert found[0, lift.sku] == pytest.approx(sigma + extra)
    siblings = [
        s for s in range(len(found[0])) if demand.catalogue.line[s] == CLASSIC and s != lift.sku and share[0, s] > 0
    ]
    assert found[0, siblings].sum() == pytest.approx((1 - sigma) - 0.30 * extra)
    kept = found[0, siblings] / share[0, siblings]
    assert np.allclose(kept, kept[0])  # each sibling gives up the same part of its cases
    assert found[1, demand.catalogue.line == CLASSIC].sum() == pytest.approx(1.0)  # no lift, no loss


def test_a10_lifts_its_sku_45_and_its_named_sibling_loses_30(demand: Demand) -> None:
    a10 = STORIES.anomaly("A-10").parameters
    promotion = _promotion(line="CLASSIC", tv=False, discount_pct=15.0, target_sku=a10["sku"], code=a10["promotion"])
    today = Terms.of(demand, STORIES).today(_world(promotions={1: promotion}), DAY, WARM)
    lifts = {demand.catalogue.sku_codes[lift.sku]: lift for lift in today.skus}
    assert lifts[a10["sku"]].uplift[0] == pytest.approx(0.45) and not lifts[a10["sku"]].cannibalises
    assert lifts[a10["sibling"]].uplift[0] == pytest.approx(-0.30)
    share = _shares(demand)
    cases, extras = lifted(np.ones((len(SPOTS), len(LINES))), today.skus, share, 0.30)
    found = sku_cases(cases, extras, share, demand.catalogue.line)
    sku, sibling = (demand.catalogue.sku_codes.index(a10[key]) for key in ("sku", "sibling"))
    assert found[0, sku] == pytest.approx(share[0, sku] * 1.45)
    assert found[0, sibling] == pytest.approx(share[0, sibling] * 0.70)
    others = (
        (demand.catalogue.line == CLASSIC) & (np.arange(len(found[0])) != sku) & (np.arange(len(found[0])) != sibling)
    )
    assert np.allclose(found[0, others], share[0, others])


def test_a_display_adds_its_uplift_while_the_loan_runs(demand: Demand) -> None:
    terms = Terms.of(demand, STORIES)
    world = _world(loans={7: Loan(101, "GOLD", 0.35, date(2025, 3, 1), date(2025, 3, 20))})
    assert _lift(terms, world)[0, GOLD] == pytest.approx(1.35)
    assert _lift(terms, world, date(2025, 3, 20))[0, GOLD] == 1.0
    assert _lift(terms, world, date(2025, 2, 28))[0, GOLD] == 1.0


def test_an_agreed_discount_adds_elasticity_times_its_percent(demand: Demand) -> None:
    terms = Terms.of(demand, STORIES)
    world = _world(agreements={9: Agreement(1, "CLASSIC", 8.0, date(2025, 1, 1), OPEN)})
    e = terms.elasticity[0, CLASSIC]
    assert e in (2.0, 0.4)
    assert _lift(terms, world)[0, CLASSIC] == pytest.approx(1 + e * 0.08)
    assert _lift(terms, world)[2, CLASSIC] == 1.0


def test_thirty_percent_of_the_customer_lines_are_elastic(opening: Setup) -> None:
    assert opening.world.demand is not None
    found = elasticity(opening.world.demand, STORIES)
    assert set(np.unique(found)) == {0.4, 2.0}
    assert (found == 2.0).mean() == pytest.approx(0.30, abs=0.02)
    again = elasticity(opening.world.demand, STORIES)
    assert np.array_equal(found, again)
    null = replace(STORIES, stories=tuple(replace(s, enabled=False) if s.story == "S6" else s for s in STORIES.stories))
    assert np.allclose(elasticity(opening.world.demand, null), 0.3 * 2.0 + 0.7 * 0.4)


def test_a6_cuts_gold_twelve_percent_in_vch_for_its_weeks(demand: Demand) -> None:
    a6 = STORIES.anomaly("A-6").parameters
    terms = Terms.of(demand, STORIES)
    day = a6["from"] + timedelta(days=3)
    inside = terms.today(_world(), day, WARM).line
    plain = Terms.of(demand, replace(STORIES, anomalies=tuple(replace(a, enabled=False) for a in STORIES.anomalies)))
    outside = plain.today(_world(), day, WARM).line
    assert inside[2, GOLD] / outside[2, GOLD] == pytest.approx(0.88)
    assert inside[0, GOLD] == outside[0, GOLD] and inside[2, CLASSIC] == outside[2, CLASSIC]
    after = a6["until"] + timedelta(days=1)
    assert terms.today(_world(), after, WARM).line[2, GOLD] == plain.today(_world(), after, WARM).line[2, GOLD]


def test_eligibility_reads_every_scope_kind(demand: Demand) -> None:
    petrol = _promotion(eligibility=(("channel", ("gas_ka", "small_retail")), ("store_class", ("petrol",))))
    assert eligible(petrol, demand).tolist() == [False, False, False, True]
    chain = _promotion(eligibility=(("chain", ("MOL",)),))
    assert eligible(chain, demand).tolist() == [False, False, False, True]
    region = _promotion(eligibility=(("region", ("VCH", "PHA")),))
    assert eligible(region, demand).tolist() == [True, True, True, False]
    phase = _promotion(eligibility=(("phase", ("vip",)),))
    assert eligible(phase, demand, ["vip", "regular", "regular", "vip"]).tolist() == [True, False, False, True]


def test_the_null_world_flattens_premium_affluence_and_tv() -> None:
    null = replace(STORIES, stories=tuple(replace(s, enabled=False) if s.story == "S2" else s for s in STORIES.stories))
    flat = PromotionTerms.of(null)
    assert flat.factor(_promotion(), LINES[GOLD], np.array([0.8, 1.3])).tolist() == [1.0, 1.0]
    assert flat.price_off(20) == 0.40 and flat.dip == pytest.approx(0.9)
    assert CHANNELS[0] == "big_retail"


def test_setup_seeds_the_calendar_into_the_world(opening: Setup) -> None:
    promotions = list(opening.world.promotions.values())
    years = {promotion.valid_from.year for promotion in promotions}
    assert years == {2023, 2024, 2025, 2026}
    assert len(promotions) == 4 * 25
    assert any(p.code == "PR-2025-03-GOLD-TV" and p.tv and p.mechanic == "price_off" for p in promotions)
