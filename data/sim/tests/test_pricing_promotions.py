"""Promotions (G1 S1.3 T5, `01` §5.4, ⚑NH-8): one golden per mechanic on the committed calendar's promotions — price
off, conditional price off, 1 + 1 (and 2 + 1, and in pieces), free goods of another SKU — then eligibility (channel,
chain, region, store class, phase, validity) and its parity with demand's, the `not_with` rule (the earlier-issued
promotion stays; one that gives nothing excludes nothing), the mandatory flag, and the calendar's rows setup writes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date

import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.loop.demand import Spot, eligible
from hartland_sim.loop.pricing import Draft, Rules, Term, admits, apply, promotions, taken
from hartland_sim.setup.promotions import status
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, Customer, Promotion, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()


@pytest.fixture(scope="module")
def world(opening: Setup) -> World:
    return Checkpoint.of(opening.world).restore()


@pytest.fixture(scope="module")
def rules(opening: Setup) -> Rules:
    return Rules.of(opening.world, WORLD, STORIES)


def _id(code: str) -> int:
    return row_id("promotion", code=code)


def _point(world: World, rules: Rules, test: Callable[[Customer, Spot], bool]) -> tuple[Customer, int]:
    """The first active customer and its point that pass `test`."""
    assert world.demand is not None
    for spot in world.demand.spots:
        customer = world.customers[spot.customer_id]
        if customer.status == "active" and test(customer, spot):
            return customer, spot.delivery_point_id
    raise LookupError("no such point")


def _shop(world: World, rules: Rules, store_class: str = "convenience") -> tuple[Customer, int]:
    return _point(
        world,
        rules,
        lambda c, s: c.channel == "small_retail" and s.store is not None and s.store.store_class == store_class,
    )


def _sku(opening: Setup, line: str) -> int:
    return next(sku.sku_id for sku in opening.portfolio.skus if sku.line == line)


def _code(rules: Rules, sku_code: str) -> int:
    return rules.sku_ids[sku_code]


def _line(sku_id: int, cases: float, units: int = 1) -> tuple[int, float, float]:
    return sku_id, cases, cases * units


# ── one golden per mechanic ──────────────────────────────────────────────────


def test_price_off_is_one_promotion_row_from_marketing(opening: Setup, world: World, rules: Rules) -> None:
    shop, point = _shop(world, rules)
    gold, classic = _sku(opening, "GOLD"), _sku(opening, "CLASSIC")
    day = date(2025, 3, 5)  # GOLD-TV runs from week 10's Monday, CLASSIC-TV from week 12's
    drafts = promotions(world, rules, shop, point, day, [_line(gold, 3), _line(classic, 2)])
    term = Term("promotion", "PR-2025-03-GOLD-TV", "marketing", pct=20, promotion_id=_id("PR-2025-03-GOLD-TV"))
    assert drafts == [Draft(gold, 3, 3, promotions=(term,)), Draft(classic, 2, 2)]
    (row,) = apply(612.0, 3, drafts[0].promotions)
    assert (row.kind, row.code, row.budget, row.pct, row.amount_case) == (
        "promotion",
        "PR-2025-03-GOLD-TV",
        "marketing",
        20,
        122.4,
    )
    assert row.promotion_id == _id("PR-2025-03-GOLD-TV") and row.price_agreement_id is None


def test_conditional_price_off_fires_at_the_condition_s_cases(world: World, rules: Rules) -> None:
    shop, point = _shop(world, rules)
    soup, reward = _code(rules, "2931000000586"), _code(rules, "2931000000739")  # POLEVKY-ZIMA: 4 × 586 → 739 −20 %
    day = date(2025, 1, 20)
    met = promotions(world, rules, shop, point, day, [_line(soup, 4), _line(reward, 2)])
    term = Term(
        "promotion", "PR-2025-01-POLEVKY-ZIMA", "marketing", pct=20, promotion_id=_id("PR-2025-01-POLEVKY-ZIMA")
    )
    assert met == [Draft(soup, 4, 4), Draft(reward, 2, 2, promotions=(term,), mandatory=("PR-2025-01-POLEVKY-ZIMA",))]
    short = promotions(world, rules, shop, point, day, [_line(soup, 3), _line(reward, 2)])
    assert short == [Draft(soup, 3, 3), Draft(reward, 2, 2)]


def test_one_plus_one_splits_off_a_free_line_at_the_list_price(opening: Setup, world: World, rules: Rules) -> None:
    shop, point = _shop(world, rules)
    gold = _sku(opening, "GOLD")
    units = rules.units[gold]
    day = date(2025, 9, 15)  # GOLD-11
    paid, free = promotions(world, rules, shop, point, day, [_line(gold, 5, units)])
    gift = Term("free_goods", "PR-2025-09-GOLD-11", "marketing", pct=100, promotion_id=_id("PR-2025-09-GOLD-11"))
    assert paid == Draft(gold, 3, 3 * units)
    assert free == Draft(gold, 2, 2 * units, True, (gift,))
    (row,) = apply(612.0, 2, free.promotions)
    assert (row.kind, row.amount_case, row.amount_line, row.budget) == ("free_goods", 612.0, 1224.0, "marketing")
    one = promotions(world, rules, shop, point, day, [_line(gold, 1, units)])
    assert one == [Draft(gold, 1, units)]


def test_two_plus_one_for_the_chains_it_names(opening: Setup, world: World, rules: Rules) -> None:
    tesco, point = _point(world, rules, lambda c, s: c.channel == "big_retail" and s.chain == "TESCO")
    lidl, other = _point(world, rules, lambda c, s: c.channel == "big_retail" and s.chain == "LIDL")
    classic = _sku(opening, "CLASSIC")
    units = rules.units[classic]
    day = date(2025, 7, 1)  # CLASSIC-KA, 2 + 1, Tesco and Albert
    paid, free = promotions(world, rules, tesco, point, day, [_line(classic, 7, units)])
    assert (paid.cases, free.cases, free.is_free_goods) == (5, 2, True)
    assert free.promotions[0].code == "PR-2025-06-CLASSIC-KA"
    assert promotions(world, rules, lidl, other, day, [_line(classic, 7, units)]) == [Draft(classic, 7, 7 * units)]


def test_a_one_plus_one_in_pieces_splits_the_pieces(world: World, rules: Rules) -> None:
    web, point = _point(world, rules, lambda c, s: c.channel == "web")
    premium = next(sku for sku, line in rules.sku_line.items() if line == "PREMIUM")
    units = rules.units[premium]
    day = date(2025, 10, 20)  # PREMIUM-11, web among its channels
    paid, free = promotions(world, rules, web, point, day, [(premium, 5 / units, 5.0)])
    assert (paid.pieces, paid.cases, free.pieces, free.cases) == (3, 3 / units, 2, 2 / units)


def test_free_goods_of_another_sku_add_its_line(world: World, rules: Rules) -> None:
    shop, point = _shop(world, rules)
    noodles, reward = _code(rules, "2931000000951"), _code(rules, "2931000000968")  # NUDLE-JARO: 2 × 951 → 1 × 968
    day = date(2025, 4, 10)
    gift = Term("free_goods", "PR-2025-04-NUDLE-JARO", "marketing", pct=100, promotion_id=_id("PR-2025-04-NUDLE-JARO"))
    drafts = promotions(world, rules, shop, point, day, [_line(noodles, 5)])
    units = rules.units[reward]
    assert drafts == [Draft(noodles, 5, 5), Draft(reward, 2, 2 * units, True, (gift,), ("PR-2025-04-NUDLE-JARO",))]
    assert promotions(world, rules, shop, point, day, [_line(noodles, 1)]) == [Draft(noodles, 1, 1)]


# ── eligibility ──────────────────────────────────────────────────────────────


def test_a_channel_outside_the_promotion_s_gets_nothing(opening: Setup, world: World, rules: Rules) -> None:
    venue, point = _point(world, rules, lambda c, s: c.channel == "horeca")
    gold = _sku(opening, "GOLD")
    assert promotions(world, rules, venue, point, date(2025, 3, 5), [_line(gold, 3)]) == [Draft(gold, 3, 3)]


def test_a_store_class_scope_admits_its_stores_only(opening: Setup, world: World, rules: Rules) -> None:
    cups = _sku(opening, "KELIMKY")
    day = date(2025, 2, 12)  # KELIMKY-ZIMA: small retail and gas KA, petrol stations only
    station, at = _shop(world, rules, "petrol")
    shop, point = _shop(world, rules, "convenience")
    assert (
        promotions(world, rules, station, at, day, [_line(cups, 2)])[0].promotions[0].code == "PR-2025-02-KELIMKY-ZIMA"
    )
    assert promotions(world, rules, shop, point, day, [_line(cups, 2)]) == [Draft(cups, 2, 2)]


@pytest.mark.parametrize("kind", ["region", "phase"])
def test_a_region_or_phase_scope_admits_its_own(opening: Setup, world: World, rules: Rules, kind: str) -> None:
    shop, point = _shop(world, rules)
    assert world.demand is not None
    spot = world.demand.spots[rules.points[point]]
    day = date(2025, 7, 20)
    held = spot.region if kind == "region" else shop.phase_on(day)
    fixture = Promotion(1, "PR-2025-07-ESPRESSO-TEST", "ESPRESSO", "price_off", day, day, False, ((kind, (held,)),), 5)
    other = replace(fixture, promotion_id=2, code="PR-2025-07-ESPRESSO-JINAM", eligibility=((kind, ("nowhere",)),))
    espresso = _sku(opening, "ESPRESSO")
    test_world = replace(world, promotions={1: fixture, 2: other})
    (draft,) = promotions(test_world, rules, shop, point, day, [_line(espresso, 1)])
    assert [term.code for term in draft.promotions] == ["PR-2025-07-ESPRESSO-TEST"]


def test_a_promotion_runs_from_its_first_day_to_its_last(opening: Setup, world: World, rules: Rules) -> None:
    shop, point = _shop(world, rules)
    gold = _sku(opening, "GOLD")
    promotion = world.promotions[_id("PR-2025-03-GOLD-TV")]
    for day, applies in [
        (promotion.valid_from - date.resolution, False),
        (promotion.valid_from, True),
        (promotion.valid_to, True),
        (promotion.valid_to + date.resolution, False),
    ]:
        (draft, *_) = promotions(world, rules, shop, point, day, [_line(gold, 2)])
        assert ("PR-2025-03-GOLD-TV" in [term.code for term in draft.promotions]) is applies, day


def test_admits_and_demand_s_eligible_admit_the_same_points(world: World) -> None:
    """Demand lifts the points `eligible` admits and pricing discounts the orders `admits` admits: on each 2024–2025
    promotion's first day, and for a fixture of every scope kind the calendar does not use, they agree point by
    point."""
    demand = world.demand
    assert demand is not None
    calendar = [p for p in world.promotions.values() if p.valid_from.year in (2024, 2025)]
    day = date(2025, 7, 20)
    scopes = [("region", (demand.regions[0],)), ("phase", ("regular",)), ("phase", ("occasional", "vip"))]
    fixtures = [
        Promotion(-n, f"PR-2025-07-TEST-{n}", "GOLD", "price_off", day, day, False, (scope,), 5)
        for n, scope in enumerate(scopes, start=1)
    ]
    owners = [world.customers[spot.customer_id] for spot in demand.spots]
    kinds = set()
    for promotion in [*calendar, *fixtures]:
        on = promotion.valid_from
        mask = eligible(promotion, demand, [world.customers[int(c)].phase_on(on) for c in demand.customer_ids])
        admitted = [admits(promotion, spot, owner, on) for spot, owner in zip(demand.spots, owners, strict=True)]
        assert admitted == mask.tolist(), promotion.code
        if 0 < mask.sum() < len(mask):
            kinds |= {kind for kind, _ in promotion.eligibility}
    assert kinds == {"channel", "chain", "region", "store_class", "phase"}


# ── the exclusions ───────────────────────────────────────────────────────────


def test_an_order_keeps_the_earlier_issued_of_two_excluded_promotions(
    opening: Setup, world: World, rules: Rules
) -> None:
    shop, point = _shop(world, rules)
    gold, classic = _sku(opening, "GOLD"), _sku(opening, "CLASSIC")
    day = date(2025, 3, 20)  # GOLD-TV (from 03-03) and CLASSIC-TV (from 03-17) overlap; they exclude each other
    assert "PR-2025-03-CLASSIC-TV" in world.promotions[_id("PR-2025-03-GOLD-TV")].excludes
    both = promotions(world, rules, shop, point, day, [_line(gold, 2), _line(classic, 2)])
    assert [[t.code for t in d.promotions] for d in both] == [["PR-2025-03-GOLD-TV"], []]
    alone = promotions(world, rules, shop, point, day, [_line(classic, 2)])
    assert [[t.code for t in d.promotions] for d in alone] == [["PR-2025-03-CLASSIC-TV"]]


def test_a_line_with_two_excluded_promotions_keeps_the_earlier_issued(
    opening: Setup, world: World, rules: Rules
) -> None:
    shop, point = _shop(world, rules)
    gold = _sku(opening, "GOLD")
    day = date(2025, 7, 20)
    channel = (("channel", ("small_retail",)),)
    first = Promotion(1, "PR-2025-07-GOLD-B", "GOLD", "price_off", date(2025, 7, 1), day, False, channel, 5)
    second = Promotion(2, "PR-2025-07-GOLD-A", "GOLD", "price_off", date(2025, 7, 7), day, False, channel, 10)
    first, second = replace(first, excludes=(second.code,)), replace(second, excludes=(first.code,))
    (draft,) = promotions(replace(world, promotions={2: second, 1: first}), rules, shop, point, day, [_line(gold, 2)])
    assert [term.code for term in draft.promotions] == ["PR-2025-07-GOLD-B"]
    unexcluded = {1: replace(first, excludes=()), 2: replace(second, excludes=())}
    (stacked,) = promotions(replace(world, promotions=unexcluded), rules, shop, point, day, [_line(gold, 2)])
    assert [term.code for term in stacked.promotions] == ["PR-2025-07-GOLD-B", "PR-2025-07-GOLD-A"]


def test_a_one_plus_one_that_frees_nothing_excludes_nothing(opening: Setup, world: World, rules: Rules) -> None:
    shop, point = _shop(world, rules)
    gold, mixes = _sku(opening, "GOLD"), _sku(opening, "SMESI")
    day = date(2025, 9, 15)  # GOLD-11 (from 09-08) and SMESI-11 (from 09-15): both 1 + 1, excluding each other
    assert "PR-2025-09-SMESI-11" in world.promotions[_id("PR-2025-09-GOLD-11")].excludes

    def priced(gold_cases: int) -> list[tuple[int, float, bool, list[str]]]:
        lines = [_line(gold, gold_cases, rules.units[gold]), _line(mixes, 4, rules.units[mixes])]
        drafts = promotions(world, rules, shop, point, day, lines)
        return [(d.sku_id, d.cases, d.is_free_goods, [t.code for t in d.promotions]) for d in drafts]

    assert priced(1) == [  # one Gold case frees nothing: SMESI-11 runs
        (gold, 1, False, []),
        (mixes, 2, False, []),
        (mixes, 2, True, ["PR-2025-09-SMESI-11"]),
    ]
    assert priced(2) == [  # two do: the earlier-issued GOLD-11 stays
        (gold, 1, False, []),
        (mixes, 4, False, []),
        (gold, 1, True, ["PR-2025-09-GOLD-11"]),
    ]


def test_taken_goes_in_issue_order_and_ties_by_code() -> None:
    day = date(2025, 5, 1)
    a = Promotion(1, "PR-B", "GOLD", "price_off", day, day, False, (), 5, excludes=("PR-A",))
    b = Promotion(2, "PR-A", "CLASSIC", "price_off", day, day, False, (), 5)
    c = Promotion(3, "PR-C", "SMESI", "price_off", day - date.resolution, day, False, (), 5, excludes=("PR-A",))
    assert [p.code for _, p in taken([(1, a), (2, b), (3, c)])] == ["PR-C", "PR-B"]


# ── the calendar's rows ──────────────────────────────────────────────────────


def test_setup_writes_the_calendar_from_the_first_written_day_with_its_campaigns(opening: Setup) -> None:
    rows = opening.tables["promotion"]
    first, years = WORLD.history.first_day, range(WORLD.history.first_day.year, WORLD.history.until.year + 1)
    assert [row["code"] for row in rows if row["valid_from"] < first] == ["PR-2023-12-PREMIUM-VANOCE"]  # into January
    assert all(row["valid_to"] >= first for row in rows)
    later = [row for row in rows if row["valid_from"] >= first]
    assert {row["valid_from"].year for row in later} == set(years)
    assert len(later) == 25 * len(years)
    campaigns = {row["campaign_id"]: row for row in opening.tables["campaign"]}
    for row in rows:
        campaign = campaigns[row["campaign_id"]]
        assert campaign["code"] == "KP-" + row["code"].removeprefix("PR-")
        assert (campaign["valid_from"], campaign["valid_to"]) == (row["valid_from"], row["valid_to"])
        assert campaign["budget"] == WORLD.promotions.campaign_budget["tv" if campaign["has_tv"] else "other"]
        assert campaign["owner_employee_id"] == row["owner_employee_id"]
        assert row["estimated_budget"] > 0
        assert row["status"] == status(opening.world.promotions[row["promotion_id"]], WORLD.history.until)
    tv = [row for row in later if campaigns[row["campaign_id"]]["has_tv"]]
    assert len(tv) == 8 * len(years)
    ids = {row["promotion_id"] for row in rows}
    scopes = opening.tables["promotion_eligibility"]
    assert {scope["promotion_id"] for scope in scopes} == ids
    exclusions = {(r["promotion_id"], r["excluded_promotion_id"]) for r in opening.tables["promotion_exclusion"]}
    assert exclusions and all((b, a) in exclusions for a, b in exclusions)
    assert (_id("PR-2025-03-GOLD-TV"), _id("PR-2025-03-CLASSIC-TV")) in exclusions


def test_a_status_follows_the_day() -> None:
    promotion = Promotion(1, "PR-X", "GOLD", "price_off", date(2025, 3, 3), date(2025, 3, 30), False, (), 5)
    assert [status(promotion, date(2025, 3, d)) for d in (2, 3, 30, 31)] == ["approved", "running", "running", "closed"]
