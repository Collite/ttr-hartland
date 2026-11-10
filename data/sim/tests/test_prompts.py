"""The rep prompts (G2 S2.2, `loop.prompts`): R-7's unsupplied items on a fixture store against its golden list (T1),
each of T2's steps alone, the mandatory promotion (R-6, T3), the display placement and the phase's cap (T4), the
classification after the visit (R-11, T5), A-13's store and the rejection memory (T6), and the purchase history the
orders book (`prompts.bought`)."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hartland_sim.checks.czech import assert_czech
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.io.reference import read
from hartland_sim.loop import prompts
from hartland_sim.loop.day import Loop
from hartland_sim.loop.demand import REFUSES, BasketReady
from hartland_sim.loop.field import PRAGUE, Listing, Planned, Visit, plan
from hartland_sim.loop.pricing import Discount, Priced, Rules
from hartland_sim.loop.prompts import (
    Briefing,
    Prompt,
    Purchases,
    Rulebook,
    classify,
    drop_rejected,
    drop_seasonally_excluded,
    generate,
    group_by_product,
    listed_not_bought,
    month_index,
    order_by_importance,
    tie_break_by_avg_revenue_per_buying_point,
    unsupplied,
)
from hartland_sim.loop.warmup import Output
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, Promotion, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
GOLDEN = Path(__file__).parent / "golden" / "r7-2025-06-15.json"
DAY = date(2025, 6, 15)
# The fixture store's year: seven SKUs it bought (catalogue codes), the last time each, days before DAY.
BOUGHT = {
    "2931000000029": 12,  # Classic 100 g
    "2931000000036": 40,  # Classic 200 g
    "2931000000173": 90,  # Gold 100 g
    "2931000000180": 200,  # Gold 200 g
    "2931000000357": 300,  # 3v1, 10 tyčinek
    "2931000000418": 364,  # Cappuccino, 10 tyčinek
    "2931000000838": 150,  # Kuřecí v kelímku
}
STALE = "2931000000043"  # Classic refill 150 g: bought 366 days ago, so unsupplied again
REJECTED = {"GOLD-CREMA": 30, "LATTE": 75}  # two products rejected in the last three months, days before


@pytest.fixture(scope="module")
def book(opening: Setup) -> Rulebook:
    return Rulebook.of(opening.world, WORLD, STORIES)


def _column(book: Rulebook, code: str) -> int:
    return book.sku_codes.index(code)


def fixture_store(opening: Setup, book: Rulebook) -> tuple[World, int, int, Briefing]:
    """T1's fixture: a small-retail customer, phase regular, with a year of orders over seven SKUs, two products
    rejected in the last three months, on 2025-06-15 (cups are excluded in June); each SKU's average revenue per
    buying point is its base case price (`catalogue.csv`). Returns its `World` (only `purchases`), the customer, its
    listing group and the day's briefing."""
    demand = opening.world.demand
    assert demand is not None
    customer = next(c for c in opening.world.customers.values() if c.channel == "small_retail" and not c.traits)
    last = np.zeros(len(book.sku_codes), dtype=np.int64)
    for code, days in {**BOUGHT, STALE: 366}.items():
        last[_column(book, code)] = (DAY - timedelta(days=days)).toordinal()
    prices = {row["sku_code"]: float(row["base_price_case"]) for row in read("catalogue")}
    purchases = Purchases.of(opening.world)
    purchases.rows = np.array([0])
    purchases.last_bought = last[None, :]
    purchases.average = np.array([prices[code] for code in book.sku_codes])
    purchases.month = month_index(DAY)
    purchases.rejected = {
        customer.customer_id: {code: (DAY - timedelta(days=days)).toordinal() for code, days in REJECTED.items()}
    }
    purchases.rejections = {customer.customer_id: {"GOLD": 1, "SMESI": 1}}
    world = World(DAY - timedelta(days=1), purchases=purchases)
    group = demand.catalogue.groups.index(("small_retail", None))
    return world, customer.customer_id, group, Briefing(DAY, demand.catalogue.live(DAY), (), purchases.average.tolist())


def _offers(book: Rulebook, offers: list[tuple[int, tuple[int, ...]]]) -> list[dict[str, Any]]:
    return [{"product": book.products[k].code, "skus": [book.sku_codes[s] for s in skus]} for k, skus in offers]


# ── T1: the fixture store's golden list ────────────────────────────────────────


def test_the_fixture_store_s_r7_list_is_the_golden(opening: Setup, book: Rulebook) -> None:
    world, customer_id, group, briefing = fixture_store(opening, book)
    drafts, left = unsupplied(world, book, briefing, 0, group, customer_id)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert golden["day"] == DAY.isoformat()
    assert _offers(book, left) == golden["offers"]
    assert [d.payload for d in drafts] == golden["prompts"]
    assert [d.explanation for d in drafts] == golden["explanations"]


def test_the_golden_list_holds_r7_s_rules(opening: Setup, book: Rulebook) -> None:
    """What the golden says, read off the rules rather than the code: no SKU bought in the year nor a cup in June, no
    product rejected in the last three months, products by importance then by their best SKU's average."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    catalogue = {row["sku_code"]: row for row in read("catalogue")}
    product_line = {row["code"]: row["product_line"] for row in read("product")}
    offered = [sku for offer in golden["offers"] for sku in offer["skus"]]
    assert not set(offered) & set(BOUGHT)
    assert STALE in offered
    assert not {offer["product"] for offer in golden["offers"]} & set(REJECTED)
    assert all(product_line[offer["product"]] != "KELIMKY" for offer in golden["offers"])
    assert all(not catalogue[sku]["is_professional"] for sku in offered)
    best = [
        (catalogue[o["skus"][0]]["importance"], float(catalogue[o["skus"][0]]["base_price_case"]))
        for o in golden["offers"]
    ]
    assert best == sorted(best, key=lambda held: (-held[0], -held[1]))
    assert len(golden["prompts"]) == 3 and all(len(p["skus"]) <= 3 for p in golden["prompts"])
    for said in golden["explanations"]:
        assert_czech(said)


# ── T2: the steps ──────────────────────────────────────────────────────────────


def test_listed_not_bought_keeps_what_is_listed_and_not_bought_in_the_lookback() -> None:
    listed = np.array([True, True, True, False])
    day = date(2025, 6, 15)
    last = np.array([0, (day - timedelta(days=365)).toordinal(), (day - timedelta(days=366)).toordinal(), 0])
    assert listed_not_bought(listed, last, day, 365).tolist() == [True, False, True, False]


def test_group_by_product_then_importance_then_the_average(book: Rulebook) -> None:
    candidates = np.zeros(len(book.sku_codes), dtype=bool)
    codes = ["2931000000012", "2931000000029", "2931000000043", "2931000000081", "2931000000098"]  # Classic, Silná
    for code in codes:
        candidates[_column(book, code)] = True
    grouped = group_by_product(candidates, book.sku_product)
    assert _offers(book, grouped) == [
        {"product": "CLASSIC", "skus": codes[:3]},
        {"product": "CLASSIC-SILNA", "skus": codes[3:]},
    ]
    ordered = order_by_importance(grouped, book.importance)
    assert _offers(book, ordered)[0]["skus"] == ["2931000000029", "2931000000043", "2931000000012"]  # 10, 5, 1
    average = [0.0] * len(book.sku_codes)
    average[_column(book, "2931000000098")] = 50.0  # Silná 200 g: importance 5 like its 100 g, a larger average
    broken = tie_break_by_avg_revenue_per_buying_point(ordered, book.importance, average)
    assert _offers(book, broken)[1]["skus"] == ["2931000000098", "2931000000081"]
    assert [o["product"] for o in _offers(book, broken)] == ["CLASSIC", "CLASSIC-SILNA"]  # 10 before 5 whatever
    average[_column(book, "2931000000081")] = 60.0
    again = tie_break_by_avg_revenue_per_buying_point(ordered, book.importance, average)
    assert _offers(book, again)[1]["skus"] == ["2931000000081", "2931000000098"]


def test_drop_rejected_forgets_after_the_memory_and_stops_a_line_after_three(book: Rulebook) -> None:
    products = {p.code: k for k, p in enumerate(book.products)}
    offers = [(products["NUDLE-KURECI"], (0,)), (products["NUDLE-KARI"], (1,)), (products["CLASSIC"], (2,))]
    day = date(2025, 6, 15)
    at = (day - timedelta(days=89)).toordinal()
    kept = drop_rejected(offers, book, {"NUDLE-KURECI": at}, {"NUDLE": 1}, day)
    assert [book.products[k].code for k, _ in kept] == ["NUDLE-KARI", "CLASSIC"]
    at = (day - timedelta(days=90)).toordinal()
    assert len(drop_rejected(offers, book, {"NUDLE-KURECI": at}, {"NUDLE": 2}, day)) == 3
    kept = drop_rejected(offers, book, {}, {"NUDLE": 3}, day)
    assert [book.products[k].code for k, _ in kept] == ["CLASSIC"]


def test_drop_seasonally_excluded_by_the_line_s_months(book: Rulebook) -> None:
    products = {p.code: k for k, p in enumerate(book.products)}
    offers = [(products["KELIMEK-KURECI"], (0,)), (products["NUDLE-KURECI"], (1,)), (products["GOLD"], (2,))]
    summer = [book.products[k].code for k, _ in drop_seasonally_excluded(offers, book, date(2025, 7, 1))]
    assert summer == ["GOLD"]  # cups 5–9, noodles 6–8
    assert len(drop_seasonally_excluded(offers, book, date(2025, 10, 1))) == 3


def test_the_ai_tip_is_an_important_5_sku_its_cluster_buys_outside_the_offered_lines(
    opening: Setup, book: Rulebook
) -> None:
    world, customer_id, group, briefing = fixture_store(opening, book)
    demand = opening.world.demand
    assert demand is not None and world.purchases is not None
    street = "residential"
    world.purchases.clusters = (street,)
    world.purchases.peers = np.zeros((1, len(book.sku_codes)))
    drafts, left = unsupplied(world, book, briefing, 0, group, customer_id)
    assert prompts.tip(world, book, street, left, drafts) == []  # no peer buys anything
    lines = {d.line for d in drafts}
    candidates = [s for k, skus in left for s in skus if book.importance[s] == 5 and book.products[k].line not in lines]
    world.purchases.peers[0, candidates] = 0.5
    world.purchases.peers[0, candidates[0]] = 0.2  # below the share: the next one
    (tip,) = prompts.tip(world, book, street, left, drafts)
    assert tip.skus == (candidates[1],) and tip.payload["ai_tip"] is True and tip.payload["peer_share"] == 0.5
    assert_czech(tip.explanation)


# ── the history ────────────────────────────────────────────────────────────────


def _priced(sku_id: int, cases: float, kind: str | None = None, free: bool = False) -> Priced:
    """A line at 90 a case net, with one discount of `kind` when given."""
    discounts = (Discount(1, kind, "X", None, None, 10.0, 10.0, 10.0, "marketing"),) if kind else ()
    return Priced(sku_id, cases, cases * 10, 100.0, 10.0, 10.0 * cases, 90.0, free, discounts)


def test_an_order_books_its_skus_and_the_mandatory_ones_cases(opening: Setup) -> None:
    world = Checkpoint.of(opening.world).restore()
    rules = Rules.of(world, WORLD, STORIES)
    assert world.purchases is not None and world.demand is not None
    customer = next(c for c in world.customers.values() if c.channel == "small_retail" and c.status == "active")
    condition = rules.sku_ids[_zima(world)[1].condition_sku or ""]
    other = next(sku for sku in rules.sku_ids.values() if sku != condition and rules.sku_line[sku] == "GOLD")
    day = date(2023, 1, 10)
    watch = prompts.Watch.of(world, rules)
    assert condition in watch.measured
    lines = [_priced(condition, 2.0), _priced(other, 1.0, "promotion"), _priced(other, 1.0, free=True)]
    prompts.bought(world, rules, watch, customer.customer_id, day, lines)
    r = int(world.purchases.rows[rules.customers[customer.customer_id]])
    column = rules.columns
    assert world.purchases.last_bought[r, column[condition]] == day.toordinal()
    assert world.purchases.last_bought[r, column[other]] == day.toordinal()
    assert world.purchases.recent[customer.customer_id] == [(day.toordinal(), condition, 2.0)]
    slot = month_index(day) % 12
    assert world.purchases.revenue[slot, column[condition]] == 180.0  # 2 cases × 90
    assert world.purchases.revenue[slot, column[other]] == 0.0  # a promotional sale is left out
    later = day + timedelta(days=28)
    prompts.bought(world, rules, watch, customer.customer_id, later, [_priced(other, 1.0)])
    assert customer.customer_id not in world.purchases.recent  # 28 days on: the condition cases are out of the period
    horeca = next(c for c in world.customers.values() if c.channel == "horeca")
    prompts.bought(world, rules, watch, horeca.customer_id, day, lines)
    assert horeca.customer_id not in world.purchases.recent


def test_the_month_s_figures_are_the_twelve_months_before_its_first_day_s_orders(
    opening: Setup, tmp_path: Path
) -> None:
    """R-7 step 4's average (review-233 R2): the field step takes the month's figures on its 1st — a Saturday here,
    with no call planned — before the orders book that day into the ring's slot of M − 12. They are the twelve months
    before, M − 12 included, over the customers who bought in the lookback before the 1st, not one whose only purchase
    is on the 1st; the plan's first working day reads them as they are."""
    world = Checkpoint.of(opening.world).restore()
    rules = Rules.of(world, WORLD, STORIES)
    purchases = world.purchases
    assert purchases is not None
    first = date(2023, 4, 1)
    assert first.weekday() == 5
    month = month_index(first)
    sku = rules.sku_ids["2931000000029"]  # Classic 100 g
    s = rules.columns[sku]
    for k in range(1, 13):
        purchases.months[(month - k) % 12] = month - k
        purchases.revenue[(month - k) % 12, s] = 100.0 * k  # M − 12 the most
    customers = list(world.customers.values())
    tiered = np.flatnonzero(purchases.rows >= 0)
    for c in tiered[:3]:
        purchases.last_bought[purchases.rows[c], s] = (first - timedelta(days=30)).toordinal()
    with CsvWriter(tmp_path) as writer:
        loop = Loop.of(world, WORLD, STORIES, opening.portfolio.skus, Output(writer, date(2023, 1, 1)))
        loop.weather(first)
        loop.field(first)
    late = customers[int(tiered[3])]
    prompts.bought(world, rules, loop.desk.watch, late.customer_id, first, [_priced(sku, 1.0)])  # step 4's order
    expected = sum(100.0 * k for k in range(1, 13)) / 3 / 12
    assert purchases.month == month and purchases.average[s] == pytest.approx(expected)
    prompts.refresh(world, loop.rulebook, date(2023, 4, 3))
    assert purchases.average[s] == pytest.approx(expected)


# ── T3: the mandatory promotion ────────────────────────────────────────────────


def _zima(world: World) -> tuple[int, Promotion]:
    return next(
        (pid, p)
        for pid, p in world.promotions.items()
        if p.mandatory and p.line == "POLEVKY" and p.valid_from.year == 2023
    )


def _sales_planned(world: World, day: date, phase: str | None = None) -> list[Planned]:
    """The day's planned visits to small-retail stores of no planted case: to its customers, or to those in `phase`
    (`potential` a store not yet one)."""
    out = []
    for p in plan(world, day):
        customer = world.customers[p.tourplan.customer_id]
        if customer.channel != "small_retail" or customer.traits:
            continue
        if customer.phase_on(day) == phase if phase else customer.active_on(day):
            out.append(p)
    return out


def test_a_mandatory_promotion_prompts_a_customer_below_its_threshold_and_flips_off(
    opening: Setup, book: Rulebook
) -> None:
    world = Checkpoint.of(opening.world).restore()
    rules = Rules.of(world, WORLD, STORIES)
    pid, zima = _zima(world)
    day = zima.valid_from + timedelta(days=7)
    while day.weekday() > 4:
        day += timedelta(days=1)
    planned = _sales_planned(world, day)[:1]
    customer_id = planned[0].tourplan.customer_id

    def mandatory_prompts() -> list[Prompt]:
        return [p for p in generate(world, book, rules, day, planned) if p.kind == "mandatory_promotion"]

    (found,) = mandatory_prompts()
    payload = json.loads(found.payload)
    assert found.priority == 1 and found.shown and payload["promotion"] == zima.code
    assert payload["threshold_cases"] == zima.condition_qty and payload["bought_cases"] == 0
    assert_czech(found.explanation)
    assert world.purchases is not None
    condition = rules.sku_ids[zima.condition_sku or ""]
    world.purchases.recent[customer_id] = [(day.toordinal() - 3, condition, float(zima.condition_qty or 1))]
    assert mandatory_prompts() == []  # bought enough in the period
    world.purchases.recent[customer_id] = [(day.toordinal() - 28, condition, float(zima.condition_qty or 1))]
    assert len(mandatory_prompts()) == 1  # bought before the period
    world.purchases.recent.pop(customer_id)
    world.promotions[pid] = replace(zima, mandatory=False)
    assert mandatory_prompts() == []
    world.promotions[pid] = replace(zima, eligibility=(("channel", ("horeca",)),))
    assert mandatory_prompts() == []
    world.promotions[pid] = replace(zima, valid_to=day - timedelta(days=1))
    assert mandatory_prompts() == []
    world.promotions[pid] = zima


def test_the_cap_cuts_a_visit_s_set_by_priority_and_never_a_mandatory_prompt(opening: Setup, book: Rulebook) -> None:
    world = Checkpoint.of(opening.world).restore()
    rules = Rules.of(world, WORLD, STORIES)
    pid, zima = _zima(world)
    day = zima.valid_from + timedelta(days=7)
    while day.weekday() > 4:
        day += timedelta(days=1)
    world.promotions[-1] = replace(zima, promotion_id=-1, code=zima.code + "-B")  # two mandatory promotions run
    planned = _sales_planned(world, day)
    tight = replace(book, caps={phase: 1 for phase in book.caps})
    by_visit: dict[int, list[Prompt]] = {}
    for p in generate(world, tight, rules, day, planned):
        by_visit.setdefault(p.visit_plan_id, []).append(p)
    assert by_visit
    for found in by_visit.values():
        priorities = [p.priority for p in found]
        assert priorities == sorted(priorities)
        shown = [p for p in found if p.shown]
        mandatory = [p for p in found if p.kind == "mandatory_promotion"]
        assert len(mandatory) == 2 and all(p.shown for p in mandatory)
        assert len(shown) == 2  # both mandatory, over a cap of one
    del world.promotions[-1]
    sets: dict[int, list[Prompt]] = {}
    for p in generate(world, book, rules, day, plan(world, day)):
        sets.setdefault(p.visit_plan_id, []).append(p)
    phases = set()
    for held in sets.values():
        phase = world.customers[held[0].customer_id].phase_on(day)
        phases.add(phase)
        flags = [p.shown for p in held]
        assert flags == sorted(flags, reverse=True) and sum(flags) == min(book.caps[phase], len(held))
    assert len(phases) >= 2, phases


# ── T4: the display placement ──────────────────────────────────────────────────


def test_a_display_prompt_needs_the_uplift_and_the_region_s_budget(opening: Setup, book: Rulebook) -> None:
    world = Checkpoint.of(opening.world).restore()
    rules = Rules.of(world, WORLD, STORIES)
    assert world.demand is not None
    day = date(2023, 3, 6)
    planned = _sales_planned(world, day)[:1]
    t = planned[0].tourplan
    region = world.demand.spots[book.points[t.delivery_point_id]].region

    def displays() -> list[Prompt]:
        return [p for p in generate(world, book, rules, day, planned) if p.kind == "pos_placement"]

    assert displays() == []  # S2.3's: nothing yet
    world.display_uplift[t.delivery_point_id] = {"GOLD": 0.22, "CLASSIC": 0.10}  # Classic under the bar
    assert displays() == []  # no budget
    world.display_budget[region] = 5000.0
    (found,) = displays()
    assert json.loads(found.payload) == {"display_uplift": 0.22, "line": "GOLD"} and found.priority == 3
    assert_czech(found.explanation)
    world.display_uplift[t.delivery_point_id]["GOLD"] = 0.14
    assert displays() == []


# ── T5: the classification ─────────────────────────────────────────────────────


def _visit(p: Prompt, world: World, minutes: int = 30, recorded: bool = True, outcome: str = "no_order") -> Visit:
    tourplan = next(t for t in world.tourplans.values() if t.customer_id == p.customer_id)
    start = datetime.combine(date(2023, 3, 6), time(9), PRAGUE)
    times = (start, start + timedelta(minutes=minutes)) if recorded else (None, None)
    return Visit(p.visit_plan_id + 1, p.visit_plan_id, tourplan, "sales", outcome, False, recorded, *times)


def _one_set(opening: Setup, book: Rulebook) -> tuple[World, list[Prompt]]:
    world = Checkpoint.of(opening.world).restore()
    rules = Rules.of(world, WORLD, STORIES)
    day = date(2023, 3, 6)
    found = generate(world, book, rules, day, _sales_planned(world, day))
    first = found[0].visit_plan_id
    return world, [p for p in found if p.visit_plan_id == first]


def test_a_prompt_whose_sku_is_on_the_visit_s_order_is_successful(opening: Setup, book: Rulebook) -> None:
    world, found = _one_set(opening, book)
    v = _visit(found[0], world)
    listing = next(p for p in found if p.kind == "unsupplied_items" and p.shown)
    order = BasketReady(
        date(2023, 3, 6),
        listing.customer_id,
        v.tourplan.delivery_point_id,
        "small_retail",
        ((listing.skus[-1], 1.0, 10.0),),
        (v.visit_id, v.tourplan.employee_id),
    )
    said = classify(world, book, date(2023, 3, 6), found, [v], [order])
    assert listing.outcome == "successful" and listing.classified_on == date(2023, 3, 6) and listing.reason is None
    assert all(p.visit_id == v.visit_id for p in found)
    name = next(product.name for product in book.products if product.code == listing.product)
    first = next(p for p in found if p.kind == "unsupplied_items" and p.outcome != "open")
    if first is listing:
        assert said[v.visit_id] == Listing(name, True)


def test_an_item_ordered_at_a_short_visit_is_successful_and_the_set_s_others_not_discussed(
    opening: Setup, book: Rulebook
) -> None:
    """review-233 R1: the system classifies an ordered item from the order, whatever the visit's length (R-11)."""
    world, found = _one_set(opening, book)
    odds = WORLD.policies.prompt_outcomes
    day = date(2023, 3, 6)
    listing = next(p for p in found if p.kind == "unsupplied_items" and p.shown)
    assert listing.priority > odds.short_visit_priority
    v = _visit(found[0], world, minutes=odds.short_visit_minutes - 1)
    order = BasketReady(
        day,
        listing.customer_id,
        v.tourplan.delivery_point_id,
        "small_retail",
        ((listing.skus[0], 1.0, 10.0),),
        (v.visit_id, v.tourplan.employee_id),
    )
    classify(world, book, day, found, [v], [order])
    assert listing.outcome == "successful" and listing.classified_on == day
    others = [p for p in found if p is not listing and p.shown and p.priority > odds.short_visit_priority]
    assert others and all(p.outcome == "not_discussed" for p in others)


def test_the_bands_of_a_discussed_prompt_without_its_item(
    opening: Setup, book: Rulebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    odds = WORLD.policies.prompt_outcomes
    for draw, expected in ((0.0, "rejected"), (odds.rejected + 0.01, "unsuccessful"), (0.99, "open")):
        world, found = _one_set(opening, book)

        def forced(stream: str, *key: object, ids: np.ndarray, value: float = draw) -> np.ndarray:
            return np.full(len(ids), value)

        monkeypatch.setattr("hartland_sim.loop.prompts.uniforms", forced)
        v = _visit(found[0], world)
        classify(world, book, date(2023, 3, 6), found, [v], [])
        for p in found:
            if not p.shown:
                assert p.outcome == "open" and p.visit_id == v.visit_id
            elif p.kind == "mandatory_promotion" and expected == "rejected":
                assert p.outcome == "unsuccessful"  # a mandatory prompt cannot be rejected
            else:
                assert p.outcome == expected
                reasons = {
                    "rejected": WORLD.policies.prompt_reasons.rejected,
                    "unsuccessful": WORLD.policies.prompt_reasons.unsuccessful,
                }.get(expected)
                assert (p.reason in reasons) if reasons else p.reason is None
                assert (p.classified_on is None) == (expected == "open")


def test_a_short_visit_discusses_only_the_urgent_and_others_leave_all_open(opening: Setup, book: Rulebook) -> None:
    world, found = _one_set(opening, book)
    short = WORLD.policies.prompt_outcomes.short_visit_minutes
    classify(world, book, date(2023, 3, 6), found, [_visit(found[0], world, minutes=short - 1)], [])
    for p in found:
        if p.shown and p.priority > WORLD.policies.prompt_outcomes.short_visit_priority:
            assert p.outcome == "not_discussed"
    world, found = _one_set(opening, book)
    classify(world, book, date(2023, 3, 6), found, [_visit(found[0], world, recorded=False)], [])
    assert all(p.outcome == "open" and p.visit_id is not None for p in found)  # A-8's unrecorded
    world, found = _one_set(opening, book)
    classify(world, book, date(2023, 3, 6), found, [], [])
    assert all(p.outcome == "open" and p.visit_id is None for p in found)  # the visit not made
    world, found = _one_set(opening, book)
    classify(world, book, date(2023, 3, 6), found, [_visit(found[0], world, outcome="closed")], [])
    assert all(p.outcome == ("not_discussed" if p.shown else "open") for p in found)


# ── T6: A-13 and the memory ────────────────────────────────────────────────────


def test_a13_s_store_rejects_every_noodle_listing_prompt_and_the_rule_stops_after_three(
    opening: Setup, book: Rulebook
) -> None:
    world = Checkpoint.of(opening.world).restore()
    rules = Rules.of(world, WORLD, STORIES)
    (store,) = [c for c in world.customers.values() if REFUSES in c.traits]
    reason = str(STORIES.anomaly("A-13").parameters["reason_cs"])
    tourplan_id = world.tourplan_of[str(store.store)]
    assert world.purchases is not None and world.demand is not None
    row = int(world.purchases.rows[book.customers[store.customer_id]])
    carried = world.demand.carried[book.customers[store.customer_id]]
    buys = np.flatnonzero(carried[np.asarray(world.demand.catalogue.line)])  # the SKUs of the lines it buys
    day, rejections, noodles_after = date(2023, 1, 2), 0, 0
    while day < date(2024, 6, 1):
        world.purchases.last_bought[row, buys] = day.toordinal() - 1  # it buys everything but noodles
        if day.weekday() < 5 and day.month not in WORLD.policies.seasonal_exclusion["NUDLE"]:
            planned = [Planned(tourplan_id * 31 + day.toordinal(), tourplan_id, world.tourplans[tourplan_id])]
            found = generate(world, book, rules, day, planned)
            noodles = [p for p in found if p.kind == "unsupplied_items" and p.line == "NUDLE"]
            if rejections >= 3:
                noodles_after += len(noodles)
            v = _visit(found[0], world) if found else None
            if v is not None:
                v = replace(v, visit_id=found[0].rep_prompt_id)
                before = rejections
                said = classify(world, book, day, found, [v], [])
                for p in noodles:
                    if p.shown:
                        assert p.outcome == "rejected" and p.reason == reason
                        rejections += 1
                if rejections > before:  # its note: the first refusal plainly, the later ones "again" (review-233 R6)
                    assert said[v.visit_id].planted and said[v.visit_id].again == (before > 0)
        day += timedelta(days=14)
    assert rejections >= 3 and noodles_after == 0
    assert world.purchases is not None and world.purchases.rejections[store.customer_id]["NUDLE"] == rejections


# ── T7: the loop ───────────────────────────────────────────────────────────────


def _rows(run: Path, table: str) -> list[dict[str, str]]:
    path = run / f"{table}.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def test_the_loop_s_prompts_answer_to_their_visits_and_orders(opening: Setup, tmp_path: Path) -> None:
    """Two written weeks of January 2023: each prompt's visit is its customer's and its rep's of its day, never a
    station's merchandising; a successful one has an offered SKU on that visit's order, and one the rep declined,
    rejected or did not discuss has none (review-233 R1); the notes tell the listing prompts as they came out."""
    world = Checkpoint.of(opening.world).restore()
    with CsvWriter(tmp_path) as writer:
        loop = Loop.of(world, WORLD, STORIES, opening.portfolio.skus, Output(writer, date(2023, 1, 1)))
        loop.until(date(2023, 1, 14))
    prompted = _rows(tmp_path, "rep_prompt")
    visits = {v["visit_id"]: v for v in _rows(tmp_path, "visit")}
    orders = {o["sales_order_id"]: o["visit_id"] for o in _rows(tmp_path, "sales_order") if o["visit_id"]}
    ordered: dict[str, set[str]] = {}
    for line in _rows(tmp_path, "sales_order_line"):
        if line["sales_order_id"] in orders:
            ordered.setdefault(orders[line["sales_order_id"]], set()).add(line["sku_id"])
    sku_ids = {code: str(sku) for code, sku in Rules.of(world, WORLD, STORIES).sku_ids.items()}
    assert len(prompted) > 5_000
    outcomes = {r["outcome"] for r in prompted}
    assert outcomes == {"open", "not_discussed", "unsuccessful", "successful", "rejected"}, outcomes
    for r in prompted:
        if r["visit_id"]:
            v = visits[r["visit_id"]]
            assert (v["customer_id"], v["employee_id"], v["day"]) == (
                r["customer_id"],
                r["employee_id"],
                r["generated_on"],
            )
            assert v["kind"] != "merchandising"
        offered = {sku_ids[code] for code in json.loads(r["payload"]).get("skus", [])}
        hit = bool(offered & ordered.get(r["visit_id"], set()))
        if r["outcome"] == "successful":
            assert hit
        elif r["outcome"] in ("unsuccessful", "rejected", "not_discussed"):
            assert not hit
    noted = [v["note"] for v in visits.values() if v["note"]]
    assert any("zalistováno" in n or "je v dnešní objednávce" in n for n in noted)
    assert any("jsem nabízel, majitel odmítl" in n or "zatím nechtějí, důvod" in n for n in noted)
