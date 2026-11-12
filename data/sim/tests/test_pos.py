"""The point of sale (G2 S2.3 T1–T5; `loop.pos`, `loop.field.handouts`): a display's uplift at a store (`04` S5's
golden table, the null world's flat), the month-end's placements by budget and skill, the skilled managers'
withdrawals, the returns, the asset ledger and its `asset_register` rows, and the give-aways at the visits. The world
is the opening's, restored fresh for each test."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import add_months
from hartland_sim.ids import row_id
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import field, pos
from hartland_sim.loop.demand import BasketReady, Store, lines
from hartland_sim.loop.field import Crew
from hartland_sim.loop.pos import Grid, Terms
from hartland_sim.loop.warmup import Output
from hartland_sim.rng import uniforms
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
S5 = STORIES.story("S5").parameters
END, FIRST = date(2023, 1, 31), date(2023, 2, 1)  # the warm-up's first month-end, its placements' day
NEVER = date(2100, 1, 1)  # an `Output` that writes nothing
DAYS = [date(2023, 3, d) for d in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]


@pytest.fixture(scope="module")
def terms(opening: Setup) -> Terms:
    assert opening.world.demand is not None
    return Terms.of(WORLD, STORIES, opening.world.demand.lines)


@pytest.fixture
def world(opening: Setup) -> World:
    return Checkpoint.of(opening.world).restore()


def _close(world: World, terms: Terms, day: date, tmp_path: Path) -> None:
    with CsvWriter(tmp_path) as writer:
        pos.close(world, terms, day, Output(writer, NEVER))


# ── T1: display_uplift(store) ─────────────────────────────────────────────────

GOLDEN = [  # (store class, street type, size class, display_uplift): 04 S5's two, then s5.yaml's grid between
    ("hyper", "shopping_centre", "L", 0.35),  # a hypermarket's coffee aisle
    ("village", "village", "S", 0.08),  # a village shop
    ("super", "high_street", "M", 0.25),
    ("super", "residential", "M", 0.175),
    ("super", "village", "M", 0.10),
    ("convenience", "high_street", "S", 0.20),
    ("convenience", "residential", "S", 0.14),
    ("minimarket", "transport_hub", "S", 0.18),
    ("petrol", "petrol", "S", 0.12),
]


@pytest.mark.parametrize(("store_class", "street", "size", "uplift"), GOLDEN)
def test_display_uplift_is_04_s_golden_table_and_the_null_world_s_flat(
    store_class: str, street: str, size: str, uplift: float
) -> None:
    store = Store(size, street, 72.0, False, store_class)
    assert Grid.of(STORIES).uplift(store) == pytest.approx(uplift)
    assert Grid.of(STORIES.null()).uplift(store) == S5["null_uplift"] == 0.15


def test_the_world_holds_each_small_retail_store_s_uplift_under_its_best_line(opening: Setup, terms: Terms) -> None:
    """S2.2's display prompt reads `World.display_uplift` (its T4 stub): every small-retail store's point, under the
    line a display there would carry; no other channel's."""
    world = opening.world
    demand = world.demand
    assert demand is not None
    small = [p for p, spot in enumerate(demand.spots) if spot.channel == "small_retail" and spot.store is not None]
    assert sorted(world.display_uplift) == sorted(int(demand.point_ids[p]) for p in small)
    for p in small[:500]:
        spot = demand.spots[p]
        assert spot.store is not None
        line = demand.lines[pos.best_line(demand, p, terms.brands)].code
        assert world.display_uplift[spot.delivery_point_id] == {line: terms.grid.uplift(spot.store)}


# ── T2: the placements, the withdrawals, the returns ─────────────────────────


def test_a_month_end_lends_each_region_its_budget_of_displays(world: World, terms: Terms, tmp_path: Path) -> None:
    _close(world, terms, END, tmp_path)
    loans = list(world.loans.values())
    assert loans and len(world.lent) == len(loans) == len({loan.delivery_point_id for loan in loans})
    displays = max(terms.materials[code].unit_cost for code in terms.displays)
    for region in terms.regions:
        budget = WORLD.field.pos_budget[region] * S5["budget_factor"].get(region, 1.0)
        spent = sum(world.assets[loan.pos_asset_id].cost for loan in loans if loan.region == region)
        assert budget - displays < spent <= budget and world.display_budget[region] == pytest.approx(budget - spent)
    tourplans = {t.delivery_point_id: t for t in world.tourplans.values() if t.valid_to >= FIRST}
    low, high = WORLD.field.pos_loans.months
    for loan_id, loan in world.loans.items():
        asset = world.assets[loan.pos_asset_id]
        assert (asset.loan, asset.region, asset.written_off_on) == (loan_id, loan.region, None)
        assert loan.lines == terms.displays[asset.material] and loan.lent_on == FIRST
        assert add_months(FIRST, low) - timedelta(days=1) <= loan.until <= add_months(FIRST, high) - timedelta(days=1)
        assert loan.placed_by == tourplans[loan.delivery_point_id].employee_id
        manager = world.duty("area_manager_region", loan.region, FIRST)
        assert manager is not None and loan.approved_by == manager.employee_id
        assert world.customers[loan.customer_id].channel == "small_retail"
    rows = [pos.loan_row(loan_id, loan) for loan_id, loan in world.loans.items()]
    assert all(r["returned_on"] is None and r["condition_on_return"] is None for r in rows)
    assert {r["pos_loan_id"] for r in rows} == {
        row_id("pos_loan", pos_asset=world.assets[loan.pos_asset_id].asset_no, lent_on=FIRST) for loan in loans
    }


def test_a_manager_takes_the_stores_by_uplift_to_the_power_of_skill_times_three(
    world: World, terms: Terms, tmp_path: Path
) -> None:
    """The order a region's manager takes its candidates in: Efraimidis–Spirakis keys log(U) ÷ uplift ^ (pos_skill × 3),
    U from stream `pos` by the region, the month and the point, the largest first — so the lent stores are the first
    of that order whose display the region holds, up to the budget."""
    demand = world.demand
    assert demand is not None
    points = {int(point): p for p, point in enumerate(demand.point_ids)}
    candidates: dict[str, list[tuple[int, float]]] = {}
    for tourplan_id in world.tourplan_of.values():
        t = world.tourplans[tourplan_id]
        customer = world.customers[t.customer_id]
        store = demand.spots[points[t.delivery_point_id]].store
        if customer.channel == "small_retail" and customer.active_on(FIRST) and store is not None:
            candidates.setdefault(terms.region_of[t.area], []).append((t.delivery_point_id, terms.grid.uplift(store)))
    _close(world, terms, END, tmp_path)
    for region in terms.regions:
        found = candidates[region]
        ids = np.array([point for point, _ in found], dtype=np.int64)
        weight = np.array([u for _, u in found]) ** (S5["pos_skill"][region] * S5["choice_exponent"])
        key = np.log(uniforms("pos", "place", region, FIRST, ids=ids)) / weight
        order = [found[k][0] for k in np.argsort(-key, kind="stable")]
        lent = {loan.delivery_point_id for loan in world.loans.values() if loan.region == region}
        assert set(order[: len(lent)]) == lent, region


def _measured(world: World, loan_id: int, *after: float) -> None:
    """The loan's lines' cases at its point: 10 a month in the two months before it, then `after`'s, a month each."""
    measure, demand, loan = world.measure, world.demand, world.loans[loan_id]
    assert measure is not None and demand is not None
    line = next(i for i, held in enumerate(demand.lines) if held.code in loan.lines)
    for month, cases in enumerate((10.0, 10.0, *after), start=-2):
        measure.book(loan.delivery_point_id, add_months(loan.lent_on, month), line, cases)


@pytest.mark.parametrize(("after", "withdrawn"), [(10.2, True), (11.0, False)])
def test_a_skilled_manager_withdraws_a_display_measured_below_5_percent_after_three_months(
    world: World, terms: Terms, tmp_path: Path, after: float, withdrawn: bool
) -> None:
    """PHA's manager (skilled) takes back a display whose lines sold 2 % more a month in its first three months than in
    the two before, not one that sold 10 % more; VCH's (unskilled) never."""
    _close(world, terms, END, tmp_path)
    for region in ("PHA", "VCH"):
        loan_id = next(i for i, loan in world.loans.items() if loan.region == region)
        world.loans[loan_id].until = date(2024, 1, 31)
        _measured(world, loan_id, after, after, after)
        for day in (date(2023, 2, 28), date(2023, 3, 31)):
            pos.returns(world, terms, day, day + timedelta(days=1))
        assert world.loans[loan_id].returned_on is None
        back = pos.returns(world, terms, date(2023, 4, 30), date(2023, 5, 1))
        assert (loan_id in back) == (withdrawn and region == "PHA")
        if loan_id in back:
            loan = world.loans[loan_id]
            assert loan.returned_on == date(2023, 5, 1) and loan.condition_on_return in pos.CONDITIONS
            assert loan.delivery_point_id not in world.lent and world.assets[loan.pos_asset_id].loan is None


@pytest.mark.parametrize(("fourth", "withdrawn"), [(10.2, True), (20.0, False)])
def test_at_four_months_the_measure_reads_the_loan_s_first_four(
    world: World, terms: Terms, tmp_path: Path, fourth: float, withdrawn: bool
) -> None:
    """review-248 R5: a skilled manager judges a loan at its `withdraw_after_months`-th month-end over that many months
    — at 4, its first four: 2 % up in the first three and a fourth that lifts it (20 cases) keeps it."""
    four = replace(terms, withdraw_after_months=4)
    _close(world, terms, END, tmp_path)
    loan_id = next(i for i, loan in world.loans.items() if loan.region == "PHA")
    world.loans[loan_id].until = date(2024, 1, 31)
    _measured(world, loan_id, 10.2, 10.2, 10.2, fourth)
    for day in (date(2023, 2, 28), date(2023, 3, 31), date(2023, 4, 30)):
        assert loan_id not in pos.returns(world, four, day, day + timedelta(days=1))
    measured = pos.uplift_measured(world, world.loans[loan_id], 4)
    assert measured == pytest.approx((30.6 + fourth) / 4 / 10 - 1)
    assert (loan_id in pos.returns(world, four, date(2023, 5, 31), date(2023, 6, 1))) == withdrawn


@pytest.mark.parametrize("months", [0, 5])
def test_the_measure_holds_four_months_after_a_loan_at_most(months: int) -> None:
    """review-248 R5: the measure keeps `MONTHS` (6): a loan's two before and at most four after — `Terms.of` refuses
    a `withdraw_after_months` it cannot hold, at load, rather than withdraw nothing."""
    s5 = replace(STORIES.story("S5"), parameters=MappingProxyType({**S5, "withdraw_after_months": months}))
    stories = replace(STORIES, stories=tuple(s5 if s.story == "S5" else s for s in STORIES.stories))
    with pytest.raises(ValueError, match=rf"^s5\.yaml withdraw_after_months: {months} — the measure keeps 6 months"):
        Terms.of(WORLD, stories, lines())


def test_a_loan_comes_back_at_its_term_or_its_customer_s_loss(world: World, terms: Terms, tmp_path: Path) -> None:
    _close(world, terms, END, tmp_path)
    ended, lost, kept = [i for i, loan in world.loans.items() if loan.region == "SJC"][:3]
    world.loans[ended].until = date(2023, 2, 28)
    world.loans[kept].until = world.loans[lost].until = date(2023, 12, 31)
    customer = world.customers[world.loans[lost].customer_id]
    customer.status, customer.until = "lost", date(2023, 2, 20)
    back = pos.returns(world, terms, date(2023, 2, 28), date(2023, 3, 1))
    assert {ended, lost} <= set(back) and kept not in back


def test_a_return_s_condition_and_a_damaged_one_written_off(world: World, terms: Terms, tmp_path: Path) -> None:
    _close(world, terms, END, tmp_path)
    for loan in world.loans.values():
        loan.until = END + timedelta(days=28)
    back = pos.returns(world, terms, date(2023, 2, 28), date(2023, 3, 1))
    conditions = Counter(world.loans[i].condition_on_return for i in back)
    assert len(back) == len(world.loans) and not world.lent
    draws = Counter(pos.drawn(terms, i) for i in back)  # the condition a return is not worse than (R8)
    assert set(conditions) <= set(pos.CONDITIONS) and draws.most_common(1)[0][0] == "good"
    for i in back:
        asset = world.assets[world.loans[i].pos_asset_id]
        damaged = world.loans[i].condition_on_return == "damaged"
        assert asset.condition == world.loans[i].condition_on_return
        assert (asset.written_off_on == date(2023, 3, 1)) == damaged


def test_a_loan_never_mends_its_asset(world: World, terms: Terms, tmp_path: Path) -> None:
    """review-248 R8: a loan comes back in the worse of its draw and its asset's own condition (good < worn < damaged,
    a `new` asset `good` once lent) — a worn asset drawn good comes back worn, a good one drawn worn worn — and the loan
    and the asset take it; only a drawn damaged one is written off."""
    _close(world, terms, END, tmp_path)
    for k, loan in enumerate(world.loans.values()):
        loan.until = END + timedelta(days=28)
        world.assets[loan.pos_asset_id].condition = ("new", "good", "worn")[k % 3]
    own = {i: world.assets[loan.pos_asset_id].condition for i, loan in world.loans.items()}
    back = pos.returns(world, terms, date(2023, 2, 28), date(2023, 3, 1))
    assert sorted(back) == sorted(own)
    seen = Counter[tuple[str, str]]()
    for i in back:
        loan, draw = world.loans[i], pos.drawn(terms, i)
        asset, held = world.assets[loan.pos_asset_id], "good" if own[i] == "new" else own[i]
        assert loan.condition_on_return == asset.condition == max(draw, held, key=pos.CONDITIONS.index)
        assert (asset.written_off_on == date(2023, 3, 1)) == (draw == "damaged")
        seen[(held, draw)] += 1
    assert seen[("worn", "good")] and seen[("good", "worn")] and sum(n for (_, d), n in seen.items() if d == "damaged")


def test_the_placements_read_each_store_s_line_and_uplift_from_the_world(opening: Setup, terms: Terms) -> None:
    """review-248 R10: `place` takes a store's line and uplift from `World.display_uplift` (the month-end makes it just
    before): a loan carries the uplift held at its point and a display of its line; a point not held there is never
    lent."""
    first = Checkpoint.of(opening.world).restore()
    dropped = {first.loans[i].delivery_point_id for i in pos.place(first, terms, FIRST)[:20]}
    world = Checkpoint.of(opening.world).restore()
    for point in dropped:
        del world.display_uplift[point]
    for held in world.display_uplift.values():
        held.update(dict.fromkeys(held, 0.123))
    loans = pos.place(world, terms, FIRST)
    assert loans
    for i in loans:
        loan = world.loans[i]
        ((line, uplift),) = world.display_uplift[loan.delivery_point_id].items()
        assert loan.uplift == uplift == 0.123 and line in loan.lines and loan.delivery_point_id not in dropped


# ── T5: the ledger ──────────────────────────────────────────────────────────


def test_the_ledger_writes_off_at_three_years_and_buys_the_pools_back_in_january(
    world: World, terms: Terms, tmp_path: Path
) -> None:
    january = date(2024, 1, 1)
    _close(world, terms, END, tmp_path / "placed")
    out = [i for i, asset in world.assets.items() if asset.loan is not None and asset.acquired_on == date(2021, 1, 1)]
    with CsvWriter(tmp_path) as writer:
        assert not pos.ledger(world, terms, date(2023, 12, 1), Output(writer, NEVER))  # not January: nothing bought
        bought = pos.ledger(world, terms, january, Output(writer, january))
    for asset_id, asset in world.assets.items():
        old = asset.acquired_on == date(2021, 1, 1)
        assert (asset.written_off_on == january) == (old and asset_id not in out), asset.asset_no
        assert asset.loan is None or asset.written_off_on is None
    held = Counter((a.region, a.material) for a in world.assets.values() if a.written_off_on is None)
    pools = {m.code: m.assets_per_region for m in terms.materials.values() if m.kind == "asset"}
    assert held == {(region, code): n for region in terms.regions for code, n in pools.items()}
    assert {world.assets[i].acquired_on for i in bought} == {january}
    rows = list(_csv(tmp_path / "asset_register.csv"))
    spent = Counter[str]()
    for i in bought:
        spent[world.assets[i].region] += round(world.assets[i].cost)
    assert {r["code"]: float(r["cost"]) for r in rows} == {f"MAJ-POS-{r}-2024": spent[r] for r in terms.regions}
    assert len({world.assets[i].asset_no for i in bought}) == len(bought)
    assert world.counters["STO-HARCAFE-PHA"] == 150 + sum(
        1 for i in bought if world.assets[i].asset_no.startswith("STO-HARCAFE-PHA-")
    )


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


# ── T4: the give-aways ──────────────────────────────────────────────────────


def test_give_aways_at_a_share_of_the_visits_a_skilled_region_s_where_visits_convert(
    world: World, terms: Terms, tmp_path: Path
) -> None:
    """At 35 % of the recorded visits that dealt with someone (SMS's generous manager × 1.5); in PHA and JMV 30 % fewer,
    all at the stores whose rep converts above the area's median (`pos.medians`). One item each, `qty` 1–5, its cost
    `unit_cost × qty`."""
    crew = Crew.of(world, WORLD, STORIES)
    for tourplan_id in world.tourplan_of.values():
        t = world.tourplans[tourplan_id]
        t.calls, t.converted = 10, (8 if tourplan_id % 2 else 3)
    pos.medians(world)
    dealt: Counter[str] = Counter()
    given: Counter[str] = Counter()
    above: Counter[str] = Counter()
    giveaways = {m.pos_material_id: m for m in crew.giveaways}
    for day in DAYS:
        visits = field.visit(world, crew, day, field.calls(world, crew, day, field.plan(world, day)))
        rows = {r["visit_id"]: r for r in field.handouts(world, crew, day, visits, [])}
        for v in visits:
            if not v.recorded or v.outcome in field.UNREACHED:
                assert v.visit_id not in rows
                continue
            region = terms.region_of[v.tourplan.area]
            dealt[region] += 1
            r = rows.get(v.visit_id)
            if r is None:
                continue
            given[region] += 1
            above[region] += v.tourplan.converted > 5
            material = giveaways[r["pos_material_id"]]
            assert 1 <= r["qty"] <= 5 and r["cost"] == pytest.approx(material.unit_cost * r["qty"])
            assert material.line is None  # no order: a brand-wide one
    share = WORLD.field.pos_handouts.share
    for region in ("SJC", "ZSC", "VCH"):
        assert given[region] / dealt[region] == pytest.approx(share, abs=0.05)
    assert given["SMS"] / dealt["SMS"] == pytest.approx(share * 1.5, abs=0.06)
    for region in ("PHA", "JMV"):
        assert above[region] == given[region] > 0
        assert given[region] / dealt[region] == pytest.approx(share * 0.7, abs=0.06)


def test_a_give_away_at_a_visit_that_orders_brands_a_line_of_its_order(world: World, terms: Terms) -> None:
    crew = Crew.of(world, WORLD, STORIES)
    demand = world.demand
    assert demand is not None
    gold = next(int(sku) for sku in demand.catalogue.sku_ids if demand.lines[crew.sku_line[int(sku)]].code == "GOLD")
    found = 0
    for day in DAYS:
        visits = [
            v
            for v in field.visit(world, crew, day, field.calls(world, crew, day, field.plan(world, day)))
            if v.recorded and v.outcome not in field.UNREACHED and terms.region_of[v.tourplan.area] == "SJC"
        ]
        events = [
            BasketReady(day, t.customer_id, t.delivery_point_id, "small_retail", ((gold, 2.0, 0.0),), (v.visit_id, 1))
            for v in visits
            for t in (v.tourplan,)
        ]
        for r in field.handouts(world, crew, day, visits, events):
            material = next(m for m in crew.giveaways if m.pos_material_id == r["pos_material_id"])
            assert material.line == "GOLD"
            found += 1
    assert found > 20


def test_a_sales_visit_that_dealt_with_someone_counts_on_its_store_s_conversion(world: World) -> None:
    crew = Crew.of(world, WORLD, STORIES)
    visits = field.visit(world, crew, DAYS[0], field.calls(world, crew, DAYS[0], field.plan(world, DAYS[0])))
    before = {id(v.tourplan): (v.tourplan.calls, v.tourplan.converted) for v in visits}
    field.tally(visits)
    for v in visits:
        calls, converted = before[id(v.tourplan)]
        counted = v.kind == "sales" and v.outcome not in field.UNREACHED
        assert v.tourplan.calls - calls >= counted and v.tourplan.converted - converted <= counted
    assert sum(v.tourplan.calls for v in visits) > 0
