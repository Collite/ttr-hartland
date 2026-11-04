"""The quarter-end's G1 half (G1 S1.4 T3b, ⚑R213-A): Q+1's standard list — each line's price index moved by its drift
(stream `pricing_agree` by quarter and line), the 2025-Q2 coffee jump on the KAVA lines, the purchase prices on the
same index, the SKUs sold in Q+1 — written from its first day; and Q+1's discount budgets from the month-end's plan.
Setup writes no price-list row: the rolls write every version from the first written period."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop.pricing import Rules, standard_list
from hartland_sim.loop.warmup import Output
from hartland_sim.period import Closing, month_end, quarter_end
from hartland_sim.rng import rng
from hartland_sim.setup.org import MARKETING
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
LOW, HIGH = WORLD.calendar.price_drift_pct
JUMP = 1 + WORLD.calendar.coffee_jump.pct / 100
NEVER = date(2100, 1, 1)  # an `Output` that writes nothing
Q2_2024 = date(2024, 4, 1)


@pytest.fixture
def world(opening: Setup) -> World:
    return Checkpoint.of(opening.world).restore()


@pytest.fixture(scope="module")
def closing(opening: Setup) -> Closing:
    rules = Rules.of(opening.world, WORLD, STORIES)
    return Closing.of(opening.world, WORLD, STORIES, opening.portfolio.skus, rules)


def _at(world: World, opening: Setup, first: date, index: dict[str, float]) -> None:
    """The `World` as a quarter-end before `first` finds it: the list of the quarter before at `index`."""
    world.price_index = dict(index)
    world.standard_list = standard_list(opening.portfolio.skus, first - timedelta(days=1), index)


def _roll(world: World, closing: Closing, day: date) -> None:
    with CsvWriter(Path("/nonexistent")) as writer:
        quarter_end.roll_lists(world, closing, day, Output(writer, NEVER))


def test_q2_2024_moves_each_line_by_its_drift_purchase_prices_too(
    world: World, closing: Closing, opening: Setup
) -> None:
    before = {line: 1.0 + 0.01 * n for n, line in enumerate(closing.rules.lines)}
    _at(world, opening, Q2_2024, before)
    old = world.standard_list
    assert old is not None and old.code == "SC-2024-Q1"
    _roll(world, closing, date(2024, 3, 31))
    new = world.standard_list
    assert new is not None and (new.code, new.valid_from, new.valid_to) == ("SC-2024-Q2", Q2_2024, date(2024, 6, 30))
    skus = {sku.sku_id: sku for sku in opening.portfolio.skus}
    for line in closing.rules.lines:
        moved = world.price_index[line] / before[line]
        assert LOW / 100 <= moved - 1 <= HIGH / 100
        assert moved - 1 == pytest.approx(
            rng("pricing_agree", "drift", Q2_2024, row_id("product_line", code=line)).uniform(LOW, HIGH) / 100
        )
    for sku_id, price in new.prices.items():
        sku = skus[sku_id]
        index = world.price_index[sku.line]
        assert price.case == round(sku.base_price_case * index, 2)  # the base × the index, never a rounded price moved
        assert price.purchase_case == round(sku.cogs_case * index, 2)
        assert price.piece == round(price.case / sku.units_per_case, 2)
        if sku_id in old.prices:
            ratio = price.case / old.prices[sku_id].case
            assert 1 + LOW / 100 - 0.002 <= ratio <= 1 + HIGH / 100 + 0.002


def test_the_same_quarter_and_line_draw_the_same_drift_whatever_the_order(
    world: World, closing: Closing, opening: Setup
) -> None:
    lines = list(closing.rules.lines)
    forward = [quarter_end.drift(closing, Q2_2024, line) for line in lines]
    backward = [quarter_end.drift(closing, Q2_2024, line) for line in reversed(lines)][::-1]
    assert forward == backward and len(set(forward)) == len(lines)
    other = Checkpoint.of(world).restore()
    for state in (world, other):
        _at(state, opening, Q2_2024, {line: 1.0 for line in lines})
    _roll(world, closing, date(2024, 3, 31))
    _roll(other, closing, date(2024, 3, 31))
    assert world.price_index == other.price_index
    assert quarter_end.drift(closing, date(2024, 7, 1), lines[0]) != forward[0]


def test_sc_2025_q2_jumps_only_the_coffee_lines_by_9_percent_more(
    world: World, closing: Closing, opening: Setup
) -> None:
    first = date(2025, 4, 1)
    assert WORLD.calendar.coffee_jump.starts == first
    _at(world, opening, first, {line: 1.0 for line in closing.rules.lines})
    _roll(world, closing, date(2025, 3, 31))
    for line in closing.rules.lines:
        drift = 1 + quarter_end.drift(closing, first, line) / 100
        jump = JUMP if closing.business[line] == "KAVA" else 1.0
        assert world.price_index[line] == pytest.approx(drift * jump)
    assert {closing.business[line] for line in closing.rules.lines} == {"KAVA", "KULINAR"}
    _at(world, opening, date(2025, 7, 1), {line: 1.0 for line in closing.rules.lines})
    _roll(world, closing, date(2025, 6, 30))
    assert all(index < 1 + HIGH / 100 + 1e-9 for index in world.price_index.values())  # the jump is once


def test_the_launches_of_2024_04_02_are_listed_in_sc_2024_q2_at_their_line_s_index(
    world: World, closing: Closing, opening: Setup
) -> None:
    launches = [sku for sku in opening.portfolio.skus if sku.launched_on == date(2024, 4, 2)]
    assert launches
    _at(world, opening, Q2_2024, {line: 1.0 for line in closing.rules.lines})
    assert world.standard_list is not None and not set(world.standard_list.prices) & {s.sku_id for s in launches}
    _roll(world, closing, date(2024, 3, 31))
    assert world.standard_list is not None
    for sku in launches:
        assert world.standard_list.prices[sku.sku_id].case == round(
            sku.base_price_case * world.price_index[sku.line], 2
        )


def test_the_rows_are_written_from_q_plus_1_s_first_day_by_marketing_s_director(
    world: World, closing: Closing, opening: Setup, tmp_path: Path
) -> None:
    first = WORLD.history.first_day
    with CsvWriter(tmp_path) as writer:
        output = Output(writer, first)
        quarter_end.roll_lists(world, closing, date(2023, 3, 31), output)  # SC-2023-Q2: not written
        world.standard_list = standard_list(opening.portfolio.skus, date(2023, 12, 1), world.price_index)
        quarter_end.roll_lists(world, closing, date(2023, 12, 31), output)  # SC-2024-Q1: the first written
    with (tmp_path / "standard_price_list.csv").open(encoding="utf-8") as source:
        (head,) = list(csv.DictReader(source))
    assert (head["code"], head["valid_from"], head["valid_to"]) == ("SC-2024-Q1", "2024-01-01", "2024-03-31")
    assert head["issued_by_employee_id"] == str(opening.org.actors[MARKETING].employee_id)
    assert quarter_end.issuer(world) == opening.org.actors[MARKETING].employee_id
    with (tmp_path / "standard_price.csv").open(encoding="utf-8") as source:
        items = list(csv.DictReader(source))
    assert world.standard_list is not None and len(items) == len(world.standard_list.prices)
    with pytest.raises(ValueError, match="quarter's last day"):
        quarter_end.roll_lists(world, closing, date(2024, 3, 30), output)


def test_setup_writes_no_price_list_row(opening: Setup) -> None:
    for table in ("standard_price_list", "standard_price", "channel_price_list", "channel_price"):
        assert not opening.tables.get(table)
    assert opening.world.price_index == {line: 1.0 for line in opening.world.price_index}
    assert opening.world.standard_list is not None and opening.world.standard_list.code == "SC-2023-Q1"


def test_q2_2024_s_budgets_are_the_rule_recomputed_from_the_plan(
    world: World, closing: Closing, opening: Setup
) -> None:
    day = date(2024, 3, 31)
    _at(world, opening, Q2_2024, {line: 1.0 for line in closing.rules.lines})
    with CsvWriter(Path("/nonexistent")) as writer:
        output = Output(writer, NEVER)
        quarter_end.roll_lists(world, closing, day, output)
        rows = month_end.plan(world, closing, day, output)
    budgets = quarter_end.roll_budgets(world, closing, day)
    assert {first for _, first in world.budgets if first == Q2_2024} == {Q2_2024}
    shares = WORLD.pricing.budgets
    areas = {row_id("area", code=area): area for _, _, area in closing.targets if area}
    seen_rep = seen_kam = 0
    for duty in world.duties:
        if not duty.on(Q2_2024):
            continue
        if duty.kind == "rep_area":
            area_id = row_id("area", code=duty.target)
            planned = sum(r["planned_net_revenue"] for r in rows if r["area_id"] == area_id)
            assert area_id in areas or planned == 0
            pct = shares.rep_discount_pct_of_planned_gross
            assert budgets[duty.employee_id].budget_amount == pytest.approx(pct / 100 * planned, abs=0.05)
            seen_rep += 1
        elif duty.kind == "kam_customer" and duty.target.startswith("KA-"):
            customers = [
                d.target
                for d in world.duties
                if d.kind == "kam_customer" and d.employee_id == duty.employee_id and d.on(Q2_2024)
            ]
            ids = {row_id("customer", code=code) for code in customers}
            planned = sum(r["planned_net_revenue"] for r in rows if r["customer_id"] in ids)
            pct = shares.kam_discount_pct_of_planned_gross
            assert budgets[duty.employee_id].budget_amount == pytest.approx(pct / 100 * planned, rel=1e-6)
            seen_kam += 1
    assert seen_rep == WORLD.salesforce.reps and seen_kam >= 8
    kams = [e for e, actor in world.actors.items() if actor.role == "kam"]
    assert set(kams) <= set(budgets)
    assert all(world.budgets[e, Q2_2024] is budgets[e] for e in budgets)
    assert all(budget.used_amount == budget.reserved == 0 for budget in budgets.values())
