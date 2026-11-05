"""The perfect-store check (G2 S2.1 T4, `loop.field.observe`): a recorded visit observes each line of its store's
class (`field.observed_lines`), its facings from the store's line share × its size class's shelf (0 for a line it
does not buy), the competitors' beside them (A-6's Kavalír adds on Gold in East Bohemia from week 45 of 2025), out of
stock once the line's basket holds `field.shelf_cycles` of its order cycles, a secondary placement while a display is
lent, and promotion compliance where a mandatory promotion of the line runs. The visits are the opening `World`'s."""

from __future__ import annotations

from collections import Counter
from datetime import date

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.loop import field
from hartland_sim.loop.field import Crew, Visit
from hartland_sim.loop.pricing import admits
from hartland_sim.setup.rows import Row, columns
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, Loan, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
A6 = STORIES.anomaly("A-6").parameters
DAYS = [date(2023, 1, d) for d in (9, 10, 11, 12, 13, 16, 17, 18, 19, 20)]
SIZE = {"convenience": "S", "minimarket": "S", "village": "S", "petrol": "S", "super": "M"}


@pytest.fixture(scope="module")
def crew(opening: Setup) -> Crew:
    return Crew.of(opening.world, WORLD, STORIES)


def _visits(world: World, crew: Crew, day: date) -> list[Visit]:
    return field.visit(world, crew, day, field.calls(world, crew, day, field.plan(world, day)))


@pytest.fixture(scope="module")
def observed(opening: Setup, crew: Crew) -> list[tuple[date, Visit, list[Row]]]:
    out = []
    for day in DAYS:
        for v in _visits(opening.world, crew, day):
            out.append((day, v, field.observe(opening.world, crew, day, [v])))
    return out


def _line(world: World, row: Row) -> int:
    demand = world.demand
    assert demand is not None
    (i,) = [
        i for i, line in enumerate(demand.lines) if row_id("product_line", code=line.code) == row["product_line_id"]
    ]
    return i


def test_a_recorded_visit_observes_its_store_class_s_lines_an_unrecorded_one_nothing(
    observed: list[tuple[date, Visit, list[Row]]], opening: Setup
) -> None:
    demand = opening.world.demand
    assert demand is not None
    codes = {row_id("product_line", code=line.code): line.code for line in demand.lines}
    assert WORLD.dials.observation_lines == "store_class"
    seen = 0
    for _, v, rows in observed:
        if not v.recorded:
            assert rows == []
            continue
        assert [codes[r["product_line_id"]] for r in rows] == list(WORLD.field.observed_lines[v.tourplan.store_class])
        assert all(tuple(r) == columns("visit_observation") and r["visit_id"] == v.visit_id for r in rows)
        seen += len(rows)
    assert seen / sum(v.recorded for _, v, _ in observed) == pytest.approx(6, abs=1.5)  # 03 §9: ~6 of 11


def test_facings_follow_the_line_share_on_the_size_class_s_shelf_none_where_it_does_not_buy(
    observed: list[tuple[date, Visit, list[Row]]], opening: Setup, crew: Crew
) -> None:
    world = opening.world
    demand = world.demand
    assert demand is not None
    shelf: Counter[str] = Counter()
    stores: Counter[str] = Counter()
    for _, v, rows in observed:
        c = demand.point_customer[crew.points[v.tourplan.delivery_point_id]]
        buys = v.kind != "acquisition"
        for r in rows:
            i = _line(world, r)
            carried = bool(demand.carried[c, i]) and buys
            assert (r["facings"] > 0) == carried
            assert r["competitor_facings"] >= 0
            total = r["facings"] + r["competitor_facings"]
            assert float(r["shelf_share_pct"]) == (round(100 * r["facings"] / total, 2) if total else 0.0)
        if v.kind == "sales" and rows:
            shelf[SIZE[v.tourplan.store_class]] += sum(r["facings"] for r in rows)
            stores[SIZE[v.tourplan.store_class]] += 1
    for size in ("S", "M"):
        assert 0.3 * WORLD.field.facings[size] < shelf[size] / stores[size] <= 1.3 * WORLD.field.facings[size]
    assert shelf["M"] / stores["M"] > 1.5 * shelf["S"] / stores["S"]


def test_a_line_is_out_of_stock_once_its_basket_holds_its_shelf_cycles(opening: Setup, crew: Crew) -> None:
    world = Checkpoint.of(opening.world).restore()  # its baskets filled to 0–2 order cycles of their demand
    demand = world.demand
    assert demand is not None
    filled = 2 * (np.arange(len(demand.basket)) * 0.6180339887 % 1.0)  # spread evenly over the points
    demand.basket[:] = demand.base * (demand.cycle * filled)[:, None]
    out = n = 0
    for day in DAYS[:4]:
        for v in _visits(world, crew, day):
            p = crew.points[v.tourplan.delivery_point_id]
            for r in field.observe(world, crew, day, [v]):
                i = _line(world, r)
                expected = demand.base[p, i] * demand.cycle[p] * WORLD.field.shelf_cycles
                assert r["out_of_stock"] == (r["facings"] > 0 and demand.basket[p, i] >= expected > 0)
                out += r["out_of_stock"]
                n += r["facings"] > 0
    assert out / n == pytest.approx(1 - WORLD.field.shelf_cycles / 2, abs=0.1)


def test_a_lent_display_is_a_secondary_placement(opening: Setup, crew: Crew) -> None:
    world = Checkpoint.of(opening.world).restore()
    day = DAYS[0]
    v = next(v for v in _visits(world, crew, day) if v.recorded and v.kind == "sales")
    rows = field.observe(world, crew, day, [v])
    assert not any(r["secondary_placement"] for r in rows)
    demand = world.demand
    assert demand is not None
    line = demand.lines[_line(world, rows[0])].code
    world.loans[1] = Loan(v.tourplan.delivery_point_id, line, 0.2, day)
    again = field.observe(world, crew, day, [v])
    assert [r["secondary_placement"] for r in again] == [k == 0 for k in range(len(rows))]
    world.loans[1] = Loan(v.tourplan.delivery_point_id, line, 0.2, DAYS[1])  # lent after the visit
    assert not any(r["secondary_placement"] for r in field.observe(world, crew, day, [v]))


def test_a_mandatory_promotion_running_is_executed_or_not_else_compliance_is_null(
    observed: list[tuple[date, Visit, list[Row]]], opening: Setup, crew: Crew
) -> None:
    world = opening.world
    demand = world.demand
    assert demand is not None
    mandatory = [p for p in world.promotions.values() if p.mandatory]
    assert mandatory
    executed: Counter[bool] = Counter()
    for day, v, rows in observed:
        p = crew.points[v.tourplan.delivery_point_id]
        spot, customer = demand.spots[p], world.customers[v.tourplan.customer_id]
        for r in rows:
            line = demand.lines[_line(world, r)].code
            runs = v.kind == "sales" and any(m.line == line and admits(m, spot, customer, day) for m in mandatory)
            assert (r["promo_compliant"] is None) == (not runs)
            if runs:
                executed[r["promo_compliant"]] += 1
    if executed:
        assert executed[True] / sum(executed.values()) == pytest.approx(WORLD.field.promo_compliance, abs=0.1)


def test_a_6_s_kavalir_adds_facings_on_gold_in_east_bohemia_from_week_45(opening: Setup, crew: Crew) -> None:
    world = opening.world
    demand = world.demand
    assert demand is not None
    inside, outside = date.fromisoformat(str(A6["sightings_from"])), date(2025, 10, 20)
    means = {}
    for day in (inside, outside):
        gold: list[int] = []
        other: list[int] = []
        for v in _visits(world, crew, date(2023, 1, 10)):
            if not v.recorded or demand.spots[crew.points[v.tourplan.delivery_point_id]].region != A6["region"]:
                continue
            for r in field.observe(world, crew, day, [v]):
                code = demand.lines[_line(world, r)].code
                (gold if code == A6["line"] else other).append(r["competitor_facings"])
        means[day] = (sum(gold) / len(gold), sum(other) / len(other))
    lift = 1 + A6["competitor_facings_lift"]
    assert means[inside][0] / means[outside][0] == pytest.approx(lift, rel=0.25)
    assert means[inside][1] / means[outside][1] == pytest.approx(1, rel=0.15)
