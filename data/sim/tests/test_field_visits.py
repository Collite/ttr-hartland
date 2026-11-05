"""The visits (G2 S2.1 T3, `loop.field`): a planned visit is made with its rep's adherence (A-7's three at 0.55), the
ad-hoc ones add `field.ad_hoc_share`; a visit's kind follows the store's customer, its outcome the kind's shares, and a
sales visit converts with S3's chance (`conversion_base` × the rep's skill), taking the customer's basket as its
`visit` order — never a second one; A-8's rep leaves 70 % unrecorded. The plans are the opening `World`'s over eight
weeks of 2023; the orders' cases run a January of the loop with the visits and without."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import is_working_day
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import day as day_loop
from hartland_sim.loop import demand as demand_step
from hartland_sim.loop import field
from hartland_sim.loop.day import Loop
from hartland_sim.loop.field import PRAGUE, Crew, Visit
from hartland_sim.loop.warmup import Output
from hartland_sim.setup.rows import columns
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
WEEKS = [d for d in (date(2023, 1, 2) + timedelta(days=k) for k in range(56)) if is_working_day(d)]
SKIPPED = STORIES.anomaly("A-7").parameters["adherence"]
SILENT = STORIES.anomaly("A-8").parameters["unrecorded_share"]
BASE = STORIES.story("S3").parameters["conversion_base"]


@pytest.fixture(scope="module")
def crew(opening: Setup) -> Crew:
    return Crew.of(opening.world, WORLD, STORIES)


@pytest.fixture(scope="module")
def visits(opening: Setup, crew: Crew) -> list[tuple[date, list[Visit], int]]:
    """Each working day's visits on the opening `World` (no orders taken), with the day's planned count."""
    world = opening.world
    out = []
    for day in WEEKS:
        planned = field.plan(world, day)
        out.append((day, field.visit(world, crew, day, field.calls(world, crew, day, planned)), len(planned)))
    return out


def test_a_rep_s_adherence_is_the_average_within_its_spread_a_7_s_three_skip(opening: Setup, crew: Crew) -> None:
    reps = {a.employee_id: a for a in opening.world.actors.values() if a.role == "rep"}
    assert set(crew.adherence) == set(reps) == set(crew.unrecorded)
    skipping = [e for e, a in reps.items() if a.skills.skips_visits]
    silent = [e for e, a in reps.items() if a.skills.records_nothing]
    assert len(skipping) == 3 and len(silent) == 1
    spread = WORLD.field.adherence_spread
    for employee, adherence in crew.adherence.items():
        if employee in skipping:
            assert adherence == SKIPPED
        else:
            assert abs(adherence - WORLD.field.adherence) <= spread
        assert crew.unrecorded[employee] == (SILENT if employee in silent else 0.0)
    assert len({round(a, 6) for e, a in crew.adherence.items() if e not in skipping}) > 40  # drawn, not one value
    assert crew.conversion_base == BASE == 0.55


def test_the_planned_visits_made_follow_the_rep_s_adherence(
    visits: list[tuple[date, list[Visit], int]], opening: Setup, crew: Crew
) -> None:
    made: Counter[int] = Counter()
    planned: Counter[int] = Counter()
    for day in WEEKS:
        for p in field.plan(opening.world, day):
            planned[p.tourplan.employee_id] += 1
    for _, day_visits, _ in visits:
        for v in day_visits:
            if v.visit_plan_id is not None:
                made[v.tourplan.employee_id] += 1
    for employee, n in planned.items():
        share = made[employee] / n
        assert share == pytest.approx(crew.adherence[employee], abs=0.08)
        assert (share <= 0.65) == (crew.adherence[employee] == SKIPPED)


def test_the_ad_hoc_visits_add_their_share_at_stores_not_planned_that_day(
    visits: list[tuple[date, list[Visit], int]], opening: Setup
) -> None:
    ad_hoc = sum(v.visit_plan_id is None for _, day_visits, _ in visits for v in day_visits)
    assert ad_hoc / sum(n for _, _, n in visits) == pytest.approx(WORLD.field.ad_hoc_share, abs=0.01)
    for day, day_visits, _ in visits:
        due = {p.tourplan.store for p in field.plan(opening.world, day)}
        stores = [v.tourplan.store for v in day_visits]
        assert len(stores) == len(set(stores))
        assert all(v.tourplan.store not in due for v in day_visits if v.visit_plan_id is None)


def test_a_visit_s_kind_follows_the_customer_and_its_outcome_the_kind_s_shares(
    visits: list[tuple[date, list[Visit], int]], opening: Setup
) -> None:
    world = opening.world
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for day, day_visits, _ in visits:
        for v in day_visits:
            customer = world.customers[v.tourplan.customer_id]
            expected = "sales" if customer.active_on(day) else "acquisition"
            assert v.kind == ("merchandising" if customer.channel == "gas_ka" else expected)
            assert v.outcome != "order"  # the orders are `take_orders`'
            assert not v.converts or (v.kind == "sales" and v.outcome == "no_order")
            by_kind[v.kind][v.outcome] += 1
    for kind, outcomes in by_kind.items():
        n = sum(outcomes.values())
        for name, share in WORLD.field.outcomes[kind].items():
            assert outcomes[name] / n == pytest.approx(share, abs=0.015)
    assert min(sum(by_kind[kind].values()) for kind in ("sales", "acquisition", "merchandising")) > 500


def test_a_sales_visit_converts_with_s3_s_chance_read_on_the_visit(
    visits: list[tuple[date, list[Visit], int]], opening: Setup
) -> None:
    world = opening.world
    fired: Counter[int] = Counter()
    reached: Counter[int] = Counter()
    for _, day_visits, _ in visits:
        for v in day_visits:
            if v.kind == "sales" and v.outcome == "no_order":
                reached[v.tourplan.employee_id] += 1
                fired[v.tourplan.employee_id] += v.converts
    skills = {e: world.actors[e].skills.conversion for e in reached}
    rates = {e: fired[e] / reached[e] for e in reached if reached[e] >= 100}
    assert len(rates) > 30
    for employee, rate in rates.items():
        assert rate == pytest.approx(BASE * skills[employee], abs=0.12)
    ranked = sorted(rates, key=rates.__getitem__)
    order = np.argsort(np.argsort([skills[e] for e in ranked]))
    assert float(np.corrcoef(order, np.arange(len(ranked)))[0, 1]) > 0.8


def test_a_rep_s_day_runs_8_to_17_one_visit_after_the_other_a_8_s_rep_records_30_pc(
    visits: list[tuple[date, list[Visit], int]], crew: Crew
) -> None:
    low, high = WORLD.field.visit_minutes
    recorded: Counter[int] = Counter()
    count: Counter[int] = Counter()
    per_day = Counter((day, v.tourplan.employee_id) for day, day_visits, _ in visits for v in day_visits)
    for day, day_visits, _ in visits:
        last: dict[int, datetime] = {}
        for v in day_visits:
            rep = v.tourplan.employee_id
            count[rep] += 1
            if not v.recorded:
                assert v.time_in is None and v.time_out is None
                continue
            recorded[rep] += 1
            assert v.time_in is not None and v.time_out is not None and v.time_in.tzinfo is PRAGUE
            assert v.time_in.date() == day and time(8) < v.time_in.time() < v.time_out.time() <= time(17)
            if per_day[day, rep] * (high + WORLD.field.travel_minutes[1]) <= 9 * 60:  # a day not compressed
                assert low <= (v.time_out - v.time_in).total_seconds() / 60 <= high
                gap = v.time_in - last[rep] if rep in last else timedelta(days=1)
                assert gap >= timedelta(minutes=WORLD.field.travel_minutes[0])
            assert v.time_in >= last.get(rep, v.time_in)
            last[rep] = v.time_out
    silent = next(e for e, share in crew.unrecorded.items() if share)
    assert recorded[silent] / count[silent] == pytest.approx(1 - SILENT, abs=0.08)
    assert all(recorded[e] == count[e] for e in count if e != silent)


def test_a_visit_row_is_its_columns(visits: list[tuple[date, list[Visit], int]]) -> None:
    day, day_visits, _ = visits[0]
    v = day_visits[0]
    row = field.visit_row(v, day)
    assert tuple(row) == columns("visit")
    assert (row["visit_id"], row["visit_plan_id"], row["day"], row["kind"]) == (
        v.visit_id,
        v.visit_plan_id,
        day,
        v.kind,
    )
    assert (row["employee_id"], row["store_id"], row["customer_id"]) == (
        v.tourplan.employee_id,
        v.tourplan.store_id,
        v.tourplan.customer_id,
    )


# ── the visits' orders ────────────────────────────────────────────────────────


def _world(opening: Setup) -> World:
    return Checkpoint.of(opening.world).restore()


def test_a_converting_visit_takes_the_triggered_basket_or_the_basket_as_it_stands(opening: Setup, crew: Crew) -> None:
    world = _world(opening)
    demand = world.demand
    assert demand is not None
    day = date(2023, 1, 16)
    shares = demand.catalogue.shares(day, len(demand.lines))
    retail = [c for c in world.customers.values() if c.channel == "small_retail" and c.active_on(day) and c.store]
    triggered, standing, empty = retail[0], retail[1], retail[2]
    for customer, cases in ((standing, 4.0), (empty, 0.0)):
        demand.basket[crew.points[customer.customer_id]] = 0.0
        demand.basket[crew.points[customer.customer_id], 0] = cases
    other = _world(opening)  # the basket that triggered today: another World's, taken there
    assert other.demand is not None
    other.demand.basket[crew.points[triggered.customer_id], 0] = 3.0
    events = demand_step.take(other, day, np.array([crew.points[triggered.customer_id]]), shares)
    assert len(events) == 1 and events[0].visit is None

    def converting(customer_id: int) -> Visit:
        tourplan = world.tourplans[world.tourplan_of[world.customers[customer_id].store or ""]]
        return Visit(customer_id, None, tourplan, "sales", "no_order", True, True, None, None)

    made = [converting(c.customer_id) for c in (triggered, standing, empty)]
    out = field.take_orders(world, crew, day, made, events, shares)
    assert [e.customer_id for e in out] == [triggered.customer_id, standing.customer_id]
    assert [e.visit for e in out] == [(v.visit_id, v.tourplan.employee_id) for v in made[:2]]
    assert [v.outcome for v in made] == ["order", "order", "no_order"]
    p = crew.points[standing.customer_id]
    assert demand.basket[p].sum() == 0 and demand.offset[p] == day.toordinal() % demand.cycle[p]
    assert sum(cases for _, cases, _ in out[1].lines) > 0


def _rows(run: Path, table: str) -> list[dict[str, str]]:
    path = run / f"{table}.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _january(opening: Setup, run: Path) -> None:
    with CsvWriter(run) as writer:
        loop = Loop.of(_world(opening), WORLD, STORIES, opening.portfolio.skus, Output(writer, date(2023, 1, 1)))
        loop.until(date(2023, 1, 31))


def test_the_visits_orders_move_the_month_s_cases_by_under_2_pc(
    opening: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _january(opening, tmp_path / "visits")
    monkeypatch.setattr(day_loop, "calls", lambda *_: [])
    _january(opening, tmp_path / "none")
    cases = {}
    for name in ("visits", "none"):
        lines = _rows(tmp_path / name, "sales_order_line")
        cases[name] = sum(float(line["qty_cases"]) for line in lines)
    assert cases["visits"] == pytest.approx(cases["none"], rel=0.02)
    assert not _rows(tmp_path / "none", "visit")
    orders = _rows(tmp_path / "visits", "sales_order")
    visits = {v["visit_id"]: v for v in _rows(tmp_path / "visits", "visit")}
    taken = [o for o in orders if o["visit_id"]]
    assert len(taken) > 1_000
    assert {v["visit_id"] for v in visits.values() if v["outcome"] == "order"} == {o["visit_id"] for o in taken}
    for o in taken:
        v = visits[o["visit_id"]]
        assert (o["source"], o["taken_by_employee_id"], o["customer_id"], o["ordered_on"]) == (
            "visit",
            v["employee_id"],
            v["customer_id"],
            v["day"],
        )
    retail = [o for o in orders if o["channel"] == "small_retail"]
    assert all(o["visit_id"] for o in retail if o["source"] == "visit")
    days = Counter((o["customer_id"], o["ordered_on"]) for o in retail)
    assert max(days.values()) == 1  # one point, one basket: never two orders a day
