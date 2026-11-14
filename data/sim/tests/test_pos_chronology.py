"""The point of sale's effective day (G2 S2.5, `loop.pos.revalidate`): a month-end lends its displays from the next
day, before that day's acquisitions and losses run; after them, before the day's demand, a placement whose customer
is not active on its day is cancelled — no loan, no asset taken, no budget spent — and every other placement keeps
its id and its draws, a restart from the month-end's checkpoint included. The `World`s are the opening's moved to
2023-01-30, so a case runs the warm-up's first month-end and the day after it."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import pos
from hartland_sim.loop.day import STEPS, Loop
from hartland_sim.loop.pricing import channel_list, standard_list
from hartland_sim.loop.warmup import Output
from hartland_sim.rng import uniforms
from hartland_sim.setup.opening import state_rows
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, Loan, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
END, FIRST = date(2023, 1, 31), date(2023, 2, 1)  # the warm-up's first month-end and its placements' day
NEVER = date(2100, 1, 1)  # an `Output` that writes nothing


def _at(opening: Setup, day: date) -> World:
    """The opening `World` moved to `day`, its lists those of `day + 1`'s month (as `test_day`'s)."""
    world = Checkpoint.of(opening.world).restore()
    world.day = day
    skus = opening.portfolio.skus
    following = day + timedelta(days=1)
    world.standard_list = standard_list(skus, following)
    world.channel_lists = {c: channel_list(world.standard_list, skus, WORLD, c, following) for c in world.channel_lists}
    return world


def _loop(opening: Setup, world: World, run: Path, first_day: date = NEVER) -> tuple[Loop, CsvWriter]:
    writer = CsvWriter(run)
    return Loop.of(world, WORLD, STORIES, opening.portfolio.skus, Output(writer, first_day)), writer


@pytest.fixture(scope="module")
def scheduled(opening: Setup, tmp_path_factory: pytest.TempPathFactory) -> Checkpoint:
    """The `World` after the month-end of `END`, its placements lent from `FIRST`: its checkpoint, saved and loaded
    as a build keeps it."""
    run = tmp_path_factory.mktemp("scheduled")
    loop, writer = _loop(opening, _at(opening, END - timedelta(days=1)), run)
    with writer:
        loop.run(END)
    Checkpoint.of(loop.world).save(run)
    return Checkpoint.load(run, END)


def _lose(monkeypatch: pytest.MonkeyPatch, day: date, *customers: int) -> None:
    """The churn draws of `customers` on `day` forced to 0, so each is lost that day; every other draw is its own."""
    lost = np.array(customers, dtype=np.int64)

    def forced(stream: str, *key: object, ids: np.ndarray) -> np.ndarray:
        out = uniforms(stream, *key, ids=ids)
        return np.where(np.isin(ids, lost), 0.0, out) if (stream, key) == ("churn", (day,)) else out

    monkeypatch.setattr("hartland_sim.loop.acquisition.uniforms", forced)


def _placed(world: World, day: date = FIRST) -> dict[int, Loan]:
    return {loan_id: loan for loan_id, loan in world.loans.items() if loan.lent_on == day}


def _victim(world: World, day: date = FIRST) -> tuple[int, Loan]:
    """A placement of `day` at a customer the churn can lose (not a planted case's), its customer's only one."""
    placed = _placed(world, day)
    per = Counter(loan.customer_id for loan in placed.values())
    return next(
        (loan_id, loan)
        for loan_id, loan in sorted(placed.items())
        if per[loan.customer_id] == 1 and not world.customers[loan.customer_id].traits
    )


def _cancelled(planned: dict[int, Loan], world: World, day: date = FIRST) -> set[int]:
    """The placements of `day` whose customer is not active on it after the run: the victim, and any other whose
    customer the churn's own draws lost that day (a few in a hundred 1sts)."""
    return {loan_id for loan_id, loan in planned.items() if not world.customers[loan.customer_id].active_on(day)}


def _refunded(
    budget: dict[str, float], world: World, planned: dict[int, Loan], cancelled: set[int]
) -> dict[str, float]:
    """The month-end's `budget`, each cancelled placement's cost back in its region's."""
    out = dict(budget)
    for loan_id in cancelled:
        loan = planned[loan_id]
        out[loan.region] += world.assets[loan.pos_asset_id].cost
    return out


def _run(opening: Setup, world: World, run: Path, *days: date, first_day: date = NEVER) -> Loop:
    loop, writer = _loop(opening, world, run, first_day)
    with writer:
        for day in days:
            loop.run(day)
    return loop


def _written(*runs: Path) -> dict[str, list[str]]:
    """Each CSV file the runs wrote, by name: its header, then each run's rows in turn."""
    out: dict[str, list[str]] = {}
    for run in runs:
        for path in sorted(run.glob("*.csv")):
            header, *rows = path.read_text(encoding="utf-8").splitlines()
            out[path.name] = out.get(path.name, [header]) + rows
    return out


def test_a_placement_whose_customer_is_lost_on_its_day_is_cancelled(
    opening: Setup, scheduled: Checkpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1's case, from the source: the month-end lends from the 1st to a customer active then; the 1st's acquisition
    step loses it — its `until` the month-end's day, so it is not active on the 1st — and no loan of the 1st is left at
    it. Every placement left is at a customer active on its day."""
    world = scheduled.restore()
    loan_id, loan = _victim(world)
    customer = world.customers[loan.customer_id]
    assert customer.active_on(FIRST) and customer.until is None  # what the month-end saw
    _lose(monkeypatch, FIRST, loan.customer_id)
    _run(opening, world, tmp_path, FIRST)
    lost = world.customers[loan.customer_id]
    assert (lost.status, lost.until) == ("lost", END) and not lost.active_on(FIRST)
    assert loan_id not in world.loans and loan.delivery_point_id not in world.lent
    assert all(world.customers[held.customer_id].active_on(FIRST) for held in _placed(world).values())


@pytest.mark.parametrize(
    ("since", "until", "kept"),
    [
        (None, END, False),  # lost on the loan's day: its last day the day before
        (None, FIRST, True),  # lost the day after: the loan's day is its last, included
        (FIRST, None, True),  # `since` the loan's day: its first day, included
        (FIRST + timedelta(days=1), None, False),  # `since` the day after: not a customer yet on the loan's day
    ],
    ids=["until-day-before", "until-the-day", "since-the-day", "since-day-after"],
)
def test_the_loan_s_day_is_judged_by_the_customer_s_inclusive_since_and_until(
    scheduled: Checkpoint, since: date | None, until: date | None, kept: bool
) -> None:
    """`revalidate` judges by `Customer.active_on`, `since` and `until` both included. The `since` cases pin the
    predicate on an active customer: the loop lends no placement to a store acquired on its day — it was potential
    at the month-end (`loop.pos`, the effective day)."""
    world = scheduled.restore()
    loan_id, loan = _victim(world)
    customer = world.customers[loan.customer_id]
    customer.since, customer.until = since or customer.since, until
    customer.status = "lost" if until is not None else "active"
    assert pos.revalidate(world, FIRST) == ([] if kept else [loan_id])
    assert (loan_id in world.loans) is kept and (world.lent.get(loan.delivery_point_id) == loan_id) is kept


def test_a_cancelled_placement_takes_no_asset_spends_no_budget_and_writes_no_loan(
    opening: Setup, scheduled: Checkpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asset is in its region's stock, its cost back in the region's budget (and in no other's), no `pos_loan` row
    is written, and the next month-end's returns do not bring it back: its condition and write-off are those it had."""
    world = scheduled.restore()
    planned, budget = _placed(world), dict(world.display_budget)
    loan_id, loan = _victim(world)
    asset = world.assets[loan.pos_asset_id]
    condition = asset.condition
    _lose(monkeypatch, FIRST, loan.customer_id)
    loop = _run(opening, world, tmp_path, FIRST)
    cancelled = _cancelled(planned, world)
    assert loan_id in cancelled
    assert asset.loan is None and asset.written_off_on is None  # in the region's stock for the next month-end
    assert world.display_budget == pytest.approx(_refunded(budget, world, planned, cancelled), abs=0.005)
    assert loan_id not in {r["pos_loan_id"] for r in state_rows(world, WORLD)["pos_loan"]}
    # a loan kept at its lost customer would come back at the month-end, in a drawn condition
    assert loan_id not in pos.returns(world, loop.closing.pos, date(2023, 2, 28), date(2023, 3, 1))
    assert (asset.condition, asset.written_off_on) == (condition, None)


def test_a_customer_lost_the_day_after_keeps_its_loan_until_the_month_end_brings_it_back(
    opening: Setup, scheduled: Checkpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The return policy is the month-end's, as built: the loan ran on the customer's last day, and its asset waits
    for the month-end's return."""
    world = scheduled.restore()
    loan_id, loan = _victim(world)
    loop, writer = _loop(opening, world, tmp_path)
    with writer:
        loop.run(FIRST)
        _lose(monkeypatch, FIRST + timedelta(days=1), loan.customer_id)
        loop.run(FIRST + timedelta(days=1))
    assert world.customers[loan.customer_id].until == FIRST
    assert world.lent[loan.delivery_point_id] == loan_id and world.loans[loan_id].returned_on is None
    end, first = date(2023, 2, 28), date(2023, 3, 1)
    assert loan_id in pos.returns(world, loop.closing.pos, end, first) and world.loans[loan_id].returned_on == first


def test_a_restart_from_the_month_end_keeps_every_valid_placement_s_id_and_draws(
    opening: Setup, scheduled: Checkpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The month-end's checkpoint holds the placements of the 1st; run from it, the 1st cancels the same one as the
    straight run does, and the rest are the month-end's, id for id: asset, store, uplift and term (its draw)."""
    base = scheduled.restore()
    planned = _placed(base)
    loan_id, loan = _victim(base)
    _lose(monkeypatch, FIRST, loan.customer_id)
    straight = _run(opening, _at(opening, END - timedelta(days=1)), tmp_path / "straight", END, FIRST).world
    restarted = _run(opening, scheduled.restore(), tmp_path / "restarted", FIRST).world
    cancelled = _cancelled(planned, straight)
    assert loan_id in cancelled
    for world in (straight, restarted):
        assert _placed(world) == {i: held for i, held in planned.items() if i not in cancelled}
    assert straight.loans == restarted.loans and straight.lent == restarted.lent
    assert straight.display_budget == restarted.display_budget
    assert {i: a.loan for i, a in straight.assets.items()} == {i: a.loan for i, a in restarted.assets.items()}


def test_a_restart_across_the_year_end_is_the_straight_run_s_whole_world(
    opening: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """31 December runs the quarter's, the month's and the year's ends, and is the warm-up's last day; 1 January is the
    first written day, as in a build. Run from 31 December's checkpoint, 1 January cancels the placements the straight
    run cancels, the whole `World` is the same, and each file the straight run writes is the eve's rows, then the
    restart's."""
    eve, new_year = date(2023, 12, 31), date(2024, 1, 1)
    kept = _run(opening, _at(opening, eve - timedelta(days=1)), tmp_path / "eve", eve, first_day=new_year).world
    Checkpoint.of(kept).save(tmp_path / "held")  # beside the eve's files, not among them
    planned = _placed(kept, new_year)
    loan_id, loan = _victim(kept, new_year)
    _lose(monkeypatch, new_year, loan.customer_id)
    straight = _run(
        opening, _at(opening, eve - timedelta(days=1)), tmp_path / "straight", eve, new_year, first_day=new_year
    ).world
    saved = Checkpoint.load(tmp_path / "held", eve).restore()
    restarted = _run(opening, saved, tmp_path / "restarted", new_year, first_day=new_year).world
    cancelled = _cancelled(planned, straight, new_year)
    assert loan_id in cancelled
    assert _placed(straight, new_year) == {i: held for i, held in planned.items() if i not in cancelled}
    assert straight == restarted  # by value, as `test_world`'s restarts (pickles may share objects otherwise)
    written = _written(tmp_path / "straight")
    assert {"visit.csv", "rep_prompt.csv", "sales_order.csv"} <= written.keys()  # the field's and the orders' rows
    assert written == _written(tmp_path / "eve", tmp_path / "restarted")


def test_the_placements_are_revalidated_after_the_acquisitions_and_before_the_demand_and_the_visits() -> None:
    assert STEPS.index("acquisition") < STEPS.index("placements") < STEPS.index("demand") < STEPS.index("field")
