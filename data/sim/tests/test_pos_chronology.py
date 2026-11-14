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


def _loop(opening: Setup, world: World, run: Path) -> tuple[Loop, CsvWriter]:
    writer = CsvWriter(run)
    return Loop.of(world, WORLD, STORIES, opening.portfolio.skus, Output(writer, NEVER)), writer


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


def _placed(world: World) -> dict[int, Loan]:
    return {loan_id: loan for loan_id, loan in world.loans.items() if loan.lent_on == FIRST}


def _victim(world: World) -> tuple[int, Loan]:
    """A placement of `FIRST` at a customer the churn can lose (not a planted case's), its customer's only one."""
    placed = _placed(world)
    per = Counter(loan.customer_id for loan in placed.values())
    return next(
        (loan_id, loan)
        for loan_id, loan in sorted(placed.items())
        if per[loan.customer_id] == 1 and not world.customers[loan.customer_id].traits
    )


def _run(opening: Setup, world: World, run: Path, *days: date) -> Loop:
    loop, writer = _loop(opening, world, run)
    with writer:
        for day in days:
            loop.run(day)
    return loop


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
        (FIRST, None, True),  # acquired on the loan's day: its first day, included
        (FIRST + timedelta(days=1), None, False),  # not a customer yet on the loan's day
    ],
    ids=["until-day-before", "until-the-day", "since-the-day", "since-day-after"],
)
def test_the_loan_s_day_is_judged_by_the_customer_s_inclusive_since_and_until(
    scheduled: Checkpoint, since: date | None, until: date | None, kept: bool
) -> None:
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
    world = scheduled.restore()
    loan_id, loan = _victim(world)
    asset = world.assets[loan.pos_asset_id]
    before = state_rows(world, WORLD)["pos_asset"]
    left = world.display_budget[loan.region]
    _lose(monkeypatch, FIRST, loan.customer_id)
    _run(opening, world, tmp_path, FIRST)
    assert asset.loan is None and asset.written_off_on is None  # in the region's stock for the next month-end
    assert world.display_budget[loan.region] == round(left + asset.cost, 2)
    rows = state_rows(world, WORLD)
    assert loan_id not in {r["pos_loan_id"] for r in rows["pos_loan"]}
    assert rows["pos_asset"] == before  # the asset's row as if never lent: its condition, no write-off
    others = {r for r in world.display_budget if r != loan.region}
    assert {r: world.display_budget[r] for r in others} == {r: scheduled.restore().display_budget[r] for r in others}


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
    planned = _placed(scheduled.restore())
    loan_id, loan = _victim(scheduled.restore())
    _lose(monkeypatch, FIRST, loan.customer_id)
    straight = _run(opening, _at(opening, END - timedelta(days=1)), tmp_path / "straight", END, FIRST).world
    restarted = _run(opening, scheduled.restore(), tmp_path / "restarted", FIRST).world
    for world in (straight, restarted):
        assert _placed(world) == {i: held for i, held in planned.items() if i != loan_id}
    assert straight.loans == restarted.loans and straight.lent == restarted.lent
    assert straight.display_budget == restarted.display_budget
    assert {i: a.loan for i, a in straight.assets.items()} == {i: a.loan for i, a in restarted.assets.items()}


def test_the_placements_are_revalidated_after_the_acquisitions_and_before_the_demand_and_the_visits() -> None:
    assert STEPS.index("acquisition") < STEPS.index("placements") < STEPS.index("demand") < STEPS.index("field")
