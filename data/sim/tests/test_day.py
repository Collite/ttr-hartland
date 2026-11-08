"""The day loop (G1 S1.4 T2, `05` §3): a day runs its steps in order — the stubs called, returning nothing — then the
period ends on the last calendar day of each (a quarter's: the list roll, the month-end, the budgets; 31 December the
year-end after them), and `last_day_done` moves; the warm-up switch writes nothing before 2024-01-01. The `World`s here
are the opening's moved to the days a case needs, so each case runs a few days, not the warm-up."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import day as day_loop
from hartland_sim.loop.day import STEPS, Loop
from hartland_sim.loop.pricing import channel_list, standard_list
from hartland_sim.loop.warmup import Output
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, Duty, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
STUBS = ("field", "supply", "delivery", "finance", "shops", "documents")


def _at(opening: Setup, day: date) -> World:
    """The opening `World` moved to `day` (the day before the case's first), its lists those of `day + 1`'s month."""
    world = Checkpoint.of(opening.world).restore()
    world.day = day
    skus = opening.portfolio.skus
    following = day + timedelta(days=1)
    world.standard_list = standard_list(skus, following)
    world.channel_lists = {c: channel_list(world.standard_list, skus, WORLD, c, following) for c in world.channel_lists}
    return world


def _loop(opening: Setup, world: World, run: Path, first_day: date = date(2023, 1, 1)) -> tuple[Loop, CsvWriter]:
    writer = CsvWriter(run)
    return Loop.of(world, WORLD, STORIES, opening.portfolio.skus, Output(writer, first_day)), writer


def _rows(run: Path, table: str) -> list[dict[str, str]]:
    path = run / f"{table}.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def test_a_day_runs_its_steps_in_order_and_the_stubs_return_nothing(
    opening: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Any]] = []
    for name in STEPS:
        original = getattr(Loop, name)

        def spy(self: Loop, day: date, _name: str = name, _original: Any = original) -> Any:
            result = _original(self, day)
            calls.append((_name, result))
            return result

        monkeypatch.setattr(Loop, name, spy)
    loop, writer = _loop(opening, _at(opening, date(2023, 1, 9)), tmp_path)
    with writer:
        loop.run(date(2023, 1, 10))
    assert [name for name, _ in calls] == list(STEPS)
    assert STEPS.index("acquisition") < STEPS.index("demand") < STEPS.index("orders") < STEPS.index("checkpoint")
    assert all(result is None for name, result in calls if name in STUBS)
    assert (loop.world.day, loop.last_day_done) == (date(2023, 1, 10), date(2023, 1, 10))
    assert loop.report.quarters["2023-Q1"]["orders"] == len(_rows(tmp_path, "sales_order")) > 100
    with pytest.raises(ValueError, match="next day"):
        loop.run(date(2023, 1, 12))


@pytest.mark.parametrize(
    ("day", "hooks"),
    [
        (date(2024, 3, 30), []),
        (date(2024, 1, 31), ["month_end"]),
        (date(2024, 2, 29), ["month_end"]),
        (date(2024, 3, 31), ["roll_lists", "month_end", "roll_budgets"]),
        (date(2024, 12, 31), ["roll_lists", "month_end", "roll_budgets", "year_end"]),
    ],
)
def test_the_period_ends_of_a_day_in_their_order(opening: Setup, tmp_path: Path, day: date, hooks: list[str]) -> None:
    loop, writer = _loop(opening, _at(opening, date(2023, 1, 9)), tmp_path)
    with writer:
        assert [name for name, _ in loop.ends(day)] == hooks


def test_a_quarter_s_last_day_rolls_the_list_then_the_month_then_the_budgets(opening: Setup, tmp_path: Path) -> None:
    loop, writer = _loop(opening, _at(opening, date(2023, 3, 29)), tmp_path)
    world = loop.world
    with writer:
        loop.until(date(2023, 4, 2))
    assert world.standard_list is not None and world.standard_list.code == "SC-2023-Q2"
    assert {listed.valid_from for listed in world.channel_lists.values()} == {date(2023, 4, 1)}
    assert {first for _, first in world.budgets} >= {date(2023, 4, 1)}
    budgets = [b.budget_amount for (_, first), b in world.budgets.items() if first == date(2023, 4, 1)]
    assert len(budgets) >= WORLD.salesforce.reps and sum(budgets) > 0
    days = {r["ordered_on"] for r in _rows(tmp_path, "sales_order")}
    assert days == {"2023-03-30", "2023-03-31", "2023-04-01", "2023-04-02"}  # April's orders at April's lists
    assert {r["valid_from"] for r in _rows(tmp_path, "standard_price_list")} == {"2023-04-01"}
    assert loop.last_day_done == date(2023, 4, 2)


def test_the_warm_up_switch_writes_from_2024_and_the_last_warm_up_day_is_kept(opening: Setup, tmp_path: Path) -> None:
    first = WORLD.history.first_day
    world = _at(opening, date(2023, 12, 29))
    loop, writer = _loop(opening, world, tmp_path, first)
    with writer:
        loop.until(date(2024, 1, 2))
    assert {r["ordered_on"] for r in _rows(tmp_path, "sales_order")} == {"2024-01-01", "2024-01-02"}
    assert {r["day"] for r in _rows(tmp_path, "weather_daily")} == {"2024-01-01", "2024-01-02"}
    assert {r["code"] for r in _rows(tmp_path, "standard_price_list")} == {"SC-2024-Q1"}
    assert {r["valid_from"] for r in _rows(tmp_path, "channel_price_list")} == {"2024-01-01"}
    assert {r["month"] for r in _rows(tmp_path, "customer_condition")} == {"2024-01-01"}
    assert min(r["month"] for r in _rows(tmp_path, "sales_plan")) == "2024-01-01"
    assert all(r["valid_from"] >= "2024-01-01" for r in _rows(tmp_path, "price_agreement"))
    prompted = _rows(tmp_path, "rep_prompt")  # G2 S2.2: none before the data (1 January is a holiday: no plan)
    assert prompted and {r["generated_on"] for r in prompted} == {"2024-01-02"}
    visits = {r["visit_id"] for r in _rows(tmp_path, "visit")}
    assert all(r["visit_id"] in visits for r in prompted if r["visit_id"])
    assert loop.warm_up_end is not None and loop.warm_up_end.day == date(2023, 12, 31)
    kept = loop.warm_up_end.restore()
    assert kept.day == date(2023, 12, 31) and kept.standard_list is not None
    assert kept.standard_list.code == "SC-2024-Q1"


def test_the_module_names_every_step_as_a_method() -> None:
    assert all(callable(getattr(Loop, name, None)) for name in STEPS)
    assert day_loop.quarter(date(2024, 12, 31)) == "2024-Q4"


def test_a_rep_area_duty_added_mid_run_is_a_move_day(opening: Setup, tmp_path: Path) -> None:
    """review-231 R6: `Loop.moves` is built again whenever `duties` is another list or another length, as
    `World.duty`'s index is — a duty the month-end's turnover adds later re-cuts the tourplans on its first day."""
    world = Checkpoint.of(opening.world).restore()
    loop, writer = _loop(opening, world, tmp_path)
    with writer:
        before = loop.moves
        held = next(d for d in world.duties if d.kind == "rep_area")
        day = date(2024, 5, 6)
        world.duties.append(Duty(held.employee_id, "rep_area", held.target, day))
        assert day not in before and loop.moves == before | {day}
        world.duties = [d for d in world.duties if d.valid_from != day]  # another list
        assert loop.moves == before
