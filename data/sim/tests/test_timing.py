"""Timing (G1 S1.4 T7) on the build tier: the story world's quarters measured day by day give what a one-year build
takes (a soft budget: a warning and the printed table, never a failure — the executor's load is not the code's), and
the per-day cost of the demand step and the orders step on a written day of 2024 is measured and printed. The unit
tier checks the table's arithmetic."""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hartland_sim.checks.timing import BUDGET_S, one_year, table
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import demand, orders
from hartland_sim.loop.build import BUILD_JSON, Built
from hartland_sim.loop.pricing import Rules
from hartland_sim.loop.warmup import Output
from hartland_sim.loop.weather import Climate
from hartland_sim.world import Checkpoint, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
DAY = date(2024, 1, 2)  # a written Tuesday


def test_one_year_is_setup_the_quarters_to_the_year_and_the_end() -> None:
    report = {
        "timings_s": {"setup": 10.0, "end": 5.0},
        "quarters": {"2023-Q4": {"days": 92, "seconds": 20.0}, "2024-Q1": {"days": 91, "seconds": 40.0}}
        | {"2025-Q1": {"days": 90, "seconds": 99.0}},
    }
    assert one_year(report, 2024) == 75.0
    text = table(report, 2024)
    assert "| 2024-Q1 | 91 | 40.0 | 440 |" in text and f"within the {BUDGET_S:.0f} s budget" in text


@pytest.mark.build
def test_a_one_year_build_fits_its_budget(story_world: Built, capsys: pytest.CaptureFixture[str]) -> None:
    report = json.loads((story_world.run / BUILD_JSON).read_text(encoding="utf-8"))
    estimate = one_year(report, 2024)
    with capsys.disabled():
        print("\n" + table(report, 2024))
    if estimate > BUDGET_S:
        warnings.warn(f"a one-year build takes ≈ {estimate:.0f} s, over the {BUDGET_S:.0f} s budget", stacklevel=1)
    assert 0 < estimate < report["timings_s"]["total"]


def _at(story_world: Built) -> World:
    return Checkpoint.load(story_world.run, date(2023, 12, 31)).restore()


def _cost(make: Callable[[], tuple[Any, ...]], step: Callable[..., Any], rounds: int = 3) -> tuple[float, Any]:
    """The fastest of `rounds` runs of `step` on fresh arguments (`make`, not timed), in ms; the last run's result.
    pytest-benchmark's fixture would do this, but its 5.3.0 fails on pytest 9.2 (`Function._nodeid`) and the pins
    stay (Found)."""
    best, result = float("inf"), None
    for _ in range(rounds):
        arguments = make()
        started = time.perf_counter()
        result = step(*arguments)
        best = min(best, 1000 * (time.perf_counter() - started))
    return best, result


def _temps(world: World) -> Any:
    assert world.demand is not None
    climate = Climate.load()
    return np.array([climate.day(region, DAY).temp_mean_c for region in world.demand.regions])


@pytest.mark.build
def test_the_demand_and_orders_steps_cost_a_day(
    story_world: Built, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def demand_args() -> tuple[Any, ...]:
        world = _at(story_world)
        assert world.demand is not None
        return world, demand.Terms.of(world.demand, STORIES), DAY, _temps(world)

    writers: list[CsvWriter] = []

    def orders_args() -> tuple[Any, ...]:
        world = _at(story_world)
        assert world.demand is not None
        events = demand.step(world, demand.Terms.of(world.demand, STORIES), DAY, _temps(world))
        desk = orders.Desk.of(world, Rules.of(world, WORLD, STORIES), WORLD)
        writers.append(CsvWriter(tmp_path / f"round-{len(writers)}"))
        return world, desk, DAY, events, Output(writers[-1], WORLD.history.first_day)

    demand_ms, events = _cost(demand_args, demand.step)
    orders_ms, done = _cost(orders_args, orders.step)
    for writer in writers:
        writer.close()
    with capsys.disabled():
        print(
            f"\n{DAY}: the demand step {demand_ms:.0f} ms ({len(events)} baskets), the orders step {orders_ms:.0f} ms"
            f" ({done.orders} orders, {done.lines} lines, written)"
        )
    assert len(events) > 500 and done.orders > 500 and done.lines > 4 * done.orders
