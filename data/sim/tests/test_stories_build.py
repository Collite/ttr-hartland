"""The DONE line of G1 S1.2 (T7): a CSV build from the warm-up writes nothing before 2024-01-01, starts that day with a
year of history, and the S1, S2 and S4 detectors' pandas twins pass on it; on the null world they fail.

The build tier (`pytest -m build`, its own CI job): two builds of the world from setup to 2025-12-31 — S1's window is
2024–2025 and S4's is 2025 — each a few minutes. The day loop and the order writer are S1.4's, so the build here is a
fixture: weather, acquisition and churn, demand, and each `BasketReady` written as an order with the columns the twins
read (`sales_order`, `sales_order_line`) — S1.4 writes the real rows. The calendar's promotions (`promotion`,
`campaign`, `promotion_eligibility`) are setup's own rows (S1.3).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from hartland_sim.checks.stories import Verdict, check
from hartland_sim.cli import main
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import acquisition, demand
from hartland_sim.loop.warmup import Output
from hartland_sim.loop.weather import Climate
from hartland_sim.setup.market import Market
from hartland_sim.setup.run import run as setup

pytestmark = pytest.mark.build

UNTIL = date(2025, 12, 31)
JUDGED = ("detectors/s1.sql", "detectors/s1_weather.sql", "detectors/s2_affluence.sql", "detectors/s4.sql")


@dataclass(frozen=True)
class Built:
    run: Path
    gained_in_warm_up: int  # potential at the opening, a customer by 2024-01-01
    lost_in_warm_up: int
    buying_with_a_basket: float  # of the points buying on 2024-01-01, the share whose basket holds cases


def build(run: Path, stories: StoriesConfig, market: Market) -> Built:
    """Setup, then the days from the warm-up to `UNTIL` (module docstring)."""
    world = WorldConfig.load()
    made = setup(world, stories, market)
    state = made.world
    assert state.demand is not None
    terms = demand.Terms.of(state.demand, stories)
    reps = acquisition.Reps.of(state, stories)
    climate = Climate.load()
    potential = {c.customer_id for c in state.customers.values() if c.status == "potential"}
    gained = lost = 0
    snapshot = 0.0
    with CsvWriter(run) as writer, _Stub(run) as stub:
        writer.write_all({table: rows for table, rows in made.tables.items() if table != "stock_day"})
        output = Output.of(writer, world)
        day = world.history.warm_up_from
        while day <= UNTIL:
            if day == world.history.first_day:
                demand_state = state.demand
                today = day.toordinal()
                buying = (demand_state.customer_since <= today) & (today <= demand_state.customer_until)
                points = buying[demand_state.point_customer]
                snapshot = float((demand_state.basket[points].sum(axis=1) > 0).mean())
            temps = np.array([climate.day(region, day).temp_mean_c for region in state.demand.regions])
            output.append(day, "weather_daily", climate.rows(day))
            shares = state.demand.catalogue.shares(day, len(state.demand.lines))  # the day's, read by both steps
            outcome = acquisition.step(state, reps, day, shares)
            if not output.writes(day):
                gained += sum(customer in potential for customer in outcome.acquired)
                lost += len(outcome.lost)
            events = demand.step(state, terms, day, temps, shares)
            if output.writes(day):
                stub.orders(events)
            state.day = day
            day += timedelta(days=1)
    return Built(run, gained, lost, snapshot)


class _Stub:
    """S1.4's order writer, stubbed: only the columns the twins read."""

    def __init__(self, run: Path) -> None:
        self.run = run
        run.mkdir(parents=True, exist_ok=True)
        self.files = [
            (run / name).open("w", encoding="utf-8", newline="") for name in ("sales_order.csv", "sales_order_line.csv")
        ]
        self.order, self.line = (csv.writer(f, lineterminator="\n") for f in self.files)
        self.order.writerow(["sales_order_id", "customer_id", "delivery_point_id", "channel", "ordered_on"])
        self.line.writerow(["sales_order_id", "line_no", "sku_id", "qty_cases", "qty_pieces", "is_free_goods"])
        self.n = 0

    def __enter__(self) -> _Stub:
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self.files:
            handle.close()

    def orders(self, events: list[demand.BasketReady]) -> None:
        for event in events:
            self.n += 1
            self.order.writerow([self.n, event.customer_id, event.delivery_point_id, event.channel, event.day])
            for k, (sku, cases, pieces) in enumerate(event.lines, start=1):
                self.line.writerow([self.n, k, sku, cases, pieces, "false"])


def _null(stories: StoriesConfig) -> StoriesConfig:
    return replace(
        stories,
        stories=tuple(replace(story, enabled=False) for story in stories.stories),
        anomalies=tuple(replace(anomaly, enabled=False) for anomaly in stories.anomalies),
    )


@pytest.fixture(scope="module")
def story_world(tmp_path_factory: pytest.TempPathFactory, osm_market: Market) -> Built:
    return build(tmp_path_factory.mktemp("story-world"), StoriesConfig.load(), osm_market)


@pytest.fixture(scope="module")
def null_world(tmp_path_factory: pytest.TempPathFactory, osm_market: Market) -> Built:
    return build(tmp_path_factory.mktemp("null-world"), _null(StoriesConfig.load()), osm_market)


def _first(path: Path, column: str) -> str:
    with path.open(encoding="utf-8", newline="") as source:
        return min(row[column] for row in csv.DictReader(source))


def _judged(verdicts: list[Verdict]) -> dict[str, Verdict]:
    return {v.check: v for v in verdicts if v.check in JUDGED}


def test_the_warm_up_writes_nothing_before_2024(story_world: Built) -> None:
    assert _first(story_world.run / "sales_order.csv", "ordered_on") == "2024-01-01"
    assert _first(story_world.run / "weather_daily.csv", "day") == "2024-01-01"


def test_2024_starts_with_a_year_of_history(story_world: Built) -> None:
    assert story_world.gained_in_warm_up > 1000  # 0.03 a month of ~5 000 potential stores
    assert story_world.lost_in_warm_up > 300  # 0.012 a month of ~3 500 customers
    assert story_world.buying_with_a_basket > 0.9
    with (story_world.run / "sales_order.csv").open(encoding="utf-8") as source:
        first_day = sum(1 for row in csv.DictReader(source) if row["ordered_on"] == "2024-01-01")
    assert first_day > 500  # the trade orders on the first day as on any Monday


def test_the_s1_s2_and_s4_twins_pass_on_the_story_world(story_world: Built, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stories-check", "--csv", str(story_world.run)]) == 0
    text = capsys.readouterr().out
    for check_sql in JUDGED:
        (row,) = [line for line in text.splitlines() if f"`{check_sql}`" in line]
        assert row.split(" | ")[-1].startswith("pass"), row  # the result, and a note after it (S1: each line's ratio)
    assert "| S4 | `kmeans` |" in text and (story_world.run / "stories.md").read_text(encoding="utf-8") == text


def test_the_null_world_fails_them(null_world: Built) -> None:
    found = _judged(check(null_world.run, StoriesConfig.load()))
    assert set(found) == set(JUDGED)
    assert not any(v.passed for v in found.values()), [(v.check, v.value) for v in found.values()]
    assert main(["stories-check", "--csv", str(null_world.run), "--null"]) == 0  # each gives the null world's side
