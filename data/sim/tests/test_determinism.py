"""The determinism battery (G1 S1.4 T5, `05` §7). The unit tier: the measures — a build's per-file hashes, and its
closed documents by a day, the rows a prolong to that day must give as a straight build does whatever was appended
after or in which order — and a prolong of a few days against a straight build. The build tier (`-m build`): two
2024 builds byte-equal, prolongs from 2024-04-09 to 05-09 and across 2024-03-31 equal to the straight 2025 build on
the closed documents, another seed another world."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from hartland_sim.checks.determinism import closed, hashes
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.loop.build import BUILD_JSON, Built, build, prolong
from hartland_sim.setup.market import Market
from hartland_sim.world import Checkpoint, Meta

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
EARLY = replace(WORLD, history=replace(WORLD.history, first_day=date(2023, 1, 1)))  # the loop writes from day one


def _write(run: Path, table: str, header: list[str], rows: list[list[str]]) -> None:
    run.mkdir(parents=True, exist_ok=True)
    with (run / f"{table}.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _run(run: Path, orders: list[list[str]], extra_plan: bool = False) -> Path:
    _write(run, "sales_order", ["sales_order_id", "ordered_on"], orders)
    lines = [[f"{o[0]}1", o[0], "2.000"] for o in orders]
    _write(run, "sales_order_line", ["sales_order_line_id", "sales_order_id", "qty_cases"], lines)
    _write(run, "sales_order_line_discount", ["sales_order_line_id", "seq"], [[line[0], "1"] for line in lines])
    plan = [["2024-02-01", "3", "1.0"], ["2024-04-01", "1", "2.0"]]  # made 2024-01-31 · 2024-01-31
    if extra_plan:
        plan.append(["2024-05-01", "1", "9.0"])  # made 2024-02-29: after the day
    _write(run, "sales_plan", ["month", "version", "planned_cases"], plan)
    _write(run, "store", ["store_id"], [["1"], ["2"]])
    _write(run, "customer", ["customer_id", "until"], [["1", ""]])
    return run


def test_closed_documents_ignore_the_rows_after_the_day_and_the_order_appended(tmp_path: Path) -> None:
    day = date(2024, 2, 10)
    straight = _run(tmp_path / "a", [["1", "2024-02-01"], ["2", "2024-02-10"], ["3", "2024-03-01"]], extra_plan=True)
    prolonged = _run(tmp_path / "b", [["2", "2024-02-10"], ["1", "2024-02-01"]])
    _write(prolonged, "customer", ["customer_id", "until"], [["1", "2024-02-10"]])  # rendered at its end: not closed
    assert closed(straight, day) == closed(prolonged, day)
    assert set(closed(straight, day)) == {
        "sales_order",
        "sales_order_line",
        "sales_order_line_discount",
        "sales_plan",
        "store",
    }
    assert hashes(straight) != hashes(prolonged)


def test_a_changed_closed_row_changes_its_table_s_hash(tmp_path: Path) -> None:
    day = date(2024, 2, 10)
    one = _run(tmp_path / "a", [["1", "2024-02-01"]])
    other = _run(tmp_path / "b", [["1", "2024-02-01"]])
    _write(other, "sales_order_line", ["sales_order_line_id", "sales_order_id", "qty_cases"], [["11", "1", "3.000"]])
    found, changed = closed(one, day), closed(other, day)
    assert [name for name in found if found[name] != changed[name]] == ["sales_order_line"]


def test_a_prolong_is_the_straight_build_on_the_closed_documents(tmp_path: Path, osm_market: Market) -> None:
    """T5's battery in miniature (the build tier runs it on 2024): a build to the 3rd prolonged to the 5th gives the
    closed documents of a straight build to the 5th, and only the days after the 3rd are appended."""
    day = date(2023, 1, 5)
    straight = build(tmp_path / "straight", day, EARLY, STORIES, market=osm_market)
    prolonged = tmp_path / "prolonged"
    build(prolonged, date(2023, 1, 3), EARLY, STORIES, market=osm_market)
    prolong(prolonged, day, EARLY, STORIES)
    assert closed(prolonged, day, EARLY) == closed(straight.run, day, EARLY)
    assert Meta.load(prolonged).last_day_done == day and Checkpoint.load(prolonged).day == day
    with (prolonged / "sales_order.csv").open(encoding="utf-8") as source:
        days = {r["ordered_on"] for r in csv.DictReader(source)}
    assert days == {f"2023-01-0{d}" for d in range(1, 6)}
    report = json.loads((prolonged / BUILD_JSON).read_text(encoding="utf-8"))
    assert [p["until"] for p in report["prolongs"]] == ["2023-01-05"]
    with pytest.raises(ValueError, match="past"):
        prolong(prolonged, day, EARLY, STORIES)
    with pytest.raises(ValueError, match="other inputs"):
        prolong(prolonged, date(2023, 1, 9), replace(EARLY, master_seed=EARLY.master_seed + 1), STORIES)


# ── the build tier ─────────────────────────────────────────────────────────────

DAY_100, DAY_130 = date(2024, 4, 9), date(2024, 5, 9)  # 2024's 100th and 130th days


@pytest.fixture(scope="module")
def twice(tmp_path_factory: pytest.TempPathFactory, osm_market: Market) -> tuple[Path, Path]:
    """Two builds of the same inputs to the end of January 2024."""
    runs = [tmp_path_factory.mktemp(name) / "run" for name in ("first", "second")]
    for run in runs:
        build(run, date(2024, 1, 31), WORLD, STORIES, market=osm_market)
    return runs[0], runs[1]


@pytest.mark.build
def test_two_builds_of_the_same_seed_are_byte_equal(twice: tuple[Path, Path]) -> None:
    first, second = twice
    found = hashes(first)
    assert found == hashes(second)
    assert {"sales_order.csv", "_sim_checkpoint.csv", "_sim_meta.csv", "discount_budget.csv"} <= set(found)


@pytest.mark.build
@pytest.mark.parametrize(
    ("built", "until"),
    [(DAY_100, DAY_130), (date(2024, 3, 20), date(2024, 4, 19))],  # the second pair crosses 2024-03-31's roll
    ids=["day-100-to-130", "across-2024-03-31"],
)
def test_a_prolong_is_the_straight_build_on_the_closed_documents_by_its_day(
    tmp_path: Path, osm_market: Market, story_world: Built, built: date, until: date
) -> None:
    run = tmp_path / "run"
    build(run, built, WORLD, STORIES, market=osm_market)
    prolong(run, until, WORLD, STORIES)
    prolonged, straight = closed(run, until), closed(story_world.run, until)
    assert set(prolonged) == set(straight)
    assert [name for name in straight if prolonged[name] != straight[name]] == []
    if built < date(2024, 3, 31) < until:  # the roll ran in the restored World: its list, index and budgets
        with (run / "standard_price_list.csv").open(encoding="utf-8") as source:
            assert {r["code"] for r in csv.DictReader(source)} == {"SC-2024-Q1", "SC-2024-Q2"}
        state = Checkpoint.load(run).restore()
        assert {q for _, q in state.budgets} >= {date(2024, 4, 1)}


@pytest.mark.build
def test_another_master_seed_builds_another_world(tmp_path: Path, osm_market: Market, story_world: Built) -> None:
    day = date(2024, 1, 7)
    run = tmp_path / "other"
    build(run, day, replace(WORLD, master_seed=WORLD.master_seed + 1), STORIES, market=osm_market)
    other, ours = closed(run, day), closed(story_world.run, day)
    for table in ("sales_order", "sales_order_line", "weather_daily", "customer_condition", "standard_price"):
        assert other[table] != ours[table], table
    assert Meta.load(run).build_id != story_world.meta.build_id
