"""The determinism battery (G1 S1.4 T5, `05` §7). The unit tier: the measures — a build's per-file hashes, and its
closed documents by a day, the rows a prolong to that day must give as a straight build does whatever was appended
after or in which order — and a prolong of a few days against a straight build: refused on a run of other inputs
(⚑R221-A), and run again after one that stopped, with no row written twice (review-221 R2). The build tier
(`-m build`): two 2024 builds byte-equal, prolongs from 2024-04-09 to 05-09 and across 2024-03-31 equal to the straight
2025 build on the closed documents, another seed another world."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

import hartland_sim
from hartland_sim.checks.determinism import closed, hashes
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.loop import build as builder
from hartland_sim.loop.build import BUILD_JSON, PROLONG_MARK, Built, build, prolong
from hartland_sim.loop.day import Loop
from hartland_sim.setup.market import Market
from hartland_sim.world import Checkpoint, Meta

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
EARLY = replace(WORLD, history=replace(WORLD.history, first_day=date(2023, 1, 1)))  # the loop writes from day one
BUILT, DAY = date(2023, 1, 3), date(2023, 1, 5)  # the miniature battery's build and its prolong's day


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


def test_closed_documents_without_the_orders_file_skip_the_lines_parents(tmp_path: Path) -> None:
    run = _run(tmp_path / "run", [["1", "2024-02-01"]])
    (run / "sales_order.csv").unlink()
    found = closed(run, date(2024, 2, 10))
    assert "sales_order" not in found and {"sales_order_line", "sales_order_line_discount"} <= set(found)


@pytest.fixture(scope="module")
def early(tmp_path_factory: pytest.TempPathFactory, osm_market: Market) -> tuple[Path, Path]:
    """The miniature battery's runs, built once: to the 3rd (each test prolongs a copy) and straight to the 5th."""
    built = build(tmp_path_factory.mktemp("built") / "run", BUILT, EARLY, STORIES, market=osm_market)
    straight = build(tmp_path_factory.mktemp("straight") / "run", DAY, EARLY, STORIES, market=osm_market)
    return built.run, straight.run


def _copy(run: Path, to: Path) -> Path:
    return Path(shutil.copytree(run, to / "run"))


def _order_ids(run: Path) -> list[str]:
    with (run / "sales_order.csv").open(encoding="utf-8") as source:
        return [r["sales_order_id"] for r in csv.DictReader(source)]


def test_a_prolong_is_the_straight_build_on_the_closed_documents(tmp_path: Path, early: tuple[Path, Path]) -> None:
    """T5's battery in miniature (the build tier runs it on 2024): a build to the 3rd prolonged to the 5th gives the
    closed documents of a straight build to the 5th, and only the days after the 3rd are appended."""
    day = DAY
    prolonged = _copy(early[0], tmp_path)
    assert prolong(prolonged, day, EARLY, STORIES).recovered is None
    assert closed(prolonged, day, EARLY) == closed(early[1], day, EARLY)
    assert not (prolonged / PROLONG_MARK).exists()
    assert Meta.load(prolonged).last_day_done == day and Checkpoint.load(prolonged).day == day
    with (prolonged / "sales_order.csv").open(encoding="utf-8") as source:
        days = {r["ordered_on"] for r in csv.DictReader(source)}
    assert days == {f"2023-01-0{d}" for d in range(1, 6)}
    report = json.loads((prolonged / BUILD_JSON).read_text(encoding="utf-8"))
    assert [p["until"] for p in report["prolongs"]] == ["2023-01-05"]
    with pytest.raises(ValueError, match="past"):
        prolong(prolonged, day, EARLY, STORIES)
    with pytest.raises(ValueError, match="other inputs .the master seed differ"):
        prolong(prolonged, date(2023, 1, 9), replace(EARLY, master_seed=EARLY.master_seed + 1), STORIES)


def test_a_prolong_refuses_a_run_of_another_build_id(tmp_path: Path, early: tuple[Path, Path]) -> None:
    """⚑R221-A: the inventory, the reference files, the version and the revision are in `build_id`, not in the three
    hashes."""
    run = _copy(early[0], tmp_path)
    replace(Meta.load(run), build_id="0" * 16).save(run)
    before = hashes(run)
    with pytest.raises(ValueError, match="other inputs .the inventory, reference files, version or generator revision"):
        prolong(run, DAY, EARLY, STORIES)
    assert hashes(run) == before and not (run / PROLONG_MARK).exists()


def test_a_prolong_refuses_a_run_of_another_generator_revision(
    tmp_path: Path, early: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """⚑R221-C (re-review 1 RR1): a bump of `REVISION` alone refuses the run, by name, without touching it."""
    run = _copy(early[0], tmp_path)
    before = hashes(run)
    monkeypatch.setattr(hartland_sim, "REVISION", hartland_sim.REVISION + 1)
    with pytest.raises(ValueError, match="other inputs .the inventory, reference files, version or generator revision"):
        prolong(run, DAY, EARLY, STORIES)
    assert hashes(run) == before and not (run / PROLONG_MARK).exists()


def test_a_marker_older_than_the_meta_cuts_nothing(tmp_path: Path, early: tuple[Path, Path]) -> None:
    """RR3: a prolong that stopped after saving `_sim_meta` finished; its marker is removed, no file cut back."""
    run = _copy(early[0], tmp_path)
    builder._mark(run, BUILT)  # as the prolong below writes it
    stale = (run / PROLONG_MARK).read_text(encoding="utf-8")
    prolong(run, DAY, EARLY, STORIES)
    (run / PROLONG_MARK).write_text(stale, encoding="utf-8")
    before = hashes(run)
    with pytest.raises(ValueError, match="past"):
        prolong(run, BUILT, EARLY, STORIES)
    assert hashes(run) == before and not (run / PROLONG_MARK).exists()


def test_a_prolong_stopped_in_its_loop_runs_again_as_the_straight_build(
    tmp_path: Path, early: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2: the 4th is appended, the 5th raises; the next prolong cuts the files back and runs from the 3rd again."""
    run = _copy(early[0], tmp_path)
    ran = Loop.run

    def stopping(loop: Loop, day: date) -> None:
        if day == DAY:
            raise KeyboardInterrupt
        ran(loop, day)

    monkeypatch.setattr(Loop, "run", stopping)
    with pytest.raises(KeyboardInterrupt):
        prolong(run, DAY, EARLY, STORIES)
    assert (run / PROLONG_MARK).is_file() and "2023-01-04" in (run / "sales_order.csv").read_text(encoding="utf-8")
    monkeypatch.undo()
    assert prolong(run, DAY, EARLY, STORIES).recovered == BUILT
    assert closed(run, DAY, EARLY) == closed(early[1], DAY, EARLY)
    ids = _order_ids(run)
    assert len(ids) == len(set(ids)) == len(_order_ids(early[1]))
    assert not (run / PROLONG_MARK).exists() and Meta.load(run).last_day_done == DAY


def test_a_prolong_stopped_at_its_meta_runs_again_as_the_straight_build(
    tmp_path: Path, early: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2: the checkpoint of the 5th is saved, `_sim_meta` is not; the prolong's first day's checkpoint is kept."""
    run = _copy(early[0], tmp_path)

    def failing(meta: Meta, into: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Meta, "save", failing)
    with pytest.raises(OSError, match="disk full"):
        prolong(run, DAY, EARLY, STORIES)
    assert Checkpoint.load(run).day == DAY and Meta.load(run).last_day_done == BUILT
    monkeypatch.undo()
    assert prolong(run, DAY, EARLY, STORIES).recovered == BUILT
    assert closed(run, DAY, EARLY) == closed(early[1], DAY, EARLY)
    ids = _order_ids(run)
    assert len(ids) == len(set(ids))
    assert Checkpoint.load(run, BUILT).day == BUILT  # kept until the next prolong's end


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
