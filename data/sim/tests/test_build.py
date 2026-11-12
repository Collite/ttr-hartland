"""A build into `build/<run>/` (G1 S1.4 T4, contracts §2): setup's tables at the start, the loop's rows through the
warm-up switch, and at the end the five state tables rendered from the `World` (the budgets from the first written
quarter), the promotions' status on the last day, `_sim_meta`, the checkpoint and `build.json`; the null world's own
stories hash. The unit tier builds a few days — the warm-up's first, written here by moving the first written day;
the build tier (`test_build_world.py`) builds 2024 and 2025."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from hartland_sim.cli import main
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.loop.build import BUILD_JSON, build, restatus
from hartland_sim.setup import opening as opening_setup
from hartland_sim.setup.market import Market
from hartland_sim.setup.opening import STATE_TABLES
from hartland_sim.world import Checkpoint, Meta

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
EARLY = replace(WORLD, history=replace(WORLD.history, first_day=date(2023, 1, 1)))  # the loop writes from day one


def _rows(run: Path, table: str) -> list[dict[str, str]]:
    path = run / f"{table}.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


@pytest.fixture(scope="module")
def early(tmp_path_factory: pytest.TempPathFactory, osm_market: Market) -> Path:
    run = tmp_path_factory.mktemp("early") / "run"
    build(run, date(2023, 1, 3), EARLY, STORIES, market=osm_market)
    return run


def test_a_build_writes_setup_s_tables_then_the_loop_s_rows(early: Path) -> None:
    assert not _rows(early, "stock_day")  # the opening's, before the warm-up: not a build's
    assert len(_rows(early, "store")) > 10_000 and _rows(early, "promotion")
    days = {r["ordered_on"] for r in _rows(early, "sales_order")}
    assert days == {"2023-01-01", "2023-01-02", "2023-01-03"}  # the web and the own shops order on a holiday too
    assert {r["day"] for r in _rows(early, "weather_daily")} == {"2023-01-01", "2023-01-02", "2023-01-03"}
    assert _rows(early, "sales_order_line") and _rows(early, "price_agreement")


def test_the_end_renders_the_state_tables_meta_the_checkpoint_and_build_json(early: Path) -> None:
    for table in STATE_TABLES:  # no display is lent before the first month-end (G2 S2.3): the file, with no row
        assert _rows(early, table) or table == "pos_loan", table
    assert (early / "pos_loan.csv").is_file()
    assert {r["quarter"] for r in _rows(early, "discount_budget")} == {"2023-01-01"}
    meta = Meta.load(early)
    assert (meta.last_day_done, meta.stories_sha256, meta.master_seed) == (
        date(2023, 1, 3),
        STORIES.sha256,
        WORLD.master_seed,
    )
    (checkpoint,) = [Checkpoint.load(early)]
    assert checkpoint.day == date(2023, 1, 3) and checkpoint.restore().day == date(2023, 1, 3)
    report = json.loads((early / BUILD_JSON).read_text(encoding="utf-8"))
    assert {"master_seed", "stories_sha256", "git_sha", "timings_s", "quarters", "build_id"} <= set(report)
    assert report["quarters"]["2023-Q1"]["orders"] == len(_rows(early, "sales_order"))
    assert report["build_id"] == meta.build_id and report["null"] is False


def test_the_promotions_carry_their_status_on_the_last_day(early: Path) -> None:
    statuses = {r["code"]: r["status"] for r in _rows(early, "promotion")}
    assert set(statuses.values()) == {"approved"}  # every written promotion runs from late December 2023 on


def test_restatus_moves_only_the_status(tmp_path: Path) -> None:
    path = tmp_path / "promotion.csv"
    path.write_text("code,valid_from,valid_to,status\nA,2024-01-01,2024-01-31,approved\nB,2024-02-01,2024-02-28,x\n")
    restatus(path, date(2024, 2, 1))
    assert path.read_text().splitlines()[1:] == ["A,2024-01-01,2024-01-31,closed", "B,2024-02-01,2024-02-28,running"]


def test_a_run_is_built_once(early: Path, osm_market: Market) -> None:
    with pytest.raises(FileExistsError):
        build(early, date(2023, 1, 3), EARLY, STORIES, market=osm_market)


def test_the_null_world_has_a_stories_hash_and_a_build_id_of_its_own() -> None:
    day = date(2024, 12, 31)
    story, null = opening_setup.meta(WORLD, STORIES, day), opening_setup.meta(WORLD, STORIES.null(), day)
    assert null.stories_sha256 != story.stories_sha256 and null.build_id != story.build_id
    assert null.world_sha256 == story.world_sha256
    assert STORIES.null().sha256 == STORIES.null().sha256
    assert not any(s.enabled for s in STORIES.null().stories) and not any(a.enabled for a in STORIES.null().anomalies)


def test_the_verb_builds_into_its_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "verb"
    assert main(["build-world", "--until", "2023-01-01", "--csv", "--null", "--run", str(run)]) == 0
    assert "build id" in capsys.readouterr().out
    assert Meta.load(run).stories_sha256 == STORIES.null().sha256
    assert json.loads((run / BUILD_JSON).read_text(encoding="utf-8"))["null"] is True
    assert not _rows(run, "sales_order")  # the warm-up writes nothing
    assert main(["build-world", "--until", "2023-01-01", "--csv", "--run", str(run)]) == 1
