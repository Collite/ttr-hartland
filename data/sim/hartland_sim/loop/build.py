"""A build into `build/<run>/` (`cli build-world`, G1 S1.4 T4; contracts §2), and prolong's loop half (T5): setup, the
day loop from the warm-up to the last day, the state at the end.

- **At the start**, setup's tables (`setup.run`'s rows), all but the opening `stock_day`: it is dated the day before
  the warm-up, a stock the warm-up moves, and the stock days are G3's from the first written day.
- **Day by day**, the loop's rows through the warm-up switch (`loop.day`), under the build's master seed.
- **At the end**:
  - the five state tables rendered from the `World` (`opening.state_rows`, ⚑R201-A), `discount_budget` from the first
    written quarter on (the warm-up's quarters stay in the `World`);
  - the calendar's promotions with their `status` on the last day (setup writes them as of `history.until`);
  - `_sim_meta`, its `last_day_done` the last day;
  - the checkpoint of the last day, beside the warm-up's last day's;
  - `build.json`: the seed, the stories' hash, the git sha, the timings, and what the run did by quarter (orders,
    grants, refusals and holds by reason).
- **The null world** (`null`): every story and planted case off (`StoriesConfig.null`), with a stories hash of its
  own; `world_sha256` and `build_id` hash the files on disk.

**Prolong** (`prolong`): the loop resumed from a run's last checkpoint to a later day, its rows appended to the run's
files, then the same end — the state tables and the promotions rendered again, `_sim_meta` and the checkpoint moved.
The run's `_sim_meta` must be this checkout's: the seed, `world.yaml`, the stories (or their null world). The
database's side (`load-world` appends, the world reacts) is G5 S5.3's.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import quarter_start
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop.day import Loop, Report
from hartland_sim.loop.warmup import Output
from hartland_sim.rng import using_master_seed
from hartland_sim.setup import opening
from hartland_sim.setup import portfolio as portfolio_setup
from hartland_sim.setup.market import Market
from hartland_sim.setup.opening import REPO_DIR, STATE_TABLES, state_rows
from hartland_sim.setup.rows import tables
from hartland_sim.setup.run import run as setup
from hartland_sim.world import Checkpoint, Meta

BUILD_JSON = "build.json"


@dataclass(frozen=True)
class Built:
    run: Path
    meta: Meta
    report: Report
    timings: dict[str, float]


def build(
    run: Path,
    until: date,
    world: WorldConfig | None = None,
    stories: StoriesConfig | None = None,
    null: bool = False,
    market: Market | None = None,
) -> Built:
    """Setup and the days from the warm-up to `until` into `run` (module docstring); `FileExistsError` when `run`
    already holds a build. `market` saves rebuilding the OSM market (tests)."""
    if run.exists() and any(run.glob("*.csv")):
        raise FileExistsError(f"{run} already holds a build")
    world = world or WorldConfig.load()
    stories = stories or StoriesConfig.load()
    if null:
        stories = stories.null()
    started = time.perf_counter()
    made = setup(world, stories, market)
    timings = {"setup": time.perf_counter() - started}
    with using_master_seed(world.master_seed), CsvWriter(run) as writer:
        writer.write_all({name: rows for name, rows in made.tables.items() if name != "stock_day"})
        loop = Loop.of(made.world, world, stories, made.portfolio.skus, Output.of(writer, world))
        mark = time.perf_counter()
        loop.until(until)
        timings["loop"] = time.perf_counter() - mark
    mark = time.perf_counter()
    meta = _end(run, loop, world, stories)
    timings["end"] = time.perf_counter() - mark
    timings["total"] = time.perf_counter() - started
    _report(run, meta, loop.report, timings, null, until)
    return Built(run, meta, loop.report, timings)


def prolong(run: Path, until: date, world: WorldConfig | None = None, stories: StoriesConfig | None = None) -> Built:
    """`run`'s loop from its last checkpoint to `until` (module docstring); `ValueError` when `run` is not a build of
    these inputs or is past `until`."""
    world = world or WorldConfig.load()
    stories = stories or StoriesConfig.load()
    held = Meta.load(run)
    null = held.stories_sha256 == stories.null().sha256
    if null:
        stories = stories.null()
    current = opening.meta(world, stories, held.last_day_done)
    if (held.master_seed, held.world_sha256, held.stories_sha256) != (
        current.master_seed,
        current.world_sha256,
        current.stories_sha256,
    ):
        raise ValueError(f"{run} is a build of other inputs (its _sim_meta: {held}); build it again")
    if until <= held.last_day_done:
        raise ValueError(f"{run} is done to {held.last_day_done}, past {until}")
    started = time.perf_counter()
    state = Checkpoint.load(run).restore()
    if state.day != held.last_day_done:
        raise ValueError(f"{run}'s last checkpoint is of {state.day}, its _sim_meta says {held.last_day_done}")
    skus = portfolio_setup.build(tables(), world, stories).skus
    timings: dict[str, float] = {}
    with using_master_seed(world.master_seed), CsvWriter(run) as writer:
        loop = Loop.of(state, world, stories, skus, Output.of(writer, world))
        mark = time.perf_counter()
        loop.until(until)
        timings["loop"] = time.perf_counter() - mark
    mark = time.perf_counter()
    meta = _end(run, loop, world, stories)
    timings["end"] = time.perf_counter() - mark
    timings["total"] = time.perf_counter() - started
    _report(run, meta, loop.report, timings, null, until, prolonged=True)
    return Built(run, meta, loop.report, timings)


def _end(run: Path, loop: Loop, world: WorldConfig, stories: StoriesConfig) -> Meta:
    """The end of a build or a prolong (module docstring)."""
    state = loop.world
    rendered = state_rows(state, world)
    first = world.history.first_day
    rendered["discount_budget"] = [r for r in rendered["discount_budget"] if r["quarter"] >= quarter_start(first)]
    for name in STATE_TABLES:  # a prolong's run holds the build's: rendered again, never appended to
        (run / f"{name}.csv").unlink(missing_ok=True)
    with CsvWriter(run) as writer:
        writer.write_all(rendered)
    restatus(run / "promotion.csv", state.day)
    meta = opening.meta(world, stories, state.day)
    meta.save(run)
    if loop.warm_up_end is not None:
        loop.warm_up_end.save(run)
    Checkpoint.of(state).save(run, keep=[first - timedelta(days=1)])
    return meta


def status(valid_from: date, valid_to: date, day: date) -> str:
    """A promotion's status on `day` (`setup.promotions.status`'s rule, on its row)."""
    if day < valid_from:
        return "approved"
    return "running" if day <= valid_to else "closed"


def restatus(path: Path, day: date) -> None:
    """`path`'s promotions with their `status` on `day`, every other cell as written."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        rows = list(reader)
    at = {name: header.index(name) for name in ("valid_from", "valid_to", "status")}
    for cells in rows:
        cells[at["status"]] = status(
            date.fromisoformat(cells[at["valid_from"]]), date.fromisoformat(cells[at["valid_to"]]), day
        )
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    partial.replace(path)


def git_sha() -> tuple[str | None, bool]:
    """The checkout's commit and whether its tracked files differ from it; (None, False) outside a checkout."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, check=True, text=True)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_DIR,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return sha.stdout.strip(), bool(dirty.stdout.strip())


def _report(
    run: Path,
    meta: Meta,
    report: Report,
    timings: dict[str, float],
    null: bool,
    until: date,
    prolonged: bool = False,
) -> None:
    sha, dirty = git_sha()
    entry: dict[str, Any] = {
        "until": until.isoformat(),
        "null": null,
        "master_seed": meta.master_seed,
        "stories_sha256": meta.stories_sha256,
        "world_sha256": meta.world_sha256,
        "build_id": meta.build_id,
        "git_sha": sha,
        "git_dirty": dirty,
        "timings_s": {name: round(seconds, 1) for name, seconds in timings.items()},
        "quarters": {quarter: dict(sorted(counts.items())) for quarter, counts in report.quarters.items()},
    }
    path = run / BUILD_JSON
    held = json.loads(path.read_text(encoding="utf-8")) if prolonged and path.is_file() else None
    if held is not None:
        held.setdefault("prolongs", []).append(entry)
        entry = held
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
