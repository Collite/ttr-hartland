"""A build into `build/<run>/` (`cli build-world`, G1 S1.4 T4; contracts §2), and prolong's loop half (T5): setup, the
day loop from the warm-up to the last day, the state at the end.

- **At the start**, setup's tables (`setup.run`'s rows), all but the opening `stock_day`: it is dated the day before
  the warm-up, a stock the warm-up moves, and the stock days are G3's from the first written day.
- **Day by day**, the loop's rows through the warm-up switch (`loop.day`), under the build's master seed.
- **At the end**:
  - the state tables rendered from the `World` (`opening.state_rows`, ⚑R201-A), `discount_budget` from the first
    written quarter on (the warm-up's quarters stay in the `World`);
  - the calendar's promotions with their `status` on the last day (setup writes them as of `history.until`);
  - the checkpoint of the last day, beside the warm-up's last day's (and a prolong's first day's, below);
  - `_sim_meta`, its `last_day_done` the last day;
  - `build.json`: the seed, the stories' hash, the git sha, the timings, and what the run did by quarter (orders,
    grants, refusals and holds by reason); a pilot-mode build's also its audit tables' export manifest (`audit`, G2
    S2.9, design09 §3): each table's rows and the sha256 of its file as rendered, and the decisions by day.
- **The null world** (`null`): every story and planted case off (`StoriesConfig.null`), with a stories hash of its
  own; `world_sha256` and `build_id` hash the files on disk.
- **A pilot mode** (`pilot`, G2 S2.6, `loop.pilot`): one of the protocol's five — `story`, `skill-neutral`,
  `full-null` and the no-PoS references `no-pos`, `no-pos-null` — with its stories (`pilot.stories_of`), to the end of
  the pilot's window at the latest. Every mode but `story` first runs the story pre-pilot of the same seed, written
  nowhere (`story_map`), on the same OSM market, and takes its map. `build_id` names the mode; `_sim_meta` carries the
  pilot's metadata, S5 NOT_CERTIFIED. A mode the protocol lacks, a pilot with `null`, or one past the window is
  refused (`PilotRefused`) before any work.

**Prolong** (`prolong`): the loop resumed from a run's last checkpoint to a later day, its rows appended to the run's
files, then the same end — the state tables and the promotions rendered again, the checkpoint and `_sim_meta` moved.
A pilot-mode run keeps its mode (`_sim_meta.pilot_mode`) and stops at the end of the pilot's window.
- **The guard** (⚑R221-A, ⚑R221-C): the run's `_sim_meta.build_id` must be this checkout's — the seed, `world.yaml`,
  the stories (or their null world), `inventory.yaml`, the reference files, the package version and the generator's
  revision (`hartland_sim.REVISION`, `opening.meta`). A run of other inputs or of another generator is built again,
  never extended; `CsvWriter` also refuses a file under another header.
- **Resumable:** before its first row a prolong writes `_prolong.json` — the day it starts from and the size of each
  file it may append to (all but the rendered tables and the run's own files, `opening.RENDERED`/`OWN`). The end
  keeps that day's checkpoint, saves `_sim_meta`, then deletes the marker. A prolong that finds the marker left by one
  that stopped (an exception, Ctrl-C, the machine) first cuts each listed file back to its size, removes a table the
  stopped one began, and runs from the marker's day again, so no row is written twice; a marker older than
  `_sim_meta`'s last day was left by one that stopped after saving it, so it finished and the marker is only removed.
  A run therefore keeps the last
  two checkpoints of the loop — a prolong's first day and its last — beside the warm-up's; the next prolong's end
  drops the older.

The database's side (`load-world` appends, the world reacts) is G5 S5.3's.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import quarter_start
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import pilot as pilots
from hartland_sim.loop.day import Loop, Report
from hartland_sim.loop.warmup import Output
from hartland_sim.protocol import Protocol
from hartland_sim.rng import using_master_seed
from hartland_sim.setup import market as market_setup
from hartland_sim.setup import opening
from hartland_sim.setup import portfolio as portfolio_setup
from hartland_sim.setup.market import Market
from hartland_sim.setup.opening import OWN, RENDERED, REPO_DIR, STATE_TABLES, state_rows
from hartland_sim.setup.promotions import status_of
from hartland_sim.setup.rows import tables
from hartland_sim.setup.run import run as setup
from hartland_sim.world import Checkpoint, Meta

BUILD_JSON = "build.json"
PROLONG_MARK = "_prolong.json"


class PilotRefused(ValueError):
    """A pilot build or prolong its inputs refuse before any work (module docstring): the CLI prints it in one line."""


@dataclass(frozen=True)
class Built:
    run: Path
    meta: Meta
    report: Report
    timings: dict[str, float]
    recovered: date | None = None  # a prolong that cut back an unfinished one: the day it ran from again


def build(
    run: Path,
    until: date,
    world: WorldConfig | None = None,
    stories: StoriesConfig | None = None,
    null: bool = False,
    market: Market | None = None,
    pilot: str | None = None,
) -> Built:
    """Setup and the days from the warm-up to `until` into `run` (module docstring), in `pilot`'s mode when given;
    `FileExistsError` when `run` already holds a build, `PilotRefused` for a mode the protocol lacks, a pilot with
    `null` or past its window. `market` saves rebuilding the OSM market (tests)."""
    if run.exists() and any(run.glob("*.csv")):
        raise FileExistsError(f"{run} already holds a build")
    world = world or WorldConfig.load()
    stories = stories or StoriesConfig.load()
    protocol = Protocol.load() if pilot is not None else None
    if protocol is not None:
        if pilot not in protocol.modes:
            raise PilotRefused(f"no pilot mode {pilot!r}: the protocol's are {', '.join(protocol.modes)}")
        if null:
            raise PilotRefused("a pilot's mode names its stories: the null world is the full-null mode's")
        _within(protocol, until)
    if null:
        stories = stories.null()
    started = time.perf_counter()
    held = None
    if pilot is not None and protocol is not None:
        if pilot != "story":  # two setups, one market (it reads only the reference files)
            market = market or market_setup.build()
        mapped = None if pilot == "story" else story_map(world, stories, market, protocol)
        held = pilots.start(pilot, protocol, mapped)
        stories = pilots.stories_of(pilot, stories)
    made = setup(world, stories, market)
    made.world.pilot = held
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


def story_map(
    world: WorldConfig, stories: StoriesConfig, market: Market | None, protocol: Protocol
) -> dict[int, pilots.Mapped]:
    """The story pre-pilot's map (module docstring): setup and the PoS-free months of a story pilot of the same seed
    and stories, to its assignment, written nowhere."""
    made = setup(world, stories, market)
    held = made.world.pilot = pilots.start("story", protocol)
    with (
        tempfile.TemporaryDirectory() as scratch,
        using_master_seed(world.master_seed),
        CsvWriter(Path(scratch)) as out,
    ):
        loop = Loop.of(made.world, world, stories, made.portfolio.skus, Output(out, date.max))
        loop.until(protocol.pos_free_until)
    return {customer_id: assigned.mapped for customer_id, assigned in held.assigned.items()}


def _within(protocol: Protocol, until: date) -> None:
    if until > protocol.window_until:
        raise PilotRefused(
            f"the pilot ends with its window on {protocol.window_until}: a longer one is another protocol, not {until}"
        )


def prolong(run: Path, until: date, world: WorldConfig | None = None, stories: StoriesConfig | None = None) -> Built:
    """`run`'s loop from its last checkpoint to `until` (module docstring), an unfinished prolong's cut back first;
    `ValueError` when `run` is not a build of these inputs or is past `until`, `PilotRefused` (one) when it is a pilot
    that `until` outlasts."""
    world = world or WorldConfig.load()
    stories = stories or StoriesConfig.load()
    held = Meta.load(run)
    mode = held.pilot_mode
    if mode is not None:
        _within(Protocol.load(), until)
        stories = pilots.stories_of(mode, stories)
    null = held.stories_sha256 == stories.null().sha256
    if null:
        stories = stories.null()
    differs = _differs(held, opening.meta(world, stories, held.last_day_done, mode=mode))
    if differs:
        raise ValueError(f"{run} is a build of other inputs ({', '.join(differs)} differ); build it again")
    recovered = _recover(run, held.last_day_done)
    since = recovered or held.last_day_done
    if until <= since:
        raise ValueError(f"{run} is done to {since}, past {until}")
    started = time.perf_counter()
    state = Checkpoint.load(run, since).restore()
    skus = portfolio_setup.build(tables(), world, stories).skus
    timings: dict[str, float] = {}
    _mark(run, since)
    with using_master_seed(world.master_seed), CsvWriter(run) as writer:
        loop = Loop.of(state, world, stories, skus, Output.of(writer, world))
        mark = time.perf_counter()
        loop.until(until)
        timings["loop"] = time.perf_counter() - mark
    mark = time.perf_counter()
    meta = _end(run, loop, world, stories, since)
    timings["end"] = time.perf_counter() - mark
    timings["total"] = time.perf_counter() - started
    _report(run, meta, loop.report, timings, null, until, prolonged=True, recovered=recovered)
    return Built(run, meta, loop.report, timings, recovered)


def _differs(held: Meta, current: Meta) -> list[str]:
    """What `held`'s inputs differ from `current`'s in, by name; empty when their `build_id`s are equal."""
    if held.build_id == current.build_id:
        return []
    named = [
        ("the master seed", held.master_seed != current.master_seed),
        ("world.yaml", held.world_sha256 != current.world_sha256),
        ("the stories", held.stories_sha256 != current.stories_sha256),
    ]
    rest = "the inventory, reference files, version or generator revision"
    return [name for name, differ in named if differ] or [
        rest + (" or the pilot's protocol" if held.pilot_mode else "")
    ]


def _appended(path: Path) -> bool:
    """A file of `run` a prolong appends to: not one the end renders, not the run's own."""
    return path.stem not in RENDERED and path.name not in OWN


def _mark(run: Path, since: date) -> None:
    """`_prolong.json` before a prolong's first row (module docstring), written whole or not at all."""
    sizes = {path.name: path.stat().st_size for path in sorted(run.glob("*.csv")) if _appended(path)}
    path = run / PROLONG_MARK
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps({"from": since.isoformat(), "sizes": sizes}, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def _recover(run: Path, done: date) -> date | None:
    """An unfinished prolong's files cut back to their sizes before it, a table it began removed; the day it ran from.
    None when `run` holds no `_prolong.json`, or one older than `done` (`_sim_meta`'s last day: that prolong finished,
    and its marker is removed with nothing cut); `ValueError`, before any file is touched, when the marker is newer
    than `done` or a listed file is gone or shorter than its size."""
    path = run / PROLONG_MARK
    if not path.is_file():
        return None
    mark = json.loads(path.read_text(encoding="utf-8"))
    since = date.fromisoformat(mark["from"])
    if since < done:
        path.unlink()
        return None
    if since > done:
        raise ValueError(f"{run}'s unfinished prolong starts {since}, after its _sim_meta's {done}; build it again")
    sizes: dict[str, int] = mark["sizes"]
    files = {p.name: p for p in sorted(run.glob("*.csv")) if _appended(p)}
    lost = sorted(name for name, size in sizes.items() if name not in files or files[name].stat().st_size < size)
    if lost:
        raise ValueError(f"{run}'s unfinished prolong cannot be cut back ({', '.join(lost)}); build it again")
    for name, file in files.items():
        if name in sizes:
            os.truncate(file, sizes[name])
        else:
            file.unlink()
    return since


def _end(run: Path, loop: Loop, world: WorldConfig, stories: StoriesConfig, since: date | None = None) -> Meta:
    """The end of a build, or of a prolong from `since` (module docstring): the checkpoints, then `_sim_meta`, then the
    prolong's marker removed."""
    state = loop.world
    rendered = state_rows(state, world)
    first = world.history.first_day
    rendered["discount_budget"] = [r for r in rendered["discount_budget"] if r["quarter"] >= quarter_start(first)]
    for name in STATE_TABLES:  # a prolong's run holds the build's: rendered again, never appended to
        (run / f"{name}.csv").unlink(missing_ok=True)
    with CsvWriter(run) as writer:
        writer.write_all(rendered)
    restatus(run / "promotion.csv", state.day)
    keep = [first - timedelta(days=1), *([since] if since else [])]
    if loop.warm_up_end is not None:
        loop.warm_up_end.save(run, keep=keep)
    Checkpoint.of(state).save(run, keep=keep)
    meta = opening.meta(world, stories, state.day, state.pilot)
    meta.save(run)
    (run / PROLONG_MARK).unlink(missing_ok=True)
    return meta


def restatus(path: Path, day: date) -> None:
    """`path`'s promotions with their `status` on `day` (`setup.promotions.status_of`), every other cell as written."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        rows = list(reader)
    at = {name: header.index(name) for name in ("valid_from", "valid_to", "status")}
    for cells in rows:
        cells[at["status"]] = status_of(
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
    recovered: date | None = None,
) -> None:
    sha, dirty = git_sha()
    entry: dict[str, Any] = {
        "until": until.isoformat(),
        "null": null,
        "master_seed": meta.master_seed,
        "stories_sha256": meta.stories_sha256,
        "world_sha256": meta.world_sha256,
        "build_id": meta.build_id,
        **({"pilot": dict(meta.pilot)} if meta.pilot else {}),
        "git_sha": sha,
        "git_dirty": dirty,
        "timings_s": {name: round(seconds, 1) for name, seconds in timings.items()},
        **({"recovered_from": recovered.isoformat()} if recovered else {}),
        **({"audit": audit(run)} if meta.pilot else {}),
        "quarters": {
            quarter: {key: round(n, 1) if isinstance(n, float) else n for key, n in sorted(counts.items())}
            for quarter, counts in report.quarters.items()
        },
    }
    path = run / BUILD_JSON
    held = json.loads(path.read_text(encoding="utf-8")) if prolonged and path.is_file() else None
    if held is not None:
        held.setdefault("prolongs", []).append(entry)
        entry = held
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


AUDITED = ("pos_pilot_assignment", "pos_placement_decision")  # a pilot's audit tables, rendered at the end


def audit(run: Path) -> dict[str, dict[str, Any]]:
    """The export manifest of `run`'s audit tables as rendered (module docstring): each one's rows and its file's
    sha256, and the decisions' rows by `decision_day` — every row, before any evaluation filters one out."""
    out: dict[str, dict[str, Any]] = {}
    for name in AUDITED:
        path = run / f"{name}.csv"
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
        out[name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if name == "pos_placement_decision":
            days: dict[str, int] = {}
            for row in rows:
                days[row["decision_day"]] = days.get(row["decision_day"], 0) + 1
            out[name]["days"] = dict(sorted(days.items()))
    return out
