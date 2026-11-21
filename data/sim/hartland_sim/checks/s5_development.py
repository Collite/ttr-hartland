"""S5's development gate (G2 S2.10; NH-D25 design09 §§5–6): the manifest-driven runner of the nine development worlds
— the evaluation's development seeds, each in the three scored modes — their source, the S2.9 judge, SQL/twin parity,
and the gate's GO or NO-GO. `python -m hartland_sim.checks.s5_development ROOT --db URL [--build | --build-only]`.

**The manifest** (`manifest`): `s5_evaluation.yaml`'s development seeds × `s5_decision.SCORED`, each world at
`ROOT/<seed>/<mode>/`. Refused before any work (`Refused`): development seeds not the protocol's or among either file's
certification seeds, a certification seed colliding with a seed built or read before the manifest (the protocol's
three lists and `PRIOR`), or a checkout with changes — the runner, the evaluator, the generator, the config and the
locks are the frozen head's, so it runs on that head, clean (`frozen`).

**Builds** (`--build`): each world without its `build.json` built in turn by `hartland-sim build-world --seed`, in a
process of its own (the build's seed guard, `evaluation.admit`, refuses a certification seed again); its command, exit,
wall time and peak memory go to `ROOT/builds.json`. A failed build stops the runner: nothing is judged. With
`--build-only` the runner stops after the builds, no world judged: the worlds are built while CI runs on the frozen
head, and no score is read before it is green.

**The source** (`source`, ⚑R380-A): a world's `build.json` names the frozen head as its `git_sha`, built clean and in
one run (no prolong); its `_sim_meta`, and `build.json`'s copies, carry the manifest's seed and mode, world.yaml's
sha256, and the `stories_sha256` and `build_id` this checkout gives that seed and mode (`expected`, `opening.meta`:
computed here, never read from a world). Every world's source is checked before any is judged: one fails, no score
is read.

**The judge** (`s5_decision.judge`, unchanged): its staleness, source and audit-manifest checks, then the claim's
verdict. The SQL alone cannot establish a complete export; the judge's source checks do.

**Parity** (`parity`, `--db`): each world's tables the twin reads (`s5_decision.TABLES`) copied into the DDL's schema
in a transaction that rolls back, analyzed before `sales_pl` is refreshed (#22 Found), both SQL files run; their
metric, `region_quality` and `round_quality` against the judge's: keys, counts and integrity exact, money within
0.01 CZK, q, Q and the metrics within 1e-9, NULL where the twin's is NaN (design09 §5). The two files' blocks agree.

**The gate** (`gate`, design09 §5): GO when the nine worlds and no other are there with their source; every story
PASSes (rank ≥ 0.8, Q_H ≥ 0.70, D ≥ 0.10, every region valid: integrity and support); every control is valid and
rejects — rank < 0.8 AND D < 0.10; an undefined or unsupported control has not rejected; and every world's parity
holds. Anything else is NO-GO, every reason given; no world is dropped, retried or replaced. The readout,
`ROOT/s5-development.json` and `.md`: the decision, each world's identity, source, claim status, metrics, regions and
24 rounds, parity, and its certification: NOT_CERTIFIED — development data, never S5's certification (S2.11).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from hartland_sim.checks import s5_decision, s5_monthly
from hartland_sim.checks.s5_monthly import BASE_REVISION, MONTHLY_FILE, MONTHLY_REVISION
from hartland_sim.config import DDL_DIR, STORIES_DIR, StoriesConfig, WorldConfig
from hartland_sim.evaluation import Evaluation
from hartland_sim.evaluation import seed as cert_seed
from hartland_sim.io.ddl import DOCWH, OWNER, READONLY
from hartland_sim.loop import pilot as pilots
from hartland_sim.loop.build import BUILD_JSON, git_sha
from hartland_sim.protocol import PROTOCOL_FILE, Protocol
from hartland_sim.setup import opening
from hartland_sim.setup.opening import REPO_DIR
from hartland_sim.world import META_FILE, Meta

READOUT, BUILDS = "s5-development", "builds.json"  # ROOT's readout (.json, .md) and build record
# seeds built or read before the manifest beside the protocol's lists: test_weather's master seed, and G2 S2.4's
# frozen prototype seed (`checks.s5_pooled.SEED`, branch feat/nh-g2-s2.4-s5-measure, never merged)
PRIOR = (1, 20261113)
MONEY = (
    "observed_margin",
    "asset_use_cost",
    "handout_cost",
    "programme_cost",
    "no_sales_cost",
    "fleet_cost",
    "idle_cost",
)
COUNTS = tuple(name for name in s5_decision.COLUMNS if name not in (*MONEY, "quality", "integrity"))
MONEY_SLACK, METRIC_SLACK = 0.01, 1e-9  # design09 §5: CZK, and q, Q and the metrics
LOCKS = ("uv.lock", "toolchain.lock")


class Refused(ValueError):
    """The runner's refusal before any work (module docstring): it prints it in one line."""


# ── the manifest and the source ──────────────────────────────────────────────


def manifest(held: Evaluation, protocol: Protocol) -> tuple[tuple[int, str], ...]:
    """The nine worlds, (seed, mode) in order (module docstring); `Refused` names the rule a seed breaks."""
    if held.revision == MONTHLY_REVISION:
        base = Evaluation.load()
        if replace(held, revision=base.revision, support=base.support, sha256=base.sha256) != base:
            raise Refused("monthly evaluation changes the registered NH-D25 window, bars, labels or seeds")
        if held.certification != tuple(cert_seed(i) for i in range(1, 11)):
            raise Refused("monthly evaluation changes design09's held-out seed derivation")
        support = {"chosen_per_month": 2, "unchosen_per_month": 2, "unchosen": 100}
        if held.support != support:
            raise Refused("monthly evaluation changes design10's support")
    if held.development != protocol.development:
        raise Refused(f"development seeds {held.development} are not the protocol's {protocol.development}")
    taken = sorted(set(held.development) & {*held.certification, *protocol.certification})
    if taken:
        raise Refused(f"development seeds {taken} are certification seeds: never a development world")
    before = {*protocol.development, *protocol.certification, *protocol.built_before, *PRIOR}
    collide = sorted(set(held.certification) & before)
    if collide or len(set(held.certification)) != len(held.certification):
        raise Refused(f"certification seeds {collide or 'twice'} collide with seeds built or read before the manifest")
    return tuple((seed, mode) for seed in held.development for mode in s5_decision.SCORED)


def frozen() -> str:
    """The checkout's head, which the runner and every world must be (module docstring); `Refused` with changes."""
    sha, dirty = git_sha()
    if sha is None:
        raise Refused("no git checkout: the frozen head is unknown")
    if dirty:
        raise Refused(f"the checkout at {sha[:12]} has changes: the frozen head runs clean")
    return sha


def expected(seed: int, mode: str, world: WorldConfig, stories: StoriesConfig, until: date) -> Meta:
    """The `_sim_meta` this checkout gives a build of `seed` in `mode` done to `until` (`opening.meta`): its seed,
    world.yaml's and the mode's stories' sha256 and its `build_id`, computed here, never read from a world."""
    return opening.meta(replace(world, master_seed=seed), pilots.stories_of(mode, stories), until, mode=mode)


def source(run: Path, seed: int, mode: str, head: str, want: Meta) -> list[str]:
    """Why `run` is not the frozen `head`'s build of `seed` in `mode` (module docstring); empty when it is."""
    if not (run / BUILD_JSON).is_file():
        return ["no build.json"]
    if not (run / META_FILE).is_file():
        return ["no _sim_meta.csv"]
    report = json.loads((run / BUILD_JSON).read_text(encoding="utf-8"))
    meta = Meta.load(run)
    out = []
    if report.get("git_sha") != head:
        out.append(f"built at {report.get('git_sha')}, not the frozen head {head}")
    if report.get("git_dirty") is not False:
        out.append("built from a checkout with changes")
    if "prolongs" in report:
        out.append("prolonged: not built in one run")
    for name, found, wanted in (
        ("master seed", meta.master_seed, seed),
        ("mode", meta.pilot_mode, mode),
        ("world_sha256", meta.world_sha256, want.world_sha256),
        ("stories_sha256", meta.stories_sha256, want.stories_sha256),
        ("build_id", meta.build_id, want.build_id),
        ("build.json master_seed", report.get("master_seed"), seed),
        ("build.json stories_sha256", report.get("stories_sha256"), want.stories_sha256),
        ("build.json build_id", report.get("build_id"), want.build_id),
    ):
        if found != wanted:
            out.append(f"{name} {found}, not {wanted}")
    return out


# ── parity ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Parity:
    """A world's SQL against its twin (`parity`): the differences (none: parity), the largest of each kind, the time."""

    differences: tuple[str, ...]
    money: float
    quality: float
    metric: float
    seconds: float


def schema(dsn: str) -> None:
    """The DDL's schema, fresh, with contracts §8's roles (as conftest's `hartland_db` and G5 S5.1's `load-world`)."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        for role in (OWNER, READONLY, DOCWH):
            conn.execute(
                f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') "
                f"THEN CREATE ROLE {role}; END IF; END $$"
            )
        conn.execute("DROP SCHEMA IF EXISTS hartland CASCADE")
        for name in ("schema.sql", "indexes.sql", "views/grants.sql"):
            conn.execute((DDL_DIR / name).read_text(encoding="utf-8"))


def load(conn: psycopg.Connection, run: Path) -> None:
    """`run`'s tables of `s5_decision.TABLES` copied in, each analyzed, then `sales_pl` refreshed (module docstring)."""
    for table in s5_decision.TABLES:
        path = run / f"{table}.csv"
        with path.open(encoding="utf-8") as source_file:
            header = source_file.readline().strip().split(",")
        statement = sql.SQL("COPY hartland.{} ({}) FROM STDIN WITH (FORMAT csv, HEADER true)").format(
            sql.Identifier(table), sql.SQL(", ").join(sql.Identifier(name) for name in header)
        )
        with conn.cursor().copy(statement) as copy, path.open("rb") as rows:
            while chunk := rows.read(1 << 20):
                copy.write(chunk)
        conn.execute(sql.SQL("ANALYZE hartland.{}").format(sql.Identifier(table)))
    conn.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")


def parity(
    dsn: str, run: Path, judged: s5_decision.Judged, monthly: tuple[s5_monthly.Case, ...] | None = None
) -> Parity:
    """`run` loaded in a transaction that rolls back, and its SQL against `judged`'s frames (module docstring)."""
    started = time.perf_counter()
    with psycopg.connect(dsn, autocommit=True) as conn, conn.transaction(force_rollback=True):
        conn.execute("SET LOCAL search_path TO hartland")
        conn.execute("SET LOCAL work_mem = '256MB'")
        load(conn, run)
        found = compare(conn, judged, monthly)
    return replace(found, seconds=round(time.perf_counter() - started, 1))


def _sql_parts(name: str) -> tuple[str, str]:
    text = (STORIES_DIR / name).read_text(encoding="utf-8")
    final = re.search(r"^SELECT", text, re.M)  # the final SELECT starts its line; the CTEs' bodies are indented
    if final is None:
        raise ValueError(f"{name} has no final SELECT")
    return text[: final.start()], text[final.start() :]


def monthly_sql() -> str:
    base, _ = _sql_parts(s5_decision.SQL[0])
    monthly = (STORIES_DIR / "pilot/s5_monthly.sql").read_text(encoding="utf-8")
    return base.replace("\nWITH ", "\nWITH RECURSIVE ", 1) + ",\n" + monthly


def _sql(conn: psycopg.Connection, name: str) -> tuple[Any, dict[str, dict[str, Any]], dict[tuple[str, date], Any]]:
    """The file's metric, its `region_quality` rows by region and its `round_quality` rows by (region, day)."""
    base, final = _sql_parts(name)
    row = conn.execute(base + final).fetchone()
    metric = row[0] if row is not None else None
    out = []
    for relation in ("region_quality", "round_quality"):
        cursor = conn.execute(base + f"SELECT * FROM {relation}")
        names = [column.name for column in cursor.description or ()]
        out.append([dict(zip(names, values, strict=True)) for values in cursor.fetchall()])
    return metric, {r["region"]: r for r in out[0]}, {(r["region"], r["decision_day"]): r for r in out[1]}


def _number(value: Any) -> float:
    return float(Decimal(value)) if isinstance(value, Decimal) else float(value)


def compare(
    conn: psycopg.Connection, judged: s5_decision.Judged, monthly: tuple[s5_monthly.Case, ...] | None = None
) -> Parity:
    """Both SQL files on the loaded rows against `judged`'s frames and metrics (module docstring)."""
    frame, rounds, v = judged.frame, judged.rounds, judged.verdict
    if frame is None or rounds is None or v is None:
        return Parity((f"no twin frames: {judged.status}",), math.nan, math.nan, math.nan, 0.0)
    out: list[str] = []
    worst = {"money": 0.0, "quality": 0.0, "metric": 0.0}

    def close(what: str, kind: str, found: Any, twin: float, slack: float) -> None:
        if found is None or math.isnan(twin):
            if (found is None) != math.isnan(twin):
                out.append(f"{what}: SQL {found}, twin {twin}")
            return
        gap = abs(_number(found) - twin)
        worst[kind] = max(worst[kind], gap)
        if gap > slack:
            out.append(f"{what}: SQL {found}, twin {twin}")

    if monthly is None:
        (rank, regions, by_round), (difference, again, by_round_again) = (_sql(conn, n) for n in s5_decision.SQL)
        if (regions, by_round) != (again, by_round_again):
            out.append("the two files' region and round frames differ")
        close("rank", "metric", rank, v.rank, METRIC_SLACK)
        close("difference", "metric", difference, v.difference, METRIC_SLACK)
    else:
        result = conn.execute(monthly_sql(), (judged.mode,)).fetchone()
        assert result is not None
        region_rows, round_rows, case_rows, inconclusive = result
        regions = {r["region"]: r for r in region_rows or []}
        by_round = {(r["region"], date.fromisoformat(r["decision_day"])): r for r in round_rows or []}
        if len(by_round) != len(round_rows or []) or len(regions) != len(region_rows or []):
            out.append("SQL returned duplicate region or round keys")
        found_cases = {tuple(r["omitted"]): r for r in case_rows or []}
        wanted_cases = {tuple(d.isoformat() for d in case.omitted): case for case in monthly}
        if len(found_cases) != len(case_rows or []) or set(found_cases) != set(wanted_cases):
            out.append("SQL and twin omission dates differ")
        for key in found_cases.keys() & wanted_cases.keys():
            found, wanted = found_cases[key], wanted_cases[key]
            what = ", ".join(key) or "full window"
            for name in ("status", "rank_holds", "q_high_holds", "difference_holds", "control_rejects"):
                if found[name] != getattr(wanted.verdict, name):
                    out.append(f"{what} {name}: SQL {found[name]}, twin {getattr(wanted.verdict, name)}")
            if found["failure"] != wanted.failure:
                out.append(f"{what} failure: SQL {found['failure']}, twin {wanted.failure}")
            for name in ("rank", "q_high", "q_low", "difference"):
                close(f"{what} {name}", "metric", found[name], getattr(wanted.verdict, name), METRIC_SLACK)
            if found["regions"].keys() != wanted.regions.keys():
                out.append(f"{what} region keys differ")
            for region in found["regions"].keys() & wanted.regions.keys():
                close(f"{what} {region} Q", "quality", found["regions"][region], wanted.regions[region], METRIC_SLACK)
        if not monthly:
            if inconclusive["status"] != v.status or v.control_rejects:
                out.append("full-window inconclusive verdict differs")
            for name in ("rank", "q_high", "q_low", "difference"):
                close(f"full window {name}", "metric", inconclusive.get(name), getattr(v, name), METRIC_SLACK)
    if sorted(regions) != sorted(frame.index):
        out.append(f"regions: SQL {sorted(regions)}, twin {sorted(frame.index)}")
    for region, row in regions.items():
        if region not in frame.index:
            continue
        mine = frame.loc[region].to_dict()
        if row["integrity"] != mine["integrity"]:
            out.append(f"{region} integrity: SQL {row['integrity']}, twin {mine['integrity']}")
        out += [f"{region} {n}: SQL {row[n]}, twin {mine[n]}" for n in COUNTS if int(row[n]) != int(mine[n])]
        for name in MONEY:
            close(f"{region} {name}", "money", row[name], float(mine[name]), MONEY_SLACK)
        close(f"{region} quality", "quality", row["quality"], float(mine["quality"]), METRIC_SLACK)
    twins = {(r["region"], r["decision_day"].date()): r for r in rounds.reset_index().to_dict("records")}
    if set(by_round) != set(twins):
        out.append(f"rounds: SQL {len(by_round)}, twin {len(twins)}, {len(set(by_round) ^ set(twins))} apart")
    for (region, day), row in sorted(by_round.items()):
        if (region, day) not in twins:
            continue
        mine = twins[(region, day)]
        for name in ("decisions", "viable", "chosen", "unchosen"):
            if int(row[name]) != int(mine[name]):
                out.append(f"{region} {day} {name}: SQL {row[name]}, twin {mine[name]}")
        close(f"{region} {day} q", "quality", row["q"], float(mine["q"]), METRIC_SLACK)
    return Parity(tuple(out), worst["money"], worst["quality"], worst["metric"], 0.0)


# ── the gate ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class World:
    """One world of the manifest: why its source fails (empty: the frozen head's), its judgement (None when no world
    was judged) and its parity (None when not run)."""

    seed: int
    mode: str
    run: Path
    source: tuple[str, ...]
    judged: s5_decision.Judged | None
    parity: Parity | None


def _metrics(v: s5_decision.Verdict | None) -> str:
    if v is None:
        return "no metrics"
    return f"rank {v.rank:.4f}, Q_H {v.q_high:.4f}, D {v.difference:.4f}"


def failures(world: World) -> list[str]:
    """Why `world` fails the gate (module docstring); empty when it holds."""
    name = f"{world.seed}/{world.mode}"
    if world.source:
        return [f"{name}: {why}" for why in world.source]
    judged = world.judged
    if judged is None:
        return [f"{name}: not judged"]
    v = judged.verdict
    out = []
    if world.mode == "story" and judged.status != "PASS":
        out.append(f"{name}: story {judged.status} ({_metrics(v)}){_why(judged)}")
    if world.mode != "story" and not (judged.status == "REJECTED" and v is not None and v.control_rejects):
        out.append(f"{name}: control {judged.status}, no rejection ({_metrics(v)}){_why(judged)}")
    if world.parity is None:
        out.append(f"{name}: parity not run")
    else:
        out += [f"{name}: parity: {difference}" for difference in world.parity.differences]
    return out


def _why(judged: s5_decision.Judged) -> str:
    return f": {'; '.join(judged.reasons)}" if judged.reasons else ""


def gate(worlds: Sequence[World], wanted: Sequence[tuple[int, str]]) -> tuple[str, list[str]]:
    """GO or NO-GO for the manifest's `wanted` worlds and every reason (module docstring)."""
    found = [(world.seed, world.mode) for world in worlds]
    out = [f"{seed}/{mode}: missing" for seed, mode in wanted if (seed, mode) not in found]
    out += [f"{seed}/{mode}: not in the manifest" for seed, mode in found if (seed, mode) not in wanted]
    out += [f"{seed}/{mode}: twice" for seed, mode in sorted(set(found)) if found.count((seed, mode)) > 1]
    for world in worlds:
        out += failures(world)
    return ("GO" if not out else "NO-GO"), out


# ── the runner ───────────────────────────────────────────────────────────────


def command(seed: int, mode: str, run: Path) -> list[str]:
    """The build of one world (module docstring), to the end of the protocol's window."""
    return [sys.executable, "-m", "hartland_sim.cli", "build-world", "--csv", "--pilot", mode, "--seed", str(seed)] + [
        "--run",
        str(run),
    ]


def build_missing(root: Path, worlds: Sequence[tuple[int, str]]) -> list[dict[str, Any]]:
    """Each world without its `build.json` built in turn, its record added to `ROOT/builds.json` (module docstring);
    `Refused` when a build fails."""
    path = root / BUILDS
    held: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    for seed, mode in worlds:
        run = root / str(seed) / mode
        if (run / BUILD_JSON).is_file():
            continue
        args = command(seed, mode, run)
        started = time.perf_counter()
        child = subprocess.Popen(args, cwd=REPO_DIR)
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        held.append(
            {
                "seed": seed,
                "mode": mode,
                "command": " ".join(["hartland-sim", *args[3:]]),
                "exit": child.returncode,
                "seconds": round(time.perf_counter() - started, 1),
                "peak_mib": round(usage.ru_maxrss / 1024),  # Linux: KiB
            }
        )
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
        if child.returncode != 0:
            raise Refused(f"the build of {seed}/{mode} exited {child.returncode}: nothing judged")
    return held


def develop(
    root: Path,
    worlds: Sequence[tuple[int, str]],
    head: str,
    held: Evaluation,
    protocol: Protocol,
    dsn: str | None,
) -> list[World]:
    """Each world's source, then — when every source holds — its judgement and parity (module docstring)."""
    config, stories = WorldConfig.load(), StoriesConfig.load()
    runs = {(seed, mode): root / str(seed) / mode for seed, mode in worlds}
    checked = {
        (seed, mode): tuple(source(run, seed, mode, head, expected(seed, mode, config, stories, protocol.window_until)))
        for (seed, mode), run in runs.items()
    }
    if any(checked.values()):
        return [World(seed, mode, run, checked[(seed, mode)], None, None) for (seed, mode), run in runs.items()]
    if dsn is not None:
        schema(dsn)
    out = []
    for (seed, mode), run in runs.items():
        judged = s5_decision.judge(run, protocol, held)
        found = parity(dsn, run, judged) if dsn is not None and judged.frame is not None else None
        out.append(World(seed, mode, run, (), judged, found))
        print(f"s5-development: {seed}/{mode} {judged.status} ({_metrics(judged.verdict)})", flush=True)
    return out


# ── the readout ──────────────────────────────────────────────────────────────


def _plain(value: Any) -> Any:
    """A JSON value: NaN as null, numpy's scalars as Python's, a day as its ISO date."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp | date):
        return value.isoformat()[:10]
    return value


def _world(world: World) -> dict[str, Any]:
    judged, out = world.judged, {"seed": world.seed, "mode": world.mode, "source": list(world.source)}
    out["certification"] = "NOT_CERTIFIED"  # development data (module docstring)
    out["gate"] = failures(world)
    if (world.run / BUILD_JSON).is_file():
        report = json.loads((world.run / BUILD_JSON).read_text(encoding="utf-8"))
        out["audit"] = {name: {"rows": a["rows"], "sha256": a["sha256"]} for name, a in report.get("audit", {}).items()}
        out["timings_s"] = report.get("timings_s")
    if judged is None:
        return _plain(out)  # type: ignore[no-any-return]
    out |= {"claim": judged.status, "reasons": list(judged.reasons), "identity": judged.identity}
    if judged.verdict is not None:
        out["verdict"] = judged.verdict.__dict__
    if judged.frame is not None and judged.rounds is not None:
        out["regions"] = {region: row.to_dict() for region, row in judged.frame.iterrows()}
        by_region: dict[str, list[dict[Any, Any]]] = {}
        for row in judged.rounds.reset_index().to_dict("records"):
            by_region.setdefault(str(row.pop("region")), []).append(row)
        out["rounds"] = by_region
    if world.parity is not None:
        out["parity"] = world.parity.__dict__
    return _plain(out)  # type: ignore[no-any-return]


def readout(
    worlds: Sequence[World], decision: str, reasons: Sequence[str], head: str, held: Evaluation, root: Path
) -> dict[str, Any]:
    """The readout (module docstring): the decision and its reasons, the frozen identities, the builds, each world."""
    builds = root / BUILDS
    return {
        "stage": "G2 S2.10 development gate (NH-D25 design09 §§5–6)",
        "decision": decision,
        "reasons": list(reasons),
        "head": head,
        "evaluation": dict(held.identity()),
        "protocol_sha256": hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest(),
        "locks": {name: hashlib.sha256((REPO_DIR / name).read_bytes()).hexdigest() for name in LOCKS},
        "certification": "NOT_CERTIFIED",
        "builds": json.loads(builds.read_text(encoding="utf-8")) if builds.is_file() else [],
        "worlds": [_world(world) for world in worlds],
    }


def _f(value: float | None) -> str:
    return "–" if value is None or math.isnan(value) else f"{value:.4f}"


def markdown(worlds: Sequence[World], decision: str, reasons: Sequence[str], head: str, held: Evaluation) -> str:
    """The readout in short: the decision, a line a world, each region's Q; then each world's Czech readout."""
    lines = [
        f"# S5 development gate: {decision}",
        "",
        f"Head `{head}` · evaluation {held.revision} · NOT_CERTIFIED",
        "",
    ]
    lines += [f"- {reason}" for reason in reasons] + [""]
    lines += [
        "| seed | mode | claim | rank | Q_H | Q_L | D | control rejects | parity |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for world in worlds:
        j, p = world.judged, world.parity
        v = j.verdict if j is not None else None
        status = j.status if j is not None else "not judged"
        par = "–" if p is None else "PARITY" if not p.differences else f"{len(p.differences)} differences"
        lines.append(
            f"| {world.seed} | {world.mode} | {status} | {_f(v.rank if v else None)} | {_f(v.q_high if v else None)} | "
            f"{_f(v.q_low if v else None)} | {_f(v.difference if v else None)} | {v.control_rejects if v else '–'} | "
            f"{par} |"
        )
    judged = [w for w in worlds if w.judged is not None and w.judged.frame is not None]
    if judged:
        regions = list(s5_decision.SKILL)
        lines += ["", "| seed | mode | " + " | ".join(regions) + " |", "|---|---|" + "---|" * len(regions)]
        for world in judged:
            assert world.judged is not None and world.judged.frame is not None
            q = world.judged.frame["quality"].reindex(regions)
            lines.append(f"| {world.seed} | {world.mode} | " + " | ".join(_f(float(x)) for x in q) + " |")
    for world in judged:
        assert world.judged is not None
        lines += ["", f"## {world.seed}/{world.mode}", "", s5_decision.markdown(world.judged, held)]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hartland_sim.checks.s5_development", description="S5's development gate (G2 S2.10)"
    )
    parser.add_argument("root", type=Path, help="the worlds' folder: ROOT/<seed>/<mode>/")
    parser.add_argument("--evaluation", choices=(BASE_REVISION, MONTHLY_REVISION), default=BASE_REVISION)
    parser.add_argument("--db", metavar="URL", help="a Postgres 16 server for SQL/twin parity (without: NO-GO)")
    building = parser.add_mutually_exclusive_group()
    building.add_argument("--build", action="store_true", help="build the missing worlds first, in turn")
    building.add_argument("--build-only", action="store_true", help="build the missing worlds and stop: none judged")
    sealing = parser.add_mutually_exclusive_group()
    sealing.add_argument("--prepare-freeze", action="store_true", help="authenticate and print the NH-D26 receipt hash")
    sealing.add_argument("--freeze-sha256", help="NH-D26 freeze receipt SHA-256 published before scoring")
    parser.add_argument("--inputs", type=Path, help="NH-D26 input worlds; separate from the readout root")
    parser.add_argument("--input-manifest", type=Path, help="original SHA-256 inventory of NH-D26 input worlds")
    parser.add_argument("--input-sha256", help="original inventory SHA-256")
    parser.add_argument("--source-head", help="commit on which the NH-D26 input worlds were built")
    args = parser.parse_args(argv)
    if args.evaluation == MONTHLY_REVISION and args.build:
        parser.error("NH-D26 builds require --build-only before a separate freeze and scoring run")
    if args.evaluation != MONTHLY_REVISION and (
        args.prepare_freeze
        or args.freeze_sha256
        or args.inputs
        or args.input_manifest
        or args.input_sha256
        or args.source_head
    ):
        parser.error("freeze and reuse options require --evaluation NH-D26/v1")
    held = Evaluation.load(MONTHLY_FILE) if args.evaluation == MONTHLY_REVISION else Evaluation.load()
    protocol = Protocol.load()
    try:
        if held.revision != args.evaluation:
            raise Refused(f"selected {args.evaluation}, but manifest declares {held.revision}")
        worlds = manifest(held, protocol)
        head = frozen()
        if args.build or args.build_only:
            build_missing(args.root, worlds)
    except Refused as error:
        print(f"s5-development: {error}", file=sys.stderr)
        return 1
    if args.build_only:
        print(f"s5-development: the nine worlds built at {head[:12]}; none judged")
        return 0
    if held.revision == MONTHLY_REVISION:
        from hartland_sim.checks import s5_monthly_development as monthly

        if not all((args.inputs, args.input_manifest, args.input_sha256, args.source_head)):
            parser.error("NH-D26 needs --inputs, --input-manifest, --input-sha256 and --source-head")
        dsn = args.db or os.environ.get("HARTLAND_S5_PG")
        if not (args.prepare_freeze or (args.freeze_sha256 and dsn)):
            parser.error("NH-D26 needs --prepare-freeze or a published --freeze-sha256 and --db (or HARTLAND_S5_PG)")
        inputs, root = args.inputs.resolve(), args.root.resolve()
        if inputs.is_relative_to(root) or root.is_relative_to(inputs):
            parser.error("NH-D26 inputs and readout must be in separate directories")
        path = root / monthly.FREEZE
        try:
            receipt = monthly.freeze(
                inputs, args.input_manifest, args.input_sha256, args.source_head, head, worlds, held, protocol
            )
            if args.prepare_freeze:
                root.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise Refused(f"freeze already exists at {path}; keep it and use a new readout root")
                monthly.write_json(path, receipt)
                print(f"NH-D26 freeze {path}: {monthly.sha256(path)} (nothing scored)")
                return 0
            if monthly.sha256(path) != args.freeze_sha256 or json.loads(path.read_text(encoding="utf-8")) != receipt:
                raise Refused("freeze receipt differs from the published hash or authenticated inputs")
            assert dsn is not None
            decision = monthly.score(root, inputs, receipt, args.freeze_sha256, worlds, held, protocol, dsn)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"s5-development: {error}", file=sys.stderr)
            return 2
        print(f"s5-development: {decision} at {head[:12]}")
        return 0 if decision == "GO" else 1
    found = develop(args.root, worlds, head, held, protocol, args.db)
    decision, reasons = gate(found, worlds)
    out = readout(found, decision, reasons, head, held, args.root)
    (args.root / f"{READOUT}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.root / f"{READOUT}.md").write_text(markdown(found, decision, reasons, head, held), encoding="utf-8")
    print(f"s5-development: {decision} at {head[:12]}" + "".join(f"\n  - {reason}" for reason in reasons))
    return 0 if decision == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
