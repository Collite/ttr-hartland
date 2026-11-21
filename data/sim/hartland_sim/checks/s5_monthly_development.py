"""Freeze, authenticate and evaluate the nine NH-D26 development worlds."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from hartland_sim.checks import s5_decision, s5_monthly
from hartland_sim.checks import s5_development as dev
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.evaluation import Evaluation
from hartland_sim.protocol import PROTOCOL_FILE, Protocol
from hartland_sim.setup.opening import REPO_DIR
from hartland_sim.world import Meta

FREEZE = "freeze.json"
# Exhaustive generator equivalence check: these are the only changes admitted when reusing an earlier build.
EVALUATOR_PATHS = {
    "data/s5_evaluation_monthly.yaml",
    "data/sim/hartland_sim/checks/s5_development.py",
    "data/sim/hartland_sim/checks/s5_monthly.py",
    "data/sim/hartland_sim/checks/s5_monthly_development.py",
    "data/sim/tests/test_s5_development.py",
    "data/sim/tests/test_s5_monthly.py",
    "data/sim/tests/test_s5_monthly_development.py",
    "data/stories/pilot/s5_monthly.sql",
}


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_DIR, text=True).strip()


def equivalent(source_head: str, head: str) -> None:
    paths = set(_git("diff", "--name-only", "-z", "--no-renames", source_head, head, "--").split("\0")) - {""}
    if paths - EVALUATOR_PATHS:
        raise dev.Refused(f"generator equivalence not established: {sorted(paths - EVALUATOR_PATHS)}")


def authenticate(inputs: Path, manifest: Path, recorded_sha: str) -> int:
    """Authenticate the original receipt and its entire file inventory, including metadata."""
    if sha256(manifest) != recorded_sha:
        raise dev.Refused("input receipt does not match the recorded SHA-256")
    wanted: dict[Path, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path in wanted:
            raise dev.Refused(f"unexpected or duplicate receipt path {name}")
        wanted[path] = digest
    found: set[Path] = set()
    for path in inputs.rglob("*"):
        if path.is_symlink():
            raise dev.Refused(f"symbolic link in input inventory: {path.relative_to(inputs)}")
        if path.is_file():
            found.add(path.relative_to(inputs))
    if not wanted or found != set(wanted):
        raise dev.Refused(f"input inventory differs: {sorted(found ^ set(wanted))[:5]}")
    for path, digest in wanted.items():
        if sha256(inputs / path) != digest:
            raise dev.Refused(f"input hash differs: {path}")
    return len(wanted)


def freeze(
    inputs: Path,
    manifest: Path,
    input_sha: str,
    source_head: str,
    head: str,
    worlds: Sequence[tuple[int, str]],
    held: Evaluation,
    protocol: Protocol,
) -> dict[str, Any]:
    """Verify source identity for every world before the evaluator is allowed to see any scores."""
    resolved_head = _git("rev-parse", "--verify", "--end-of-options", f"{source_head}^{{commit}}")
    equivalent(resolved_head, head)
    count = authenticate(inputs, manifest, input_sha)
    config, stories, baseline = WorldConfig.load(), StoriesConfig.load(), Evaluation.load()
    expected = []
    for seed, mode in worlds:
        run = inputs / str(seed) / mode
        want = dev.expected(seed, mode, config, stories, protocol.window_until)
        why = dev.source(run, seed, mode, resolved_head, want) + s5_decision.stale(run, protocol, baseline)
        if (run / "_sim_meta.csv").is_file() and Meta.load(run).last_day_done != protocol.window_until:
            why.append("incorrect last day")
        if why:
            raise dev.Refused(f"unauthenticated {seed}/{mode}: {', '.join(why)}")
        expected.append({"seed": seed, "mode": mode, **dev._plain(asdict(want))})
    return {
        "head": head,
        "source_head": resolved_head,
        "evaluator": dict(held.identity()),
        "build_evaluation": dict(baseline.identity()),
        "locks": {name: sha256(REPO_DIR / name) for name in dev.LOCKS},
        "protocol_sha256": sha256(PROTOCOL_FILE),
        "inputs_sha256": input_sha,
        "input_count": count,
        "worlds": expected,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(".partial")
    with partial.open("w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    partial.replace(path)


@contextmanager
def scratch_database(dsn: str) -> Iterator[str]:
    """Each parity transaction gets its own local database, so rolled-back bulk loads cannot accumulate."""
    name = f"s5_dev_{uuid.uuid4().hex}"
    identifier = sql.Identifier(name)
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(identifier))
        try:
            database = make_conninfo(dsn, dbname=name)
            dev.schema(database)
            yield database
        finally:
            conn.execute(sql.SQL("DROP DATABASE {}").format(identifier))


def _world(
    inputs: Path, seed: int, mode: str, dsn: str, held: Evaluation, protocol: Protocol, baseline: Evaluation
) -> dict[str, Any]:
    run = inputs / str(seed) / mode
    judged = s5_monthly.judge(s5_decision.judge(run, protocol, baseline), held)
    cases = s5_monthly.cases(judged, held)
    if cases:
        judged = replace(judged, verdict=cases[0].verdict, status=cases[0].verdict.status)
    parity = None
    if judged.frame is not None:
        with scratch_database(dsn) as database:
            parity = dev.parity(database, run, judged, cases)
    found = dev._world(dev.World(seed, mode, run, (), judged, parity))
    summary = s5_monthly.assessment(cases)
    found["sensitivity"] = dev._plain(summary)
    found["cases"] = [dev._plain(asdict(case)) for case in cases]
    if summary["failure_type"] == "sensitivity":
        found["gate"].append(f"{seed}/{mode}: sensitivity failure in {len(summary['failures'])} omissions")
    return found


def score(
    root: Path,
    inputs: Path,
    receipt: dict[str, Any],
    receipt_sha: str,
    worlds: Sequence[tuple[int, str]],
    held: Evaluation,
    protocol: Protocol,
    dsn: str,
) -> str:
    baseline = Evaluation.load()
    reports = []
    checkpoints = root / "worlds"
    checkpoints.mkdir(exist_ok=True)
    for seed, mode in worlds:
        checkpoint = checkpoints / f"{seed}-{mode}.json"
        label = f"{seed}/{mode}"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            found = saved["world"]
            if saved["freeze_sha256"] != receipt_sha or (found["seed"], found["mode"]) != (seed, mode):
                raise dev.Refused(f"{label}: checkpoint belongs to a different freeze or world")
            print(f"{label}: resumed checkpoint", flush=True)
        else:
            print(f"{label}: starting audit and parity", flush=True)
            found = _world(inputs, seed, mode, dsn, held, protocol, baseline)
            write_json(checkpoint, {"freeze_sha256": receipt_sha, "world": found})
            print(f"{label}: {found['claim']}, {found['sensitivity']['failure_type'] or 'omissions hold'}", flush=True)
        reports.append(found)
    reasons = [reason for found in reports for reason in found["gate"]]
    decision = "NO-GO" if reasons else "GO"
    write_json(
        root / f"{dev.READOUT}.json",
        {
            "decision": decision,
            "certification": "NOT_CERTIFIED",
            "freeze_sha256": receipt_sha,
            "freeze": receipt,
            "reasons": reasons,
            "worlds": reports,
        },
    )
    report = markdown(receipt, receipt_sha, reports, decision, reasons)
    (root / f"{dev.READOUT}.md").write_text(report, encoding="utf-8")
    return decision


def markdown(receipt: dict[str, Any], sha: str, worlds: list[dict[str, Any]], decision: str, reasons: list[str]) -> str:
    lines = [
        f"# NH-D26/v1 development: {decision}",
        "",
        f"Executable `{receipt['head']}` · freeze SHA-256 `{sha}` · NOT_CERTIFIED",
        "",
        "| seed/mode | full verdict | rank | Q_H | D | omissions evaluated | failures | parity |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for world in worlds:
        verdict, sens = world.get("verdict", {}), world["sensitivity"]
        parity = world.get("parity")
        values = [dev._f(verdict.get(name)) for name in ("rank", "q_high", "difference")]
        lines.append(
            f"| {world['seed']}/{world['mode']} | {world['claim']} | "
            + " | ".join(values)
            + f" | {sens['count']} | {len(sens['failures'])} | "
            + ("–" if parity is None else "PASS" if not parity["differences"] else "FAIL")
            + " |"
        )
    lines += ["", *[f"- {reason}" for reason in reasons]]
    for world in worlds:
        sens = world["sensitivity"]
        lines += ["", f"## {world['seed']}/{world['mode']}", ""]
        lines += [f"- {reason}" for reason in world["reasons"]]
        if sens["full_failure"]:
            lines += [f"- Full-window failure: {sens['full_failure']}"]
        for name, (low, high) in sens.get("ranges", {}).items():
            lines += [f"- {name} range: {dev._f(low)} to {dev._f(high)}"]
        for failure in sens["failures"]:
            lines += [f"- Omit {', '.join(failure['omitted'])}: {failure['condition']}"]
    return "\n".join(lines) + "\n"
