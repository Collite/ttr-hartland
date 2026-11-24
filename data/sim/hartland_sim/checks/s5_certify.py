"""NH-D28 held-out certification of forecast-based placement quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg

from hartland_sim.checks import s5_decision, s5_monthly
from hartland_sim.checks import s5_development as dev
from hartland_sim.checks.s5_monthly_development import scratch_database, sha256, write_json
from hartland_sim.config import STORIES_DIR, Detector, StoriesConfig, WorldConfig
from hartland_sim.evaluation import Evaluation
from hartland_sim.loop.build import AUDITED, BUILD_JSON, build, git_sha
from hartland_sim.protocol import PROTOCOL_FILE, Protocol
from hartland_sim.setup.opening import REPO_DIR, inputs
from hartland_sim.world import META_FILE, Meta

if TYPE_CHECKING:
    from hartland_sim.checks.stories import Verdict

WorldKey = tuple[int, str]
GROUP_SQL = "detectors/s5_best_worst.sql"
GAP_METRIC = "placement_quality_group_gap"


def manifest(held: Evaluation, protocol: Protocol) -> tuple[WorldKey, ...]:
    """Validate the frozen protocol before admitting just its thirty held-out identities."""
    if held.revision != s5_monthly.GROUP_REVISION:
        raise dev.Refused("certification requires NH-D28/v1")
    dev.manifest(held, protocol)
    return tuple((seed, mode) for seed in held.certification for mode in s5_decision.SCORED)


def check_detector(detector: Detector, held: Evaluation) -> None:
    """Cross-check the canonical answer key against the frozen evaluator."""
    detector_rule = (detector.sql, detector.metric, detector.direction, detector.threshold, detector.null_world)
    if detector_rule != (GROUP_SQL, GAP_METRIC, ">=", 0.10, "<") or held.acceptance["difference"] != 0.10:
        raise dev.Refused("S5 requires the NH-D28 group gap and its 0.10 difference threshold")


@dataclass(frozen=True)
class Assessment:
    judged: s5_decision.Judged
    cases: tuple[s5_monthly.Case, ...]

    @property
    def failures(self) -> tuple[str, ...]:
        expected = {"story": "PASS", "skill-neutral": "REJECTED", "full-null": "REJECTED"}
        reasons = list(self.judged.reasons)
        if self.judged.status != expected.get(self.judged.mode or ""):
            reasons.append(f"full window: {self.judged.status}")
        if len(self.cases) != 301:
            reasons.append(f"{len(self.cases)} of 301 required date sets")
        for case in self.cases:
            if case.failure:
                label = " and ".join(day.isoformat() for day in case.omitted) or "full window"
                conditions = (case.failure,) if isinstance(case.failure, str) else case.failure
                reasons.append(f"{label}: {', '.join(conditions)}")
        return tuple(reasons)


def assess(
    run: Path,
    key: WorldKey,
    head: str,
    held: Evaluation,
    protocol: Protocol,
    world: WorldConfig,
    stories: StoriesConfig,
) -> Assessment:
    """Authenticate the source first, then use the same audited group judge as the ordinary acceptance path."""
    seed, mode = key
    wanted = dev.expected(seed, mode, world, stories, protocol.window_until)
    reasons = dev.source(run, seed, mode, head, wanted)
    if not reasons and Meta.load(run).last_day_done != protocol.window_until:
        reasons.append(f"build must end on {protocol.window_until}")
    if reasons:
        return Assessment(s5_decision.Judged(run, mode, "NOT_CERTIFIED", tuple(reasons), None, None, None, {}), ())
    base = s5_decision.judge(run, protocol, source_evaluation=held)
    return Assessment(*s5_monthly.evaluate(base, held))


def suite_status(results: Sequence[tuple[WorldKey, Sequence[str]]], held: Evaluation, protocol: Protocol) -> str:
    """Only a clean result for every registered seed and mode can pass the finite suite."""
    if tuple(key for key, _ in results) != manifest(held, protocol) or any(failures for _, failures in results):
        return "NOT_CERTIFIED"
    return "CERTIFIED"


def check(run: Path, detector: Detector, stories: StoriesConfig, meta: Meta | None) -> Verdict:
    """Use the source judge for the ordinary S5 acceptance report, including the higher-group guard."""
    from hartland_sim.checks.stories import Verdict

    held = Evaluation.load(s5_monthly.GROUP_FILE)
    check_detector(detector, held)
    protocol = Protocol.load()
    mode = meta.pilot_mode if meta else None
    direction = detector.null_world if mode in s5_decision.SCORED[1:] else detector.direction
    common = ("S5", detector.sql, detector.metric)
    if mode is None:
        audit_exists = any((run / f"{table}.csv").is_file() for table in AUDITED)
        return Verdict(
            *common,
            None,
            direction,
            detector.threshold,
            False if audit_exists else None,
            "Vyžaduje auditovaná data pilotního režimu",
            "NOT_CERTIFIED",
        )
    assert meta is not None
    if mode not in s5_decision.SCORED or meta.last_day_done < protocol.window_until:
        return Verdict(*common, None, direction, detector.threshold, False, "Vyžaduje úplné období", "NOT_CERTIFIED")
    head, dirty = git_sha()
    if head is None or dirty:
        return Verdict(*common, None, direction, detector.threshold, False, "Vyžaduje čistý zdroj", "NOT_CERTIFIED")
    assessed = assess(run, (meta.master_seed, mode), head, held, protocol, WorldConfig.load(), stories)
    judged = assessed.judged
    found = judged.verdict
    passed = not assessed.failures
    values = group_summary(found) if found is not None else ""
    notes = [values, *judged.reasons, "Kauzální návratnost: NOT_ESTABLISHED."]
    failed_omissions = [case for case in assessed.cases[1:] if case.failure]
    if assessed.cases:
        notes.append(f"Nesplněné výpustky měsíců: {len(failed_omissions)} z 300.")
        if mode == "story":
            guard = "splněn" if found and found.q_high_holds else "nesplněn"
            notes.append(f"Práh Q_H 70 % za celé období: {guard}.")
    detail = ""
    if judged.frame is not None:
        detail = s5_decision.markdown(judged, held, group_scores=values, overall_status="PASS" if passed else "FAIL")
        if failed_omissions:
            detail += _omission_details(failed_omissions, held, direction)
    return Verdict(
        *common,
        found.difference if found is not None else None,
        direction,
        detector.threshold,
        passed,
        " ".join(note for note in notes if note),
        "NOT_CERTIFIED" if judged.status == "NOT_CERTIFIED" else "",
        detail,
    )


def _omission_details(cases: Sequence[s5_monthly.Case], held: Evaluation, direction: str) -> str:
    high = f"{held.acceptance['q_high']:.2f}".replace(".", ",")
    gap = f"{held.acceptance['difference']:.2f}".replace(".", ",")
    thresholds = {"q_high": f"Q_H ≥ {high}", "difference": f"D {'≥' if direction == '>=' else '<'} {gap}"}
    lines = ["", "## Nesplněné výpustky měsíců", ""]
    for case in cases:
        dates = " a ".join(day.isoformat() for day in case.omitted)
        conditions = (case.failure,) if isinstance(case.failure, str) else case.failure or ()
        lines.append(f"- {dates} — {', '.join(thresholds[condition] for condition in conditions)}")
    return "\n".join(lines) + "\n"


def group_summary(found: s5_decision.Verdict) -> str:
    """Display the higher and lower means as percentages and their difference as percentage points."""
    from math import isfinite

    def percent(value: float) -> str:
        return f"{value * 100:.2f}".replace(".", ",") if isfinite(value) else "–"

    return (
        "Podle předpovědí dostupných před rozhodnutím. "
        f"Vyšší skupina Q_H {percent(found.q_high)} %, nižší skupina Q_L {percent(found.q_low)} %, "
        f"rozdíl D {percent(found.difference)} procentních bodů."
    )


def sql() -> str:
    return (STORIES_DIR / GROUP_SQL).read_text(encoding="utf-8")


@dataclass(frozen=True)
class Candidate:
    held: Evaluation
    protocol: Protocol
    world: WorldConfig
    stories: StoriesConfig
    receipt: dict[str, Any]

    @property
    def digest(self) -> str:
        data = json.dumps(self.receipt, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def current(cls, head: str) -> Candidate:
        actual = dev.frozen()
        if actual != head or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_DIR
        ):
            raise dev.Refused(f"expected a clean frozen checkout at {head}; found {actual}")
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_DIR, text=True).strip()
        held, protocol = Evaluation.load(s5_monthly.GROUP_FILE), Protocol.load()
        world, stories = WorldConfig.load(), StoriesConfig.load()
        check_detector(stories.story("S5").detector, held)
        worlds = manifest(held, protocol)
        if world.master_seed in held.certification:
            raise dev.Refused("held-out seed collides with the world's default")
        receipt = {
            "head": head,
            "tree": tree,
            "evaluator": dict(held.identity()),
            "canonical_sql_sha256": sha256(STORIES_DIR / GROUP_SQL),
            "protocol_sha256": sha256(PROTOCOL_FILE),
            "locks": {name: sha256(REPO_DIR / name) for name in dev.LOCKS},
            "generator_inputs": {str(path.relative_to(REPO_DIR)): sha256(path) for path in inputs()},
            "worlds": [[seed, mode, *source_ids(seed, mode, world, stories, protocol)] for seed, mode in worlds],
        }
        return cls(held, protocol, world, stories, receipt)


def source_ids(seed: int, mode: str, world: WorldConfig, stories: StoriesConfig, protocol: Protocol) -> tuple[str, str]:
    found = dev.expected(seed, mode, world, stories, protocol.window_until)
    return found.stories_sha256, found.build_id


def _world(root: Path, candidate: Candidate, key: WorldKey, dsn: str) -> int:
    """Build, authenticate and judge one world in a fresh process. Only this path admits a held-out seed."""
    if key not in manifest(candidate.held, candidate.protocol):
        raise dev.Refused("world is outside the held-out manifest")
    seed, mode = key
    world = replace(candidate.world, master_seed=seed)
    run = root / "inputs" / str(seed) / mode
    build(run, candidate.protocol.window_until, world, candidate.stories, pilot=mode)
    head = candidate.receipt["head"]
    assessed = assess(run, key, head, candidate.held, candidate.protocol, candidate.world, candidate.stories)
    judged = assessed.judged
    parity = None
    if judged.frame is not None:
        with scratch_database(dsn) as database:
            parity = dev.parity(
                database, run, judged, assessed.cases, revision=s5_monthly.GROUP_REVISION, statement=sql()
            )
    report = dev._world(dev.World(seed, mode, run, (), judged, parity), group=True)
    del report["gate"], report["certification"]
    if "verdict" in report:
        del report["verdict"]["rank"], report["verdict"]["rank_holds"]
    cases = [dev._plain(asdict(case)) for case in assessed.cases]
    for case in cases:
        del case["verdict"]["rank"], case["verdict"]["rank_holds"]
    failures = [*assessed.failures, *(parity.differences if parity else ("parity not run",))]
    files = (BUILD_JSON, META_FILE, *(f"{table}.csv" for table in s5_decision.TABLES))
    report.update(
        freeze_sha256=candidate.digest,
        failures=failures,
        inputs={name: sha256(run / name) for name in files if (run / name).is_file()},
        cases=cases,
    )
    write_json(root / f"{seed}-{mode}.json", report)
    if judged.verdict is not None:
        print(group_summary(judged.verdict), flush=True)
    print(f"{seed}/{mode}: {judged.status}; {len(cases)} date sets; {len(failures)} failures", flush=True)
    return int(bool(failures))


def run_seed(root: Path, index: int, candidate: Candidate) -> int:
    """Run the three modes sequentially and stop at the first required validation failure."""
    if not 1 <= index <= len(candidate.held.certification):
        raise dev.Refused("seed index outside the held-out manifest")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "freeze.json", candidate.receipt)
    with psycopg.connect(os.environ["HARTLAND_S5_PG"]) as conn:
        postgres_version = conn.info.server_version
    if postgres_version // 10_000 != 16:
        raise dev.Refused("certification requires PostgreSQL 16")
    seed = candidate.held.certification[index - 1]
    execution = []
    for mode in s5_decision.SCORED:
        command = [
            sys.executable,
            "-m",
            "hartland_sim.checks.s5_certify",
            "world",
            str(root),
            "--head",
            candidate.receipt["head"],
            "--freeze",
            candidate.digest,
            "--index",
            str(index),
            "--mode",
            mode,
        ]
        started = time.perf_counter()
        child = subprocess.Popen(command, cwd=REPO_DIR)
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        execution.append(
            {
                "seed": seed,
                "mode": mode,
                "command": shlex.join(command),
                "exit": child.returncode,
                "seconds": round(time.perf_counter() - started, 1),
                "peak_mib": round(usage.ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024), 1),
            }
        )
        write_json(
            root / "execution.json",
            {
                "freeze_sha256": candidate.digest,
                "python": sys.version.split()[0],
                "postgres_version": postgres_version,
                "worlds": execution,
            },
        )
        if child.returncode:
            return 1
    return 0


def verify(root: Path, candidate: Candidate) -> int:
    """Accept only the thirty successful world reports of this frozen CI execution."""
    reports = []
    expected_files = {BUILD_JSON, META_FILE, *(f"{table}.csv" for table in s5_decision.TABLES)}
    for index, seed in enumerate(candidate.held.certification, start=1):
        folder = root / f"seed-{index}"
        if not folder.is_dir():
            continue
        if json.loads((folder / "freeze.json").read_bytes()) != candidate.receipt:
            raise dev.Refused(f"seed {index} has the wrong candidate receipt")
        ledger = json.loads((folder / "execution.json").read_bytes())
        if ledger["freeze_sha256"] != candidate.digest or ledger["postgres_version"] // 10_000 != 16:
            raise dev.Refused(f"seed {index} has a stale execution ledger")
        for attempt in ledger["worlds"]:
            mode = attempt["mode"]
            if attempt["seed"] != seed or mode not in s5_decision.SCORED:
                raise dev.Refused(f"seed {index} has an unexpected world")
            path = folder / f"{seed}-{mode}.json"
            if not path.is_file():
                continue
            report = json.loads(path.read_bytes())
            wanted = dev.expected(seed, mode, candidate.world, candidate.stories, candidate.protocol.window_until)
            identity = report["identity"]
            if (
                report["freeze_sha256"] != candidate.digest
                or (report["seed"], report["mode"]) != (seed, mode)
                or identity.get("git_sha") != candidate.receipt["head"]
                or identity.get("git_dirty") is not False
                or identity.get("build_id") != wanted.build_id
                or identity.get("stories_sha256") != wanted.stories_sha256
                or identity.get("world_sha256") != wanted.world_sha256
                or identity.get("last_day_done") != wanted.last_day_done.isoformat()
                or identity.get("pilot_mode") != mode
                or identity.get("master_seed") != seed
                or any(identity.get(name) != value for name, value in candidate.held.identity())
                or set(report["inputs"]) != expected_files
            ):
                raise dev.Refused(f"{seed}/{mode} has stale or incomplete certification evidence")
            failures = list(report["failures"])
            if attempt["exit"] or report.get("parity", {}).get("differences") != [] or len(report["cases"]) != 301:
                failures.append("incomplete execution, parity or calendar evidence")
            reports.append(((seed, mode), failures))
    status = suite_status(reports, candidate.held, candidate.protocol)
    print(f"{status}: {len(reports)}/30 worlds against frozen head {candidate.receipt['head']}")
    print("Finite held-out decision-quality suite. Causal return: NOT_ESTABLISHED.")
    return int(status != "CERTIFIED")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "seed", "world", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--index", type=int)
    parser.add_argument("--mode", choices=s5_decision.SCORED)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        candidate = Candidate.current(args.head)
        if args.command == "prepare":
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / "freeze.json", candidate.receipt)
            print(candidate.digest)
            return 0
        if args.freeze != candidate.digest:
            raise dev.Refused("the current candidate does not match the published freeze hash")
        if args.command == "verify":
            return verify(root, candidate)
        if args.index is None or not 1 <= args.index <= len(candidate.held.certification):
            parser.error("seed and world require a registered --index from 1 to 10")
        if args.command == "seed":
            return run_seed(root, args.index, candidate)
        if args.mode is None:
            parser.error("world requires a registered --mode")
        return _world(
            root, candidate, (candidate.held.certification[args.index - 1], args.mode), os.environ["HARTLAND_S5_PG"]
        )
    except (dev.Refused, OSError, ValueError, KeyError) as error:
        print(f"s5-certify: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
