"""`hartland-sim` — the data verbs of the repo's `just` surface (contracts §7).

Every verb is wired from the start (G0 S0.1) and implemented by the stage named in `VERBS`; until then it exits
non-zero naming that stage, so the `just` surface and CI exist before the bodies do. A stage that implements a verb
replaces its `pending` handler with the real one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from hartland_sim import __version__
from hartland_sim.io.reference import REFERENCE_DIR

Handler = Callable[[argparse.Namespace], int]


def pending(args: argparse.Namespace) -> int:
    """The body of a verb whose stage has not landed yet: says so, exits 1."""
    print(f"hartland-sim {args.verb}: not implemented yet — arrives at {VERBS[args.verb].stage}", file=sys.stderr)
    return 1


def _no_arguments(parser: argparse.ArgumentParser) -> None:
    del parser


def reference_check(args: argparse.Namespace) -> int:
    from hartland_sim.checks.reference import check

    report = check(args.dir)
    for problem in report.problems:
        print(problem, file=sys.stderr)
    if report.count:
        print(f"hartland-sim reference-check: {report.count} problem{'s' * (report.count != 1)}", file=sys.stderr)
        return 1
    print(f"hartland-sim reference-check: {report.files} files, {report.rows} rows — OK")
    return 0


def osm_extract(args: argparse.Namespace) -> int:
    from hartland_sim.io.osm_extract import run

    run(
        args.dir,
        endpoint=args.endpoint,
        only=args.only,
        pbf_dir=args.from_pbf,
        pbf_source=args.pbf_source,
        pbf_osm_base=args.pbf_osm_base,
    )
    return 0


def _osm_extract_arguments(parser: argparse.ArgumentParser) -> None:
    from hartland_sim.io.osm_names import ENDPOINT, OUTPUTS

    parser.add_argument("--endpoint", default=ENDPOINT, help=f"the Overpass API (default: {ENDPOINT})")
    parser.add_argument("--only", nargs="+", default=(), choices=OUTPUTS, help="rerun these only")
    parser.add_argument(
        "--from-pbf",
        type=Path,
        metavar="DIR",
        help="read scripts/osm_extract_pbf.sh's osmium output in DIR instead of asking Overpass",
    )
    parser.add_argument("--pbf-source", metavar="NAME", help="with --from-pbf: the PBF, for the meta")
    parser.add_argument("--pbf-osm-base", metavar="TIME", help="with --from-pbf: the PBF's timestamp, for the meta")
    parser.add_argument(
        "--dir", type=Path, default=REFERENCE_DIR, help="the reference files (default: the repo's data/reference)"
    )


def _reference_check_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dir", type=Path, default=REFERENCE_DIR, help="the reference files (default: the repo's data/reference)"
    )


def ddl(args: argparse.Namespace) -> int:
    from hartland_sim.config import ConfigError, InventoryConfig
    from hartland_sim.io.ddl import drift, write

    try:
        inventory = InventoryConfig.load()
        problems = drift(inventory, args.ddl_dir) if args.check else []
        changed = [] if args.check else write(inventory, args.ddl_dir)
    except ConfigError as error:  # the inventory, or a view file missing from --ddl-dir
        print(error, file=sys.stderr)
        return 1
    if args.check:
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            print("hartland-sim ddl --check: run `just ddl` and commit data/ddl/", file=sys.stderr)
            return 1
        print(f"hartland-sim ddl --check: data/ddl/ matches data/inventory.yaml ({len(inventory.tables)} tables)")
        return 0
    print(f"hartland-sim ddl: {', '.join(changed) or 'nothing'} {'written' if changed else 'changed'}")
    return 0


def _ddl_arguments(parser: argparse.ArgumentParser) -> None:
    from hartland_sim.config import DDL_DIR

    parser.add_argument("--check", action="store_true", help="fail when data/ddl/ differs from the render (CI)")
    parser.add_argument("--ddl-dir", type=Path, default=DDL_DIR, help="the DDL folder (default: the repo's data/ddl)")


def _run_dir(name: str) -> Path:
    """`data/build/<name>/`, or `name` itself when it is a path (holds a `/`)."""
    return Path(name) if "/" in name else BUILD_DIR / name


def build_world(args: argparse.Namespace) -> int:
    """The world built from the warm-up to `--until` into `data/build/<run>/` (`loop.build`); CSV only until G5 S5.1
    loads it into Postgres."""
    if not args.csv:
        print(
            "hartland-sim build-world: --csv only — loading a build into Postgres arrives at G5 S5.1", file=sys.stderr
        )
        return 1
    from hartland_sim.config import WorldConfig
    from hartland_sim.loop.build import build
    from hartland_sim.loop.day import quarter

    world = WorldConfig.load()
    until = args.until or world.history.until
    run = _run_dir(args.run or f"{'null' if args.null else 'world'}-{until.isoformat()}")
    try:
        built = build(run, until, world, null=args.null)
    except FileExistsError as error:
        print(f"hartland-sim build-world: {error} (another --run, or remove it)", file=sys.stderr)
        return 1
    first = world.history.first_day  # a quarter's first day: the quarters from its own are the written ones
    written = sum(q["orders"] for name, q in built.report.quarters.items() if name >= quarter(first))
    print(
        f"hartland-sim build-world: {run} — {written} orders written, {first} to {until},"
        f" in {built.timings['total']:.0f} s (build id {built.meta.build_id})"
    )
    return 0


def _build_world_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--until", type=date.fromisoformat, metavar="DAY", help="the last simulated day (ISO date)")
    parser.add_argument("--csv", action="store_true", help="write the CSV set to data/build/<run>/")
    parser.add_argument("--null", action="store_true", help="the null world: every story disabled (04 §9)")
    parser.add_argument(
        "--run", metavar="NAME", help="the folder under data/build/ (default: world-<until>), or a path"
    )


def _load_world_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", metavar="URL", help="the target Postgres (default: the local one)")
    parser.add_argument("--csv-dir", metavar="DIR", help="the build to load (default: the latest data/build/<run>/)")


def prolong(args: argparse.Namespace) -> int:
    """`--csv RUN`: a CSV build's loop resumed from its last checkpoint to DAY (`loop.build.prolong`, G1 S1.4 T5); the
    database's prolong arrives with G5 S5.3."""
    if not args.csv:
        print("hartland-sim prolong: --csv RUN only — prolonging the database arrives at G5 S5.3", file=sys.stderr)
        return 1
    from hartland_sim.loop.build import prolong as prolong_run

    run = _run_dir(args.csv)
    try:
        built = prolong_run(run, args.until)
    except (FileNotFoundError, ValueError) as error:
        print(f"hartland-sim prolong: {error}", file=sys.stderr)
        return 1
    if built.recovered:
        print(f"hartland-sim prolong: {run} held an unfinished prolong, cut back and run again from {built.recovered}")
    print(f"hartland-sim prolong: {run} to {args.until} in {built.timings['total']:.0f} s")
    return 0


def _prolong_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("until", type=date.fromisoformat, metavar="DAY", help="simulate up to this day (ISO date)")
    parser.add_argument("--csv", metavar="RUN", help="prolong the CSV build data/build/<run>/ (or a path)")


def stories_check(args: argparse.Namespace) -> int:
    """`--dry` lists the answer key; `--csv RUN` runs the pandas twins over a CSV build (G1 S1.2); the database
    runner arrives with G5 S5.2."""
    if args.csv and not args.dry:
        return _stories_check_csv(args)
    if not args.dry:
        print(
            "hartland-sim stories-check: not implemented yet — running the detectors on Postgres arrives at G5 S5.2"
            " (--csv RUN reads a CSV build, --dry lists them)",
            file=sys.stderr,
        )
        return 1
    from hartland_sim.config import ConfigError, StoriesConfig

    try:
        stories = StoriesConfig.load()
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 1
    for story in stories.stories:
        print(f"{story.story}  {story.title_cs}{'' if story.enabled else '  [vypnuto]'}")
        for detector in story.detectors:
            print(
                f"      {detector.sql:<28} {detector.metric} {detector.direction} {detector.threshold}"
                f" · null world {detector.null_world}"
            )
    for anomaly in stories.anomalies:
        print(f"{anomaly.id:<4} {anomaly.title_cs}{'' if anomaly.enabled else '  [vypnuto]'}")
        print(f"      {anomaly.probe.sql:<28} expect {anomaly.probe.expect}")
    reps = ", ".join(f"{pin.slot} {pin.profile}" for pin in stories.pins.reps)
    print(f"pins: {len(stories.pins.reps)} reps ({reps}); {len(stories.pins.customers)} customers by rule")
    detectors = sum(len(story.detectors) for story in stories.stories)
    print(
        f"hartland-sim stories-check --dry: {len(stories.stories)} stories, {detectors} detectors,"
        f" {len(stories.anomalies)} probes — nothing run (stories sha256 {stories.sha256[:12]})"
    )
    return 0


BUILD_DIR = REFERENCE_DIR.parent / "build"


def _stories_check_csv(args: argparse.Namespace) -> int:
    """The twins over `data/build/<RUN>/` (or a folder path; bare `--csv` = the latest build, the one whose newest CSV
    is the newest — a folder's own time moves when `stories.md` is first written into it): `stories.md` written beside
    the CSVs, the report printed; exit 1 when a judged check fails."""
    from hartland_sim.checks.stories import check, report
    from hartland_sim.config import ConfigError, StoriesConfig

    if args.csv == "latest":
        written = {p: [f.stat().st_mtime for f in p.glob("*.csv")] for p in BUILD_DIR.glob("*") if p.is_dir()}
        runs = sorted((p for p, times in written.items() if times), key=lambda p: max(written[p]))
        if not runs:
            print(f"hartland-sim stories-check: no build under {BUILD_DIR}", file=sys.stderr)
            return 1
        run = runs[-1]
    else:
        run = Path(args.csv) if Path(args.csv).is_dir() else BUILD_DIR / args.csv
    if not run.is_dir():
        print(f"hartland-sim stories-check: no build {run}", file=sys.stderr)
        return 1
    try:
        stories = StoriesConfig.load()
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 1
    verdicts = check(run, stories, null=args.null)
    text = report(verdicts, run, args.null)
    (run / "stories.md").write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if any(v.passed is False for v in verdicts) else 0


def _stories_check_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--null", action="store_true", help="expect the null world's opposite direction")
    parser.add_argument(
        "--csv",
        nargs="?",
        const="latest",
        metavar="RUN",
        help="read a CSV build (data/build/<run>/ or a folder; bare --csv = the latest) instead of Postgres",
    )
    parser.add_argument("--dry", action="store_true", help="list the detectors and probes, run nothing")


def _load_documents_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cluster", metavar="NAME", help="post to that cluster's DocWH (default: the local one)")


def _h_l_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("cluster", metavar="CLUSTER", help="the cluster whose Golem answers the corpus")


@dataclass(frozen=True)
class Verb:
    help: str
    stage: str
    arguments: Callable[[argparse.ArgumentParser], None] = _no_arguments
    handler: Handler = pending


VERBS: dict[str, Verb] = {
    "reference-check": Verb(
        "every data/reference/*.csv against its schema + row counts",
        "G0 S0.2",
        _reference_check_arguments,
        reference_check,
    ),
    "osm-extract": Verb(
        "the Overpass queries → data/reference/osm-*.csv (+ the extract's meta)",
        "G0 S0.3",
        _osm_extract_arguments,
        osm_extract,
    ),
    "ddl": Verb("data/inventory.yaml → data/ddl/schema.sql + indexes.sql", "G0 S0.4", _ddl_arguments, ddl),
    "build-world": Verb("the simulation → data/build/<run>/", "G1 S1.4", _build_world_arguments, build_world),
    "load-world": Verb("COPY in load order · views · indexes · the _sim_meta guard", "G5 S5.1", _load_world_arguments),
    "prolong": Verb(
        "from the last checkpoint: the world reacts, appends", "G1 S1.4 (--csv) · G5 S5.3", _prolong_arguments, prolong
    ),
    "stories-check": Verb(
        "the answer key: every detector and probe",
        "G0 S0.4 (--dry) · G1 S1.2 (--csv) · G5 S5.2",
        _stories_check_arguments,
        stories_check,
    ),
    "recon": Verb("the invariants + the R0/R1 diff", "G5 S5.1"),
    "load-documents": Verb(
        "data/documents/ → the DocWH, with the registry's metadata", "DOC.3", _load_documents_arguments
    ),
    "docwh-slice-check": Verb("the DX slice validates against run-set/docwh/schemas/", "DOC.3"),
    "h-l-sql": Verb("the twelve H-L questions as hand SQL, against the expected rows", "G5 S5.4"),
    "h-l-run": Verb("the H-L corpus against a cluster's Golem", "M5 S5.1", _h_l_run_arguments),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hartland-sim", description="The New Hartland generator (contracts §7).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="VERB")
    for name, verb in VERBS.items():
        sub = verbs.add_parser(name, help=f"{verb.help} [{verb.stage}]", description=verb.help)
        verb.arguments(sub)
        sub.set_defaults(handler=verb.handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
