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
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 1
    if args.check:
        problems = drift(inventory, args.ddl_dir)
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            print("hartland-sim ddl --check: run `just ddl` and commit data/ddl/", file=sys.stderr)
            return 1
        print(f"hartland-sim ddl --check: data/ddl/ matches data/inventory.yaml ({len(inventory.tables)} tables)")
        return 0
    changed = write(inventory, args.ddl_dir)
    print(f"hartland-sim ddl: {', '.join(changed) or 'nothing'} {'written' if changed else 'changed'}")
    return 0


def _ddl_arguments(parser: argparse.ArgumentParser) -> None:
    from hartland_sim.io.ddl import DDL_DIR

    parser.add_argument("--check", action="store_true", help="fail when data/ddl/ differs from the render (CI)")
    parser.add_argument("--ddl-dir", type=Path, default=DDL_DIR, help="the DDL folder (default: the repo's data/ddl)")


def _build_world_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--until", type=date.fromisoformat, metavar="DAY", help="the last simulated day (ISO date)")
    parser.add_argument("--csv", action="store_true", help="write the CSV set to data/build/<run>/")
    parser.add_argument("--null", action="store_true", help="the null world: every story disabled (04 §9)")


def _load_world_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", metavar="URL", help="the target Postgres (default: the local one)")
    parser.add_argument("--csv-dir", metavar="DIR", help="the build to load (default: the latest data/build/<run>/)")


def _prolong_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("until", type=date.fromisoformat, metavar="DAY", help="simulate up to this day (ISO date)")


def stories_check(args: argparse.Namespace) -> int:
    """`--dry` lists the answer key; running the detectors and probes arrives with the runner (G5 S5.2)."""
    if not args.dry:
        print(
            "hartland-sim stories-check: not implemented yet — running the detectors arrives at G5 S5.2"
            " (--dry lists them)",
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


def _stories_check_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--null", action="store_true", help="expect the null world's opposite direction")
    parser.add_argument(
        "--csv",
        nargs="?",
        const="latest",
        metavar="RUN",
        help="read a CSV build (data/build/<run>/; bare --csv = the latest) instead of Postgres",
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
    "build-world": Verb("the simulation → data/build/<run>/", "G1 S1.4", _build_world_arguments),
    "load-world": Verb("COPY in load order · views · indexes · the _sim_meta guard", "G5 S5.1", _load_world_arguments),
    "prolong": Verb("from the last checkpoint: the world reacts, appends", "G5 S5.3", _prolong_arguments),
    "stories-check": Verb(
        "the answer key: every detector and probe",
        "G0 S0.4 (--dry) · G5 S5.2",
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
