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
    if report.problems:
        count = len(report.problems)
        print(f"hartland-sim reference-check: {count} problem{'s' * (count != 1)}", file=sys.stderr)
        return 1
    print(f"hartland-sim reference-check: {report.files} files, {report.rows} rows — OK")
    return 0


def _reference_check_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dir", type=Path, default=Path("data/reference"), help="the reference files (default: data/reference)"
    )


def _ddl_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--check", action="store_true", help="fail when data/ddl/ differs from the render (CI)")


def _build_world_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--until", type=date.fromisoformat, metavar="DAY", help="the last simulated day (ISO date)")
    parser.add_argument("--csv", action="store_true", help="write the CSV set to data/build/<run>/")
    parser.add_argument("--null", action="store_true", help="the null world: every story disabled (04 §9)")


def _load_world_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", metavar="URL", help="the target Postgres (default: the local one)")
    parser.add_argument("--csv-dir", metavar="DIR", help="the build to load (default: the latest data/build/<run>/)")


def _prolong_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("until", type=date.fromisoformat, metavar="DAY", help="simulate up to this day (ISO date)")


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
    "osm-extract": Verb("the Overpass queries → data/reference/osm-*.csv (+ the extract's meta)", "G0 S0.3"),
    "ddl": Verb("data/inventory.yaml → data/ddl/schema.sql + indexes.sql", "G0 S0.4", _ddl_arguments),
    "build-world": Verb("the simulation → data/build/<run>/", "G1 S1.4", _build_world_arguments),
    "load-world": Verb("COPY in load order · views · indexes · the _sim_meta guard", "G5 S5.1", _load_world_arguments),
    "prolong": Verb("from the last checkpoint: the world reacts, appends", "G5 S5.3", _prolong_arguments),
    "stories-check": Verb(
        "the answer key: every detector and probe", "G0 S0.4 (--dry) · G5 S5.2", _stories_check_arguments
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
