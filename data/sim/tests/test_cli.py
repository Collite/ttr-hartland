import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hartland_sim import __version__
from hartland_sim.cli import VERBS, Verb, build_parser, main, pending

# contracts §7 — the data verbs; `test` is the justfile's own, not the generator's.
CONTRACT_VERBS = {
    "reference-check",
    "osm-extract",
    "ddl",
    "build-world",
    "load-world",
    "prolong",
    "stories-check",
    "recon",
    "load-documents",
    "docwh-slice-check",
    "h-l-sql",
    "h-l-run",
}

# The smallest valid invocation of each verb (the positional arguments it requires).
INVOCATIONS = {name: [name] for name in VERBS} | {
    "prolong": ["prolong", "2026-12-31"],
    "h-l-run": ["h-l-run", "bp-dsk"],
}


def run_console_script(*args: str) -> subprocess.CompletedProcess[str]:
    script = Path(sys.executable).parent / "hartland-sim"
    return subprocess.run([str(script), *args], capture_output=True, text=True, check=False)


def test_the_cli_wires_exactly_the_contract_verbs() -> None:
    assert set(VERBS) == CONTRACT_VERBS


def test_help_lists_every_verb() -> None:
    result = run_console_script("--help")
    assert result.returncode == 0, result.stderr
    for verb in CONTRACT_VERBS:
        assert re.search(rf"^\s+{re.escape(verb)}\s", result.stdout, re.M), verb


def test_version() -> None:
    result = run_console_script("--version")
    assert result.stdout.strip() == f"hartland-sim {__version__}"


def test_building_the_parser_imports_no_verbs_machinery() -> None:
    """`--version` and `--help` stay light: a verb's heavy imports (the extractor, numpy) wait for its handler."""
    probe = (
        "import json, sys; from hartland_sim.cli import build_parser; build_parser(); print(json.dumps([*sys.modules]))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    modules = set(json.loads(result.stdout))
    assert not modules & {"numpy", "hartland_sim.io.osm_extract"}


PENDING = sorted(name for name, verb in VERBS.items() if verb.handler is pending)


def test_the_landed_verbs_are_no_longer_pending() -> None:
    landed = {"reference-check", "osm-extract", "ddl", "stories-check", "build-world", "prolong"}
    assert set(VERBS) - set(PENDING) == landed


@pytest.mark.parametrize(("argv", "stage"), [(["build-world"], "G5 S5.1"), (["prolong", "2026-12-31"], "G5 S5.3")])
def test_build_world_and_prolong_are_csv_only_until_g5(
    argv: list[str], stage: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 1
    assert stage in capsys.readouterr().err


def test_build_world_takes_a_run_name_or_a_path() -> None:
    parser = build_parser()
    args = parser.parse_args(["build-world", "--until", "2024-12-31", "--csv", "--null", "--run", "r1"])
    assert (args.until.isoformat(), args.csv, args.null, args.run) == ("2024-12-31", True, True, "r1")
    from hartland_sim.cli import BUILD_DIR, _run_dir

    assert _run_dir("r1") == BUILD_DIR / "r1" and _run_dir("/tmp/x/r1") == Path("/tmp/x/r1")
    assert parser.parse_args(["prolong", "2024-05-09", "--csv", "r1"]).csv == "r1"


def test_build_world_counts_the_orders_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """review-221 N5: the warm-up's quarters order too, but write nothing."""
    from collections import Counter
    from datetime import date

    from hartland_sim.loop import build
    from hartland_sim.loop.day import Report
    from hartland_sim.world import Meta

    report = Report({"2023-Q4": Counter(orders=7), "2024-Q1": Counter(orders=5), "2024-Q2": Counter(orders=4)})
    meta = Meta(1, "s", "w", date(2024, 6, 30), "b" * 16)
    monkeypatch.setattr(build, "build", lambda run, *_, **__: build.Built(run, meta, report, {"total": 1.0}))
    assert main(["build-world", "--until", "2024-06-30", "--csv", "--run", str(tmp_path / "r")]) == 0
    assert "— 9 orders written, 2024-01-01 to 2024-06-30," in capsys.readouterr().out


@pytest.mark.parametrize("verb", PENDING)
def test_a_pending_verb_fails_naming_its_stage(verb: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(INVOCATIONS[verb]) == 1
    err = capsys.readouterr().err
    assert f"hartland-sim {verb}: not implemented yet" in err
    assert VERBS[verb].stage in err


def test_a_verb_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_:
        main([])
    assert exit_.value.code == 2
    assert "VERB" in capsys.readouterr().err


def test_prolong_takes_an_iso_day(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["prolong", "31.12.2026"])
    assert "invalid fromisoformat value" in capsys.readouterr().err


def test_stories_check_csv_is_a_flag_with_an_optional_run() -> None:
    parser = build_parser()
    dry = parser.parse_args(["stories-check", "--csv", "--dry"])
    assert (dry.csv, dry.dry) == ("latest", True)
    assert parser.parse_args(["stories-check", "--csv", "r1"]).csv == "r1"


@pytest.mark.parametrize("argv", [["stories-check"], ["stories-check", "--null"]])
def test_stories_check_runs_nothing_on_postgres_yet(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 1
    assert "running the detectors on Postgres arrives at G5 S5.2" in capsys.readouterr().err


def test_bare_csv_checks_the_build_with_the_newest_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checking the older build writes its `stories.md`, which moves its folder's time; the newer stays the latest."""
    import os

    from hartland_sim import cli

    monkeypatch.setattr(cli, "BUILD_DIR", tmp_path)
    for name, when in (("older", 1_000_000_000), ("newer", 1_000_100_000)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "weather_daily.csv").write_text("region_id,day,temp_mean_c\n", encoding="utf-8")
        os.utime(tmp_path / name / "weather_daily.csv", (when, when))
    (tmp_path / "empty").mkdir()  # no CSV: never a build
    main(["stories-check", "--csv", "older"])
    assert (tmp_path / "older" / "stories.md").is_file()
    capsys.readouterr()
    main(["stories-check", "--csv"])
    assert capsys.readouterr().out.startswith("# stories-check --csv newer\n")


def test_stories_check_csv_refuses_a_build_that_is_not_there(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stories-check", "--csv", "no-such-run-r1"]) == 1
    assert "no build" in capsys.readouterr().err


def test_a_not_implemented_error_inside_a_real_handler_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(args: object) -> int:
        raise NotImplementedError("boom")

    monkeypatch.setitem(VERBS, "boom", Verb("a verb whose body calls a pending stub", "nowhere", handler=boom))
    with pytest.raises(NotImplementedError, match="boom"):
        main(["boom"])
