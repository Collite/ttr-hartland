import re
import subprocess
import sys
from pathlib import Path

import pytest

from hartland_sim import __version__
from hartland_sim.cli import VERBS, Verb, main, pending

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


PENDING = sorted(name for name, verb in VERBS.items() if verb.handler is pending)


def test_the_landed_verbs_are_no_longer_pending() -> None:
    assert set(VERBS) - set(PENDING) == {"reference-check"}


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


@pytest.mark.parametrize("argv", [["stories-check", "--csv", "--dry"], ["stories-check", "--csv", "r1"]])
def test_stories_check_csv_is_a_flag_with_an_optional_run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 1
    assert "hartland-sim stories-check: not implemented yet — arrives at G0 S0.4" in capsys.readouterr().err


def test_a_not_implemented_error_inside_a_real_handler_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(args: object) -> int:
        raise NotImplementedError("boom")

    monkeypatch.setitem(VERBS, "boom", Verb("a verb whose body calls a pending stub", "nowhere", handler=boom))
    with pytest.raises(NotImplementedError, match="boom"):
        main(["boom"])
