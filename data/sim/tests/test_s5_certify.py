"""The held-out gate's behavior on constructed exports and suite outcomes."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.checks import s5_certify as cert
from hartland_sim.checks import s5_development as dev
from hartland_sim.checks.s5_monthly import GROUP_FILE
from hartland_sim.checks.stories import check
from hartland_sim.config import Detector, StoriesConfig, WorldConfig
from hartland_sim.evaluation import Evaluation
from hartland_sim.protocol import Protocol
from hartland_sim.world import Meta

HELD = Evaluation.load(GROUP_FILE)
PROTOCOL = Protocol.load()
HEAD = "3" * 40
KEY = (31102026, "skill-neutral")


def edit_report(run: Path, **values: Any) -> None:
    path = run / "build.json"
    report = json.loads(path.read_bytes())
    path.write_text(json.dumps(report | values), encoding="utf-8")


@pytest.fixture(scope="module")
def configs() -> tuple[WorldConfig, StoriesConfig]:
    return WorldConfig.load(), StoriesConfig.load()


@pytest.fixture
def export(tmp_path: Path, decision_rows: Callable[..., Any], configs: tuple[WorldConfig, StoriesConfig]) -> Path:
    run = tmp_path / "world"
    case = decision_rows()
    want = dev.expected(*KEY, *configs, PROTOCOL.window_until)
    case.meta.update(
        dict(HELD.identity()),
        pilot_mode=KEY[1],
        stories_sha256=want.stories_sha256,
        build_id=want.build_id,
    )
    case.write(run)
    edit_report(run, git_sha=HEAD, stories_sha256=want.stories_sha256, build_id=want.build_id)
    return run


@pytest.mark.parametrize("problem", ["missing", "stale hash"])
def test_certification_rejects_missing_or_stale_audit_exports(
    problem: str, export: Path, configs: tuple[WorldConfig, StoriesConfig]
) -> None:
    path = export / "pos_placement_decision.csv"
    if problem == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes().replace(b",no_budget,", b",selected,", 1))
    assessed = cert.assess(export, KEY, HEAD, HELD, PROTOCOL, *configs)
    assert assessed.judged.status == "INCONCLUSIVE"
    assert assessed.cases == () and assessed.failures


def test_one_failed_seed_or_incomplete_suite_cannot_certify() -> None:
    keys = cert.manifest(HELD, PROTOCOL)
    assert len(keys) == 30
    assert len({seed for seed, _ in keys}) == 10
    passing = [(key, ()) for key in keys]
    assert cert.suite_status(passing, HELD, PROTOCOL) == "CERTIFIED"
    failure = (keys[-1], ("control did not reject",))
    assert cert.suite_status([*passing[:-1], failure], HELD, PROTOCOL) == "NOT_CERTIFIED"
    assert cert.suite_status(passing[:-1], HELD, PROTOCOL) == "NOT_CERTIFIED"
    assert cert.suite_status([*passing, passing[0]], HELD, PROTOCOL) == "NOT_CERTIFIED"


@pytest.mark.parametrize("changed", ["git_sha", "stories_sha256", "build_id", "s5_evaluation_sha256"])
def test_certification_rejects_wrong_source_or_evaluator(
    changed: str, export: Path, configs: tuple[WorldConfig, StoriesConfig]
) -> None:
    if changed.startswith("s5_"):
        path = export / "_sim_meta.csv"
        text = path.read_text(encoding="utf-8").replace(HELD.sha256, "wrong")
        path.write_text(text, encoding="utf-8")
    else:
        edit_report(export, **{changed: "wrong"})
    assessed = cert.assess(export, KEY, HEAD, HELD, PROTOCOL, *configs)
    assert assessed.judged.status == "NOT_CERTIFIED"
    assert assessed.cases == () and assessed.failures


def test_short_smoke_cannot_certify(export: Path, configs: tuple[WorldConfig, StoriesConfig]) -> None:
    path = export / "_sim_meta.csv"
    text = path.read_text(encoding="utf-8").replace("last_day_done,2025-12-31", "last_day_done,2024-03-31")
    assert text != path.read_text(encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    assessed = cert.assess(export, KEY, HEAD, HELD, PROTOCOL, *configs)
    assert assessed.judged.status == "NOT_CERTIFIED"
    assert assessed.cases == () and assessed.failures


def test_legacy_ratio_cannot_judge_the_group_gap() -> None:
    detector = Detector(cert.GROUP_SQL, cert.GAP_METRIC, 0.10, ">=", "<")
    cert.check_detector(detector, HELD)
    with pytest.raises(dev.Refused, match="group gap"):
        cert.check_detector(replace(detector, threshold=1.35), HELD)


def test_valid_control_rejection_is_a_successful_world(
    export: Path, configs: tuple[WorldConfig, StoriesConfig]
) -> None:
    assessed = cert.assess(export, KEY, HEAD, HELD, PROTOCOL, *configs)
    assert assessed.judged.status == "REJECTED"
    assert len(assessed.cases) == 301
    assert not assessed.failures


def test_ordinary_acceptance_reports_group_units_and_rejects_broken_sources(
    export: Path, configs: tuple[WorldConfig, StoriesConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cert, "git_sha", lambda: (HEAD, False))
    stories = configs[1]
    result = cert.check(export, stories.story("S5").detector, stories, Meta.load(export))
    assert result.passed is True
    assert "Q_H 100,00 %" in result.note and "D 0,00 procentních bodů" in result.note
    assert "NOT_ESTABLISHED" in result.note and "nevyužito" in result.detail
    (export / "pos_placement_decision.csv").unlink()
    (failed,) = [result for result in check(export, stories) if result.story == "S5"]
    assert failed.passed is False and "pos_placement_decision" in failed.note
