"""S5's pilot modes (G2 S2.6 T5, `loop.build`, `loop.pilot.stories_of`; design08 §8): the five frozen modes and their
stories, every mode on the story pre-pilot's map, the build's pilot metadata — S5 NOT_CERTIFIED, never a pass — a
prolong that keeps its mode and stops at the window's end, and a default build that knows no pilot. The builds run
to the assignment's effective day: setup, the PoS-free months, the first month-end's decisions."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from hartland_sim import evaluation
from hartland_sim.checks.stories import check, report
from hartland_sim.cli import main
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.loop.build import Built, PilotRefused, audit, build, prolong
from hartland_sim.loop.pilot import stories_of
from hartland_sim.protocol import META_KEYS, MODES, PROTOCOL_FILE, Protocol
from hartland_sim.setup import opening
from hartland_sim.setup.market import Market
from hartland_sim.world import Checkpoint, Meta

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
FIRST = date(2023, 3, 1)  # the assignment's day, its first placements' too


@pytest.fixture(scope="module")
def story(osm_market: Market, tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build(tmp_path_factory.mktemp("story"), FIRST, WORLD, STORIES, market=osm_market, pilot="story")


@pytest.fixture(scope="module")
def reference(osm_market: Market, tmp_path_factory: pytest.TempPathFactory) -> Built:
    return build(tmp_path_factory.mktemp("no-pos"), FIRST, WORLD, STORIES, market=osm_market, pilot="no-pos-null")


def test_each_mode_has_its_stories() -> None:
    assert Protocol.load().modes == MODES == ("story", "skill-neutral", "full-null", "no-pos", "no-pos-null")
    assert stories_of("story", STORIES) is STORIES and stories_of("no-pos", STORIES) is STORIES
    assert stories_of("full-null", STORIES) == stories_of("no-pos-null", STORIES) == STORIES.null()
    neutral = stories_of("skill-neutral", STORIES)
    s5, held = neutral.story("S5").parameters, STORIES.story("S5").parameters
    assert set(s5["pos_skill"].values()) == {1.0} and set(s5["pos_skill"]) == set(held["pos_skill"])
    assert s5["budget_factor"] == held["budget_factor"] == {"SMS": 1.5}  # generosity apart from skill
    assert {k: v for k, v in s5.items() if k != "pos_skill"} == {k: v for k, v in held.items() if k != "pos_skill"}
    assert [s for s in neutral.stories if s.story != "S5"] == [s for s in STORIES.stories if s.story != "S5"]
    assert neutral.sha256 not in (STORIES.sha256, STORIES.null().sha256)


def test_a_pilot_build_carries_its_metadata_and_s5_is_not_certified(story: Built) -> None:
    held = dict(Meta.load(story.run).pilot)
    assert held["pilot_mode"] == "story" and held["s5_status"] == "NOT_CERTIFIED"
    assert held["pilot_design"] == "NH-D21-B/v1" and held["pilot_term_days"] == "1461/8"
    assert held["pilot_window"] == "2024-01-01/2025-12-31" and len(held["assignment_digest"]) == 64
    world = Checkpoint.load(story.run, FIRST).restore()
    assert world.pilot is not None and held["assignment_digest"] == world.pilot.digest
    s5 = [v for v in check(story.run, STORIES) if v.story == "S5"]
    assert s5 and all((v.status, v.passed) == ("NOT_CERTIFIED", False) for v in s5)
    assert _rows(story.run / "pos_pilot_assignment.csv") == 8160 and _rows(story.run / "pos_placement_decision.csv")
    assert {key: held[key] for key in evaluation.META_KEYS} == dict(evaluation.identity())  # G2 S2.9


def test_a_pilot_build_writes_its_audit_manifest(story: Built) -> None:
    """G2 S2.9 T3 (design09 §3): `build.json` holds the audit tables' rows, file sha256 and decisions by day, as
    rendered — every decision, before any evaluation filter."""
    held = json.loads((story.run / "build.json").read_text(encoding="utf-8"))["audit"]
    assert held == audit(story.run)
    decided = held["pos_placement_decision"]
    assert decided["rows"] == _rows(story.run / "pos_placement_decision.csv") == sum(decided["days"].values())
    assert list(decided["days"]) == ["2023-02-28"] and held["pos_pilot_assignment"]["rows"] == 8160


def test_stories_check_counts_not_certified_apart_and_refuses_the_wrong_direction(
    story: Built, reference: Built, capsys: pytest.CaptureFixture[str]
) -> None:
    """review-295 N2: the tail counts S5's NOT_CERTIFIED as neither run nor passed; `--null` must match the mode."""
    tail = report(check(story.run, STORIES), story.run, False).splitlines()[-1]
    assert " 2 NOT_CERTIFIED;" in tail and tail.startswith(tuple(f"{n} of " for n in range(30)))
    capsys.readouterr()
    assert main(["stories-check", "--csv", str(story.run), "--null"]) == 1
    assert main(["stories-check", "--csv", str(reference.run)]) == 1
    assert capsys.readouterr().err.count("judges it in the wrong direction") == 2


def test_the_protocol_is_an_input_of_a_pilot_build_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """review-295 N3: a default build reads no protocol, so the file's bytes move a pilot's build id only."""
    day = date(2023, 1, 3)
    before = (opening.meta(WORLD, STORIES, day), opening.meta(WORLD, STORIES, day, mode="story"))
    edited = tmp_path / "s5_protocol.yaml"
    edited.write_bytes(PROTOCOL_FILE.read_bytes() + b"# another revision\n")
    monkeypatch.setattr("hartland_sim.setup.opening.PROTOCOL_FILE", edited)
    after = (opening.meta(WORLD, STORIES, day), opening.meta(WORLD, STORIES, day, mode="story"))
    assert after[0].build_id == before[0].build_id and after[1].build_id != before[1].build_id


def test_meta_reads_the_pilots_keys_only(story: Built, tmp_path: Path) -> None:
    """review-295 N7: an unknown `_sim_meta` key is not a pilot key."""
    held = Meta.load(story.run)
    assert [key for key, _ in held.pilot] == [*META_KEYS, *evaluation.META_KEYS]
    lines = (story.run / "_sim_meta.csv").read_text(encoding="utf-8") + "some_later_key,1\n"
    (tmp_path / "_sim_meta.csv").write_text(lines, encoding="utf-8")
    assert Meta.load(tmp_path).pilot == held.pilot


def test_another_mode_takes_the_story_map_with_its_own_baselines(story: Built, reference: Built) -> None:
    mine, theirs = (Checkpoint.load(b.run, FIRST).restore().pilot for b in (reference, story))
    assert mine is not None and theirs is not None and mine.mode == "no-pos-null"
    assert mine.digest == theirs.digest == dict(Meta.load(reference.run).pilot)["assignment_digest"]
    assert {c: a.mapped for c, a in mine.assigned.items()} == {c: a.mapped for c, a in theirs.assigned.items()}
    assert {c: a.baseline for c, a in mine.assigned.items()} != {c: a.baseline for c, a in theirs.assigned.items()}
    assert not mine.decisions and _rows(reference.run / "pos_loan.csv") == 0  # no PoS at all
    assert reference.meta.build_id != story.meta.build_id


def test_a_prolong_keeps_the_mode_and_stops_at_the_window(
    story: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*_: object) -> None:
        raise AssertionError("a refused pilot runs no setup")

    monkeypatch.setattr("hartland_sim.loop.build.setup", never)  # refused before any work (review-295 N6)
    with pytest.raises(PilotRefused, match="ends with its window"):
        prolong(story.run, date(2026, 1, 1), WORLD, STORIES)
    with pytest.raises(PilotRefused, match="ends with its window"):
        build(tmp_path / "long", date(2026, 1, 1), WORLD, STORIES, pilot="story")
    with pytest.raises(PilotRefused, match="full-null"):
        build(tmp_path / "null", FIRST, WORLD, STORIES, null=True, pilot="story")
    with pytest.raises(PilotRefused, match="no pilot mode 'bogus'"):
        build(tmp_path / "bogus", FIRST, WORLD, STORIES, pilot="bogus")
    monkeypatch.undo()
    copy = tmp_path / "copy"
    copy.mkdir()
    for path in story.run.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    later = prolong(copy, date(2023, 3, 2), WORLD, STORIES)
    assert later.meta.pilot_mode == "story" and later.meta.build_id == story.meta.build_id


def test_the_cli_refuses_a_pilot_with_null(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["build-world", "--csv", "--null", "--pilot", "story", "--run", str(tmp_path / "x")]) == 1
    assert "full-null" in capsys.readouterr().err


def test_a_failure_inside_a_build_keeps_its_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a pilot's refusal is one line and exit 1 (review-295 R1): any other `ValueError` propagates."""

    def broken(*_: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr("hartland_sim.loop.build.setup", broken)
    for pilot in ([], ["--pilot", "story"]):
        with pytest.raises(ValueError, match="boom"):
            main(["build-world", "--csv", "--until", "2023-03-01", *pilot, "--run", str(tmp_path / f"r{len(pilot)}")])


def _rows(path: Path) -> int:
    with path.open(encoding="utf-8") as source:
        return sum(1 for _ in source) - 1
