"""G2 S2.10 T1–T2 — S5's development gate (`checks.s5_development`; NH-D25 design09 §§5–6, ⚑R380-A) on conftest's
hand-built decisions: the manifest is the three development seeds in the three scored modes and refuses a
certification or colliding seed; a world's source is the frozen head's build of its seed and mode, its identities
computed from the checkout; and the gate — a valid story/control suite is GO; a missing or invalid region, one failed
story, a control whose rank OR group difference is too high, an undefined control, a wrong source or seed, a missing
world or a parity difference is NO-GO. Then the runner's refusals and its readout."""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.checks import s5_decision
from hartland_sim.checks import s5_development as dev
from hartland_sim.checks.s5_development import Parity, Refused, World
from hartland_sim.checks.s5_monthly import MONTHLY_FILE, MONTHLY_REVISION
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.evaluation import Evaluation
from hartland_sim.protocol import Protocol
from hartland_sim.world import Meta

DecisionRows = Callable[..., Any]  # conftest's `decision_rows`
REGIONS = ("PHA", "JMV", "SJC", "ZSC", "VCH", "SMS")
EVALUATION, PROTOCOL = Evaluation.load(), Protocol.load()
WORLDS = dev.manifest(EVALUATION, PROTOCOL)
HEAD = "f" * 40
LAST = date(2025, 12, 31)
HOLDS = Parity((), 0.0, 0.0, 0.0, 0.0)


def _cycled(best: dict[str, list[int]]) -> Callable[[str, int], list[tuple[int, str]]]:
    """Scores 1…10 a round, the five ending at the region's best of the round chosen: q = (best − 5) / 5."""

    def pool(region: str, k: int) -> list[tuple[int, str]]:
        b = best[region][k % len(best[region])]
        return [(s, "selected" if b - 4 <= s <= b else "no_budget") for s in range(1, 11)]

    return pool


POOLS = {
    # Q 1, 0.8, 0.6, 0.4, 0.2, 0: rank 0.99, Q_H 0.9, D 0.7 — a story PASS
    "graded": _cycled({r: [10 - i] for i, r in enumerate(REGIONS)}),
    # every Q 1: rank 0, D 0 — a control's rejection
    "flat": _cycled({r: [10] for r in REGIONS}),
    # Q 0.6, 0.55, 0.533, 0.517, 0.5, 0.5 in the skills' order: rank 1, D 0.07 — no PASS, no rejection
    "ranked": _cycled(
        {
            "PHA": [8],
            "JMV": [8] * 9 + [7] * 3,
            "SJC": [8] * 8 + [7] * 4,
            "ZSC": [8] * 7 + [7] * 5,
            "VCH": [8, 7],
            "SMS": [8, 7],
        }
    ),
    # Q 1, 0, 0.6, 0.2, 0.2, 0.2: rank 0.34, D 0.3 — no PASS, no rejection
    "apart": _cycled({"PHA": [10], "JMV": [5], "SJC": [8], "ZSC": [6], "VCH": [6], "SMS": [6]}),
    # SMS chooses three a round, 72 in all: short of its support — INCONCLUSIVE
    "unsupported": lambda region, k: [
        (s, "selected" if s >= (8 if region == "SMS" else 6) else "no_budget") for s in range(1, 11)
    ],
}


def _expected(seed: int, mode: str) -> Meta:
    return dev.expected(seed, mode, WorldConfig.load(), StoriesConfig.load(), LAST)


def _write(case: Any, run: Path, seed: int, mode: str) -> None:
    """`case` as the frozen head's build of `seed` in `mode`: its `_sim_meta` and `build.json` carry the identities
    the checkout gives them."""
    want = _expected(seed, mode)
    meta = {"master_seed": str(seed), "stories_sha256": want.stories_sha256, "build_id": want.build_id}
    dataclasses.replace(case, meta=case.meta | meta | {"pilot_mode": mode}).write(run)
    report = json.loads((run / "build.json").read_text(encoding="utf-8"))
    report |= {"git_sha": HEAD, "stories_sha256": want.stories_sha256, "build_id": want.build_id}
    (run / "build.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture(scope="module")
def suite(tmp_path_factory: pytest.TempPathFactory, decision_rows: DecisionRows) -> Path:
    """The nine worlds, each the frozen head's: the stories graded, the controls flat."""
    root = tmp_path_factory.mktemp("suite")
    cases = {"graded": decision_rows(POOLS["graded"]), "flat": decision_rows(POOLS["flat"])}
    for seed, mode in WORLDS:
        _write(cases["graded" if mode == "story" else "flat"], root / str(seed) / mode, seed, mode)
    return root


@pytest.fixture(scope="module")
def judged(tmp_path_factory: pytest.TempPathFactory, decision_rows: DecisionRows) -> dict[str, s5_decision.Judged]:
    """Each pool's judgement, as a story build of world.yaml's seed."""
    root = tmp_path_factory.mktemp("pools")
    out = {}
    for name, pool in POOLS.items():
        _write(decision_rows(pool), root / name, 31102026, "story")
        out[name] = s5_decision.judge(root / name)
    return out


def _copy(suite: Path, tmp_path: Path) -> Path:
    root = tmp_path / "suite"
    shutil.copytree(suite, root)
    return root


def _worlds(judged: dict[str, s5_decision.Judged], **pools: str) -> list[World]:
    """The nine worlds judged: the stories graded and the controls flat, but those named `<seed>_<mode>` (the mode's
    dash as an underscore), with parity holding."""
    out = []
    for seed, mode in WORLDS:
        name = pools.get(f"s{seed}_{mode.replace('-', '_')}", "graded" if mode == "story" else "flat")
        out.append(World(seed, mode, Path(f"{seed}/{mode}"), (), judged[name], HOLDS))
    return out


# ── the manifest ─────────────────────────────────────────────────────────────


def test_the_manifest_is_the_development_seeds_in_the_scored_modes() -> None:
    assert WORLDS == tuple(
        (seed, mode) for seed in (31102026, 1797428728, 285672490) for mode in ("story", "skill-neutral", "full-null")
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"development": (31102026, 1797428728, 895899358)}, "not the protocol's"),
        ({"certification": (*EVALUATION.certification[:9], 31102027)}, "collide"),  # the determinism tests'
        ({"certification": (*EVALUATION.certification[:9], 20261113)}, "collide"),  # G2 S2.4's prototype
        ({"certification": (*EVALUATION.certification[:9], 1206907222)}, "collide"),  # a retired one
        ({"certification": (*EVALUATION.certification[:9], EVALUATION.certification[0])}, "collide"),
    ],
    ids=["a certification seed developed", "the tests' world", "S2.4's seed", "a retired seed", "a seed twice"],
)
def test_the_manifest_refuses_a_certification_or_colliding_seed(change: dict[str, Any], reason: str) -> None:
    with pytest.raises(Refused, match=reason):
        dev.manifest(dataclasses.replace(EVALUATION, **change), PROTOCOL)


def test_a_development_seed_that_is_a_certification_seed_is_refused() -> None:
    """Both files moved to a certification seed: still no development world."""
    seeds = (31102026, 1797428728, EVALUATION.certification[4])
    with pytest.raises(Refused, match="certification seeds"):
        dev.manifest(
            dataclasses.replace(EVALUATION, development=seeds), dataclasses.replace(PROTOCOL, development=seeds)
        )
    seeds = (31102026, 1797428728, PROTOCOL.certification[0])
    with pytest.raises(Refused, match="certification seeds"):
        dev.manifest(
            dataclasses.replace(EVALUATION, development=seeds), dataclasses.replace(PROTOCOL, development=seeds)
        )


def test_monthly_manifest_preserves_seeds_and_is_explicit() -> None:
    monthly = Evaluation.load(MONTHLY_FILE)
    assert monthly.revision == MONTHLY_REVISION and monthly.sha256 != EVALUATION.sha256
    assert dev.manifest(monthly, PROTOCOL) == WORLDS
    assert monthly.certification == EVALUATION.certification
    with pytest.raises(Refused, match="registered"):
        dev.manifest(dataclasses.replace(monthly, development=monthly.certification[:3]), PROTOCOL)


# ── the source (⚑R380-A) ─────────────────────────────────────────────────────


def test_each_world_s_identity_is_its_seed_s_and_mode_s() -> None:
    """Nine build ids, computed from the checkout; the stories' sha256 the mode's."""
    ids = {_expected(seed, mode).build_id for seed, mode in WORLDS}
    assert len(ids) == 9
    stories = StoriesConfig.load()
    for mode, held in (("story", stories), ("skill-neutral", stories.skill_neutral()), ("full-null", stories.null())):
        assert _expected(285672490, mode).stories_sha256 == held.sha256
        assert _expected(285672490, mode).master_seed == 285672490


def _edit_meta(run: Path, key: str, value: str) -> None:
    path = run / "_sim_meta.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    out = [f"{key},{value}" if line.split(",", 1)[0] == key else line for line in lines]
    assert out != lines
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _edit_report(run: Path, **values: Any) -> None:
    report = json.loads((run / "build.json").read_text(encoding="utf-8"))
    (run / "build.json").write_text(json.dumps(report | values), encoding="utf-8")


ONE = Path("1797428728/skill-neutral")
SOURCES: dict[str, tuple[Callable[[Path], None], str]] = {
    "a dirty checkout": (lambda run: _edit_report(run, git_dirty=True), "with changes"),
    "another head": (lambda run: _edit_report(run, git_sha="e" * 40), "not the frozen head"),
    "unknown git": (lambda run: _edit_report(run, git_sha=None, git_dirty=False), "not the frozen head"),
    "prolonged": (lambda run: _edit_report(run, prolongs=[{"git_sha": HEAD}]), "prolonged"),
    "the story's stories": (
        lambda run: _edit_meta(run, "stories_sha256", _expected(1797428728, "story").stories_sha256),
        "stories_sha256",
    ),
    "the story's build id": (
        lambda run: _edit_meta(run, "build_id", _expected(1797428728, "story").build_id),
        "build_id",
    ),
    "another seed's build id": (
        lambda run: _edit_report(run, build_id=_expected(285672490, "skill-neutral").build_id),
        "build.json build_id",
    ),
    "another seed": (lambda run: _edit_meta(run, "master_seed", "285672490"), "master seed"),
    "another mode": (lambda run: _edit_meta(run, "pilot_mode", "full-null"), "mode"),
    "no build.json": (lambda run: (run / "build.json").unlink(), "no build.json"),
    "no world": (lambda run: shutil.rmtree(run), "no build.json"),
}


@pytest.mark.parametrize("name", list(SOURCES))
def test_a_wrong_source_or_seed_is_no_go_and_nothing_is_judged(suite: Path, tmp_path: Path, name: str) -> None:
    root = _copy(suite, tmp_path)
    spoil, reason = SOURCES[name]
    spoil(root / ONE)
    found = dev.develop(root, WORLDS, HEAD, EVALUATION, PROTOCOL, None)
    assert all(world.judged is None and world.parity is None for world in found)  # no score read
    decision, reasons = dev.gate(found, WORLDS)
    spoiled = [r for r in reasons if r.startswith("1797428728/skill-neutral: ")]
    assert decision == "NO-GO" and spoiled and any(reason in r for r in spoiled), reasons
    assert sorted(set(reasons) - set(spoiled)) == sorted(
        f"{seed}/{mode}: not judged" for seed, mode in WORLDS if (seed, mode) != (1797428728, "skill-neutral")
    )


def test_the_valid_suite_is_go_with_parity(suite: Path, tmp_path: Path) -> None:
    """End to end: every source holds, the stories PASS and the controls reject; without parity it is NO-GO."""
    found = dev.develop(_copy(suite, tmp_path), WORLDS, HEAD, EVALUATION, PROTOCOL, None)
    assert [w.source for w in found] == [()] * 9
    assert [w.judged.status for w in found if w.judged] == ["PASS", "REJECTED", "REJECTED"] * 3
    decision, reasons = dev.gate(found, WORLDS)
    assert decision == "NO-GO" and reasons == [f"{s}/{m}: parity not run" for s, m in WORLDS]
    assert dev.gate([dataclasses.replace(w, parity=HOLDS) for w in found], WORLDS) == ("GO", [])


# ── the gate ─────────────────────────────────────────────────────────────────


def test_the_pools_judge_as_designed(judged: dict[str, s5_decision.Judged]) -> None:
    v = {name: j.verdict for name, j in judged.items()}
    assert judged["graded"].status == "PASS" and v["graded"] and v["graded"].difference == pytest.approx(0.7)
    assert judged["flat"].status == "REJECTED" and v["flat"] and v["flat"].control_rejects
    ranked, apart = v["ranked"], v["apart"]
    assert ranked and ranked.rank == pytest.approx(1.0) and ranked.difference == pytest.approx(0.0694, abs=1e-4)
    assert apart and apart.rank == pytest.approx(0.339, abs=1e-3) and apart.difference == pytest.approx(0.3)
    assert judged["ranked"].status == judged["apart"].status == "REJECTED"
    assert not ranked.control_rejects and not apart.control_rejects
    assert judged["unsupported"].status == "INCONCLUSIVE" and judged["unsupported"].reasons == ("SMS: support",)


def test_a_valid_story_control_suite_is_go(judged: dict[str, s5_decision.Judged]) -> None:
    assert dev.gate(_worlds(judged), WORLDS) == ("GO", [])


@pytest.mark.parametrize(
    ("pools", "reason"),
    [
        ({"s285672490_story": "unsupported"}, "285672490/story: story INCONCLUSIVE"),  # an invalid region
        ({"s1797428728_story": "flat"}, "1797428728/story: story REJECTED"),  # one failed story
        ({"s31102026_story": "ranked"}, "31102026/story: story REJECTED"),
        ({"s31102026_skill_neutral": "ranked"}, "31102026/skill-neutral: control REJECTED, no rejection"),  # rank
        ({"s285672490_full_null": "apart"}, "285672490/full-null: control REJECTED, no rejection"),  # D
        ({"s1797428728_full_null": "graded"}, "1797428728/full-null: control PASS, no rejection"),
        ({"s1797428728_skill_neutral": "unsupported"}, "1797428728/skill-neutral: control INCONCLUSIVE"),
    ],
    ids=[
        "a region unsupported",
        "one story rejected",
        "a story short of Q_H",
        "a control's rank too high",
        "a control's difference too high",
        "a control passing",
        "an undefined control",
    ],
)
def test_one_failing_world_is_no_go(judged: dict[str, s5_decision.Judged], pools: dict[str, str], reason: str) -> None:
    decision, reasons = dev.gate(_worlds(judged, **pools), WORLDS)
    assert decision == "NO-GO" and len(reasons) == 1 and reasons[0].startswith(reason), reasons


def test_a_missing_extra_or_twice_built_world_is_no_go(judged: dict[str, s5_decision.Judged]) -> None:
    worlds = _worlds(judged)
    assert dev.gate(worlds[:8], WORLDS) == ("NO-GO", ["285672490/full-null: missing"])
    extra = dataclasses.replace(worlds[0], seed=EVALUATION.certification[0])
    assert dev.gate([*worlds, extra], WORLDS)[1] == [f"{EVALUATION.certification[0]}/story: not in the manifest"]
    assert dev.gate([*worlds, worlds[4]], WORLDS)[1] == ["1797428728/skill-neutral: twice"]


def test_a_parity_difference_or_none_is_no_go(judged: dict[str, s5_decision.Judged]) -> None:
    worlds = _worlds(judged)
    worlds[2] = dataclasses.replace(worlds[2], parity=Parity(("PHA quality: SQL 1, twin 0.9",), 0, 0.1, 0, 1))
    worlds[3] = dataclasses.replace(worlds[3], parity=None)
    assert dev.gate(worlds, WORLDS)[1] == [
        "31102026/full-null: parity: PHA quality: SQL 1, twin 0.9",
        "1797428728/story: parity not run",
    ]


# ── the runner ───────────────────────────────────────────────────────────────


def test_the_runner_refuses_a_checkout_with_changes_before_any_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def never(*_: object) -> None:
        raise AssertionError("built")

    monkeypatch.setattr(dev, "build_missing", never)
    monkeypatch.setattr(dev, "git_sha", lambda: (HEAD, True))
    assert dev.main([str(tmp_path), "--build"]) == 1
    assert "has changes" in capsys.readouterr().err and not any(tmp_path.iterdir())
    monkeypatch.setattr(dev, "git_sha", lambda: (None, False))
    assert dev.main([str(tmp_path), "--build"]) == 1


def test_build_only_builds_the_manifest_and_judges_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    built: list[tuple[Path, tuple[tuple[int, str], ...]]] = []
    monkeypatch.setattr(dev, "build_missing", lambda root, worlds: built.append((root, tuple(worlds))))
    monkeypatch.setattr(dev, "develop", lambda *_: pytest.fail("judged"))
    monkeypatch.setattr(dev, "git_sha", lambda: (HEAD, False))
    assert dev.main([str(tmp_path), "--build-only"]) == 0
    assert dev.main([str(tmp_path), "--evaluation", MONTHLY_REVISION, "--build-only"]) == 0
    assert built == [(tmp_path, WORLDS)] * 2 and "none judged" in capsys.readouterr().out
    assert not (tmp_path / "s5-development.json").exists()


def test_the_build_command_is_build_world_s_with_the_seed(tmp_path: Path) -> None:
    args = dev.command(1797428728, "full-null", tmp_path / "w")
    assert args[1:] == [
        "-m",
        "hartland_sim.cli",
        "build-world",
        "--csv",
        "--pilot",
        "full-null",
        "--seed",
        "1797428728",
        "--run",
        str(tmp_path / "w"),
    ]


def test_the_readout_reports_every_world_round_and_region(
    suite: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without parity the runner says NO-GO, and its readout gives each world's claim, metrics, the six regions, their
    24 rounds, the audit hashes and NOT_CERTIFIED; the Czech readout of each world follows the table."""
    root = _copy(suite, tmp_path)
    monkeypatch.setattr(dev, "git_sha", lambda: (HEAD, False))
    assert dev.main([str(root)]) == 1
    out = json.loads((root / "s5-development.json").read_text(encoding="utf-8"))
    assert out["decision"] == "NO-GO" and out["head"] == HEAD and out["certification"] == "NOT_CERTIFIED"
    assert out["evaluation"] == dict(EVALUATION.identity()) and set(out["locks"]) == {"uv.lock", "toolchain.lock"}
    assert [(w["seed"], w["mode"]) for w in out["worlds"]] == list(WORLDS)
    story = out["worlds"][0]
    assert story["claim"] == "PASS" and story["certification"] == "NOT_CERTIFIED"
    assert story["verdict"]["rank"] == pytest.approx(0.9856, abs=1e-4) and story["gate"] == [
        "31102026/story: parity not run"
    ]
    assert list(story["regions"]) == list(REGIONS) and story["regions"]["PHA"]["chosen"] == 120
    assert {region: len(rounds) for region, rounds in story["rounds"].items()} == dict.fromkeys(REGIONS, 24)
    assert story["rounds"]["SMS"][0] == {
        "decision_day": "2023-12-31",
        "region_id": 6,
        "decisions": 13,
        "viable": 10,
        "chosen": 5,
        "unchosen": 5,
        "q": 0.0,
    }
    assert set(story["audit"]) == {"pos_pilot_assignment", "pos_placement_decision"}
    text = (root / "s5-development.md").read_text(encoding="utf-8")
    assert text.startswith("# S5 development gate: NO-GO") and text.count("Kvalita výběru umístění stojanů") == 9
