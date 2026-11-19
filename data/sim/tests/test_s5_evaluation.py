"""G2 S2.9 T4 — S5's evaluation (`data/s5_evaluation.yaml`, design09 §§3, 5–6): its claim and revision, its bars and
support, its Czech labels, and its seeds — the protocol's development three and ten certification seeds of its own,
the design's formula's, colliding with no seed the protocol names or built before. G2 S2.10: the build's seed override
(`evaluation.admit`, `build-world --seed`; ttr-hartland#22's tests, `42b0d67`) refuses a certification seed of either
file and a seed built before the manifest that is no development seed, before any work."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.cli import main
from hartland_sim.config import WorldConfig
from hartland_sim.evaluation import CLAIM, Evaluation, SeedRefused, admit, seed
from hartland_sim.loop.build import build
from hartland_sim.protocol import Protocol


@pytest.fixture(scope="module")
def held() -> Evaluation:
    return Evaluation.load()


def test_the_claim_and_bars_are_the_designs(held: Evaluation) -> None:
    assert (held.revision, held.claim) == ("NH-D25/v1", CLAIM) == ("NH-D25/v1", "placement_decision_quality")
    assert (held.window_from, held.window_until, held.rounds) == (date(2024, 1, 1), date(2025, 12, 31), 24)
    assert held.support == {"comparable_rounds": 12, "chosen": 100, "unchosen": 100}
    bars = held.acceptance
    assert (bars["rank"], bars["q_high"], bars["difference"]) == (0.8, 0.70, 0.10)
    assert (bars["high"], bars["low"]) == (["PHA", "JMV"], ["ZSC", "VCH", "SMS"])
    assert "group_ratio" not in bars and "rms" not in bars  # the causal bars are retired, not carried over


def test_the_evaluation_names_the_protocols_revisions_and_window(held: Evaluation) -> None:
    protocol = Protocol.load()
    assert (held.pilot_revision, held.policy_revision) == (protocol.pilot_revision, protocol.policy_revision)
    assert (held.window_from, held.window_until) == (protocol.window_from, protocol.window_until)


def test_the_seeds_are_the_designs_and_collide_with_nothing(held: Evaluation) -> None:
    protocol = Protocol.load()
    assert held.development == protocol.development == (31102026, 1797428728, 285672490)
    assert held.certification == tuple(seed(i) for i in range(1, 11))
    assert held.certification[:3] == (895899358, 415492059, 825635122)  # design09 §6's numbers
    every = (*protocol.development, *protocol.certification, *protocol.built_before)
    assert not set(held.certification) & set(every) and len(set(held.certification)) == 10
    assert all(1 <= s <= 2**31 - 2 for s in held.certification)


def test_the_labels_read_as_czech_and_name_no_gain(held: Evaluation) -> None:
    assert held.labels["title_cs"] == "Kvalita výběru umístění stojanů"
    assert held.labels["disclosure_cs"] == "Podle předpovědi před rozhodnutím; nejde o prokázaný přírůstek marže."
    assert_czech_all(list(held.labels.values()), what="the evaluation's labels")


def test_the_seed_override_refuses_certification_and_stale_seeds(held: Evaluation) -> None:
    protocol = Protocol.load()
    for i, taken in enumerate(held.certification, start=1):
        with pytest.raises(SeedRefused, match=f"certification seed {i} of NH-D25/v1"):
            admit(taken)
    for taken in protocol.certification:
        with pytest.raises(SeedRefused, match="retired certification seed"):
            admit(taken)
    with pytest.raises(SeedRefused, match="built before the manifest"):
        admit(31102027)  # the determinism tests' world, no development seed
    for admitted in (*held.development, 12345):
        admit(admitted)


class Reached(Exception):
    """Raised by the patched setup: a build reached its work."""


@pytest.mark.parametrize("taken", [895899358, 1139534158, 1206907222, 31102027])
def test_a_refused_seed_builds_nothing(
    taken: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refused before any work — no setup, no OSM market, no file — by `build` and by the CLI (one line, exit 1)."""

    def never(*_: object) -> None:
        raise Reached

    monkeypatch.setattr("hartland_sim.loop.build.setup", never)
    monkeypatch.setattr("hartland_sim.setup.market.build", never)
    with pytest.raises(SeedRefused):
        build(tmp_path / "b", date(2023, 3, 1), pilot="story", seed=taken)
    assert main(["build-world", "--csv", "--pilot", "no-pos", "--seed", str(taken), "--run", str(tmp_path / "c")]) == 1
    assert "seed" in capsys.readouterr().err and not list(tmp_path.rglob("*.csv"))


def test_an_admitted_seed_is_the_build_s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A development seed replaces world.yaml's for the whole build (setup runs under it)."""
    seen: list[int] = []

    def setup(world: WorldConfig, *_: object) -> None:
        seen.append(world.master_seed)
        raise Reached

    monkeypatch.setattr("hartland_sim.loop.build.setup", setup)
    with pytest.raises(Reached):
        build(tmp_path / "b", date(2023, 3, 1), pilot="story", seed=1797428728)
    assert seen == [1797428728]
