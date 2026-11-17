"""G2 S2.6 pre-flight — S5's pilot protocol (`data/s5_protocol.yaml`, design08 §8): its seeds are the design's
formula's, collide with nothing, and the certification seeds were never built before it. G2 S2.7 T4: the build's seed
override (`protocol.admit`, `build-world --seed`) refuses a certification seed and a seed built before the manifest
that is no development seed, before any work (review-295 P1); S2.8's certification run alone may take them."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from hartland_sim.cli import main
from hartland_sim.config import WorldConfig
from hartland_sim.loop.build import build
from hartland_sim.protocol import Protocol, SeedRefused, admit, seed


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    return Protocol.load()


def test_the_seeds_are_the_designs(protocol: Protocol) -> None:
    assert protocol.development == (WorldConfig.load().master_seed, seed("dev", 1), seed("dev", 2))
    assert protocol.development == (31102026, 1797428728, 285672490)  # design08 §8's numbers
    assert protocol.certification == tuple(seed("cert", i) for i in range(1, 11))
    assert protocol.certification[:3] == (1206907222, 47937215, 2091533869)


def test_no_seed_collides_and_no_certification_seed_was_built_before(protocol: Protocol) -> None:
    every = protocol.development + protocol.certification
    assert len(set(every)) == len(every)
    assert not set(protocol.certification) & set(protocol.built_before)
    assert all(1 <= s <= 2**31 - 2 for s in every)


def test_the_dates_are_the_designs(protocol: Protocol) -> None:
    assert (protocol.pos_free_from, protocol.assigned_on) == (date(2023, 1, 1), date(2023, 3, 1))
    assert protocol.baseline_from == date(2023, 1, 4)  # the 56 days before the assignment
    assert (protocol.pos_free_until - protocol.baseline_from).days + 1 == protocol.baseline_days == 56
    assert (protocol.window_until - protocol.window_from + timedelta(days=1)).days == 731
    assert protocol.modes == ("story", "skill-neutral", "full-null", "no-pos", "no-pos-null")
    assert protocol.placement_budget_share == 0.5
    assert protocol.acceptance["high"] == ["PHA", "JMV"] and protocol.acceptance["group_ratio"] == 1.35


def test_the_seed_override_refuses_certification_and_stale_seeds(protocol: Protocol) -> None:
    for i, held in enumerate(protocol.certification, start=1):
        with pytest.raises(SeedRefused, match=f"certification seed {i}"):
            admit(held, protocol)
        admit(held, protocol, certifying=True)  # S2.8's `just s5-certify`
    with pytest.raises(SeedRefused, match="built before the manifest"):
        admit(31102027, protocol)  # the determinism tests' world, no development seed
    for held in (*protocol.development, 12345):
        admit(held, protocol)


class Refused(Exception):
    """Raised by the patched setup: a build reached its work."""


@pytest.mark.parametrize("held", [1206907222, 1321611371, 31102027])
def test_a_refused_seed_builds_nothing(
    held: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refused before any work — no setup, no OSM market, no file — by `build` and by the CLI (one line, exit 1)."""

    def never(*_: object) -> None:
        raise Refused

    monkeypatch.setattr("hartland_sim.loop.build.setup", never)
    monkeypatch.setattr("hartland_sim.setup.market.build", never)
    with pytest.raises(SeedRefused):
        build(tmp_path / "b", date(2023, 3, 1), pilot="story", seed=held)
    assert main(["build-world", "--csv", "--pilot", "no-pos", "--seed", str(held), "--run", str(tmp_path / "c")]) == 1
    assert "seed" in capsys.readouterr().err and not list(tmp_path.rglob("*.csv"))


def test_an_admitted_seed_is_the_build_s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A development seed replaces world.yaml's for the whole build (setup runs under it)."""
    seen: list[int] = []

    def setup(world: WorldConfig, *_: object) -> None:
        seen.append(world.master_seed)
        raise Refused

    monkeypatch.setattr("hartland_sim.loop.build.setup", setup)
    with pytest.raises(Refused):
        build(tmp_path / "b", date(2023, 3, 1), pilot="story", seed=1797428728)
    assert seen == [1797428728]
