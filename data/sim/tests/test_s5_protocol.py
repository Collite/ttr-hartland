"""G2 S2.6 pre-flight — S5's pilot protocol (`data/s5_protocol.yaml`, design08 §8): its seeds are the design's
formula's, collide with nothing, and the certification seeds were never built before it."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from hartland_sim.config import WorldConfig
from hartland_sim.loop.pilot import Protocol, seed


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
