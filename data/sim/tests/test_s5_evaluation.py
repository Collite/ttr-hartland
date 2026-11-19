"""G2 S2.9 T4 — S5's evaluation (`data/s5_evaluation.yaml`, design09 §§3, 5–6): its claim and revision, its bars and
support, its Czech labels, and its seeds — the protocol's development three and ten certification seeds of its own,
the design's formula's, colliding with no seed the protocol names or built before."""

from __future__ import annotations

from datetime import date

import pytest

from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.evaluation import CLAIM, Evaluation, seed
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
