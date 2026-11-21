"""A development readout only proceeds with sealed inputs and resumes its own checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hartland_sim.checks import s5_monthly_development as run
from hartland_sim.checks.s5_development import Refused
from hartland_sim.checks.s5_monthly import MONTHLY_FILE
from hartland_sim.evaluation import Evaluation
from hartland_sim.protocol import Protocol


@pytest.mark.parametrize("change", ["receipt", "content", "extra", "link"])
def test_input_attestation_detects_contamination(tmp_path: Path, change: str) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    original = inputs / "world.csv"
    original.write_text("original\n")
    receipt = tmp_path / "input.sha256"
    receipt.write_text(f"{run.sha256(original)}  ./world.csv\n")
    seal = run.sha256(receipt)
    assert run.authenticate(inputs, receipt, seal) == 1
    if change == "receipt":
        receipt.write_text("wrong")
    elif change == "content":
        original.write_text("changed")
    elif change == "extra":
        (inputs / "extra.csv").touch()
    else:
        (inputs / "link.csv").symlink_to(original)
    reason = {"receipt": "receipt", "content": "hash", "extra": "inventory", "link": "link"}[change]
    with pytest.raises(Refused, match=reason):
        run.authenticate(inputs, receipt, seal)


def test_resume_is_keyed_by_the_published_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def stable(_inputs: Path, seed: int, mode: str, *_: Any) -> dict[str, Any]:
        return {
            "seed": seed,
            "mode": mode,
            "claim": "PASS",
            "reasons": [],
            "gate": [],
            "sensitivity": {"count": 300, "failure_type": None, "failures": [], "full_failure": None},
        }

    worlds = [(31102026, "story")]
    held, protocol = Evaluation.load(MONTHLY_FILE), Protocol.load()
    monkeypatch.setattr(run, "_world", stable)
    source = tmp_path / "input"
    root = tmp_path / "out"
    root.mkdir()
    receipt = {"head": "head"}
    assert run.score(root, source, receipt, "published", worlds, held, protocol, "unused") == "GO"
    monkeypatch.setattr(run, "_world", lambda *_: pytest.fail("already assessed"))
    assert run.score(root, source, receipt, "published", worlds, held, protocol, "unused") == "GO"
    with pytest.raises(Refused, match="different freeze"):
        run.score(root, source, receipt, "different", worlds, held, protocol, "unused")
