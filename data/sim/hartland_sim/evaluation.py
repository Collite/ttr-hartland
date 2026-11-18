"""S5's evaluation (G2 S2.9; NH-D25 design09 §§2–6): `data/s5_evaluation.yaml`, read without the generator's machinery.

It owns the claim — the quality of the managers' placement decisions against their own forecasts, never a measured
gain or payback — its revision, the window and rounds of its measure, the support each region needs, the acceptance
bars, the Czech labels that keep forecast and observed apart, and the seeds: the development three (the protocol's)
and ten certification seeds of its own, each `seed(i)`. `data/s5_protocol.yaml` stays the generator's contract: this
file is no input of a build's id, but a pilot-mode build records its revision, its sha256 and its claim in `_sim_meta`
(`META_KEYS`, `identity`), and a build of another evaluation file is not certified by this one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from hartland_sim.config import DATA_DIR

EVALUATION_FILE = DATA_DIR / "s5_evaluation.yaml"
CLAIM = "placement_decision_quality"
# the `_sim_meta` keys a pilot-mode build adds after the protocol's (`loop.pilot.metadata`), in its order
META_KEYS = ("s5_evaluation_revision", "s5_evaluation_sha256", "s5_claim")


def seed(i: int) -> int:
    """design09 §6's certification seed i: a master seed in [1, 2**31 − 2] from the revision's label and the index."""
    digest = hashlib.sha256(f"NH-D25/v1/cert/{i}".encode()).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (2**31 - 2)


@dataclass(frozen=True)
class Evaluation:
    """`data/s5_evaluation.yaml` (module docstring)."""

    revision: str
    claim: str
    pilot_revision: str
    policy_revision: str
    window_from: date
    window_until: date
    rounds: int
    support: dict[str, int]
    acceptance: dict[str, Any]
    labels: dict[str, str]
    development: tuple[int, ...]
    certification: tuple[int, ...]

    @classmethod
    def load(cls, path: Path = EVALUATION_FILE) -> Evaluation:
        held = yaml.safe_load(path.read_text(encoding="utf-8"))
        seeds = held["seeds"]
        return cls(
            held["revision"],
            held["claim"],
            held["pilot_revision"],
            held["policy_revision"],
            held["window"]["from"],
            held["window"]["until"],
            int(held["rounds"]),
            {name: int(n) for name, n in held["support"].items()},
            dict(held["acceptance"]),
            {name: str(text) for name, text in held["labels"].items()},
            tuple(int(s) for s in seeds["development"]),
            tuple(int(s) for s in seeds["certification"]),
        )


def identity(path: Path = EVALUATION_FILE) -> tuple[tuple[str, str], ...]:
    """The `_sim_meta` keys of `META_KEYS` for the evaluation file at `path`: its revision, its sha256, its claim."""
    held = Evaluation.load(path)
    return tuple(
        zip(META_KEYS, (held.revision, hashlib.sha256(path.read_bytes()).hexdigest(), held.claim), strict=True)
    )
