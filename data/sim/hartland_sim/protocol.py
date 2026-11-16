"""S5's pilot protocol (G2 S2.6; NH-D21-B design08 §8): `data/s5_protocol.yaml`, read without the generator's machinery;
`MODES` repeats the file's five for the CLI's parser, which reads no file (a test holds the two equal).

It is committed before any pilot-mode build runs and never edited: a change is a new design revision with a
`pilot_revision` of its own. It holds the pilot's revisions, the PoS-free months and the assignment day after them,
the baseline's days, the window W, the share of the display budgets the pilot places, the five modes, the acceptance
bars S2.7–S2.8 apply, and the seeds: the development three, the ten certification seeds — each `seed(label, i)` — and
the master seeds built before the manifest, which a certification seed may not be.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from hartland_sim.config import DATA_DIR

PROTOCOL_FILE = DATA_DIR / "s5_protocol.yaml"
MODES = ("story", "skill-neutral", "full-null", "no-pos", "no-pos-null")  # the file's, for the CLI's parser
# the `_sim_meta` keys a pilot-mode build adds (`loop.pilot.metadata`), in its order; `world.Meta.load` reads these only
META_KEYS = (
    "pilot_mode",
    "pilot_design",
    "pilot_revision",
    "policy_revision",
    "pilot_protocol_sha256",
    "pilot_window",
    "pilot_term_days",
    "assignment_digest",
    "s5_status",
)


def seed(label: str, i: int) -> int:
    """design08 §8's `seed(label, i)`: a master seed in [1, 2**31 − 2] from the design's label and the index."""
    digest = hashlib.sha256(f"NH-D21-B/v1/{label}/{i}".encode()).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (2**31 - 2)


@dataclass(frozen=True)
class Protocol:
    """`data/s5_protocol.yaml` (module docstring)."""

    design: str
    pilot_revision: str
    assignment_revision: int
    policy_revision: str
    pos_free_from: date
    pos_free_until: date
    baseline_days: int
    window_from: date
    window_until: date
    placement_budget_share: float
    asset_days_per_month: float
    modes: tuple[str, ...]
    acceptance: dict[str, Any]
    development: tuple[int, ...]
    certification: tuple[int, ...]
    built_before: tuple[int, ...]

    @property
    def assigned_on(self) -> date:
        """The assignment's day, the first a pilot display may be lent from: the day after the PoS-free months."""
        return self.pos_free_until + timedelta(days=1)

    @property
    def baseline_from(self) -> date:
        return self.pos_free_until - timedelta(days=self.baseline_days - 1)

    @classmethod
    def load(cls, path: Path = PROTOCOL_FILE) -> Protocol:
        held = yaml.safe_load(path.read_text(encoding="utf-8"))
        seeds = held["seeds"]
        return cls(
            held["design"],
            held["pilot_revision"],
            int(held["assignment_revision"]),
            held["policy_revision"],
            held["pos_free"]["from"],
            held["pos_free"]["until"],
            int(held["baseline_days"]),
            held["window"]["from"],
            held["window"]["until"],
            float(held["placement_budget_share"]),
            float(held["asset_days_per_month"]),
            tuple(held["modes"]),
            dict(held["acceptance"]),
            tuple(int(s) for s in seeds["development"]),
            tuple(int(s) for s in seeds["certification"]),
            tuple(int(s) for s in seeds["built_before"]),
        )
