"""S5's opt-in pilot (G2 S2.6; NH-D21-B design08 §§3–4 and §9, NH-D22): its frozen protocol (`Protocol`,
`data/s5_protocol.yaml`).

**The protocol** is committed before any pilot-mode build runs and never edited: a change is a new design revision
with a `pilot_revision` of its own. It holds the pilot's revisions, the PoS-free months and the assignment day after
them, the baseline's days, the window W, the share of the display budgets the pilot places, the five modes, the
acceptance bars S2.7–S2.8 apply, and the seeds: the development three, the ten certification seeds — each
`seed(label, i)` — and the master seeds built before the manifest, which a certification seed may not be.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from hartland_sim.config import DATA_DIR
from hartland_sim.ids import row_id
from hartland_sim.setup.rows import Row, row

PROTOCOL_FILE = DATA_DIR / "s5_protocol.yaml"


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


# ── the state (`World.pilot`) and its rows ──────────────────────────────────

ARMS = ("programme", "holdout", "outside")
OUTCOMES = ("selected", "no_budget", "no_asset", "nonpositive_score", "ineligible", "cancelled")


@dataclass(frozen=True)
class Mapped:
    """A customer's place on the frozen map (the story pre-pilot's, design08 §3): its region, arm, pair and the
    portfolio it was sorted by."""

    region: str
    arm: str  # programme · holdout · outside
    pair_id: int | None
    outside_reason: str | None  # unpaired · cross_region; None in a pair
    portfolio_key: str


@dataclass
class Assigned:
    """A customer's `pos_pilot_assignment` row: its place on the map and its own observed baseline, the whole
    customer's Customer Margin over the baseline days, in haléře."""

    mapped: Mapped
    baseline: int


@dataclass
class Decision:
    """A `pos_placement_decision` row (design08 §§4, 9; NH-D22): the inputs snapshotted before the month's draw — the
    forecast material (None when none was in stock), the 56 days' margin of its lines at the store and the customer's
    give-aways (haléře), the asset's cost, the forecast gain and cost (None without a material), the catalogue fit,
    the score and the forecast's days — then its outcome and the loan it made."""

    customer_id: int
    delivery_point_id: int
    region: str
    manager_id: int | None
    material: str | None
    prior56_line: int
    prior56_handouts: int
    asset_cost: float | None
    gain: float | None
    cost: float | None
    fit: float
    score: float
    forecast_days: int
    outcome: str = "ineligible"
    loan_id: int | None = None


@dataclass
class Pilot:
    """A pilot-mode build's S5 state (module docstring): its mode and protocol, the story pre-pilot's map (the modes
    other than `story` take it), the assignments by customer_id and their digest, the decisions by (decision day,
    delivery_point_id), and the observed history the score reads — the baseline days' margin by customer, and by day
    (its ordinal) the margin of each (store's delivery point, line) and the give-aways of each customer, in haléře."""

    mode: str
    protocol: Protocol
    story_map: dict[int, Mapped] | None = None
    assigned: dict[int, Assigned] = field(default_factory=dict)
    digest: str = ""
    decisions: dict[tuple[date, int], Decision] = field(default_factory=dict)
    baseline: dict[int, int] = field(default_factory=dict)
    lines: dict[int, dict[tuple[int, int], int]] = field(default_factory=dict)
    handouts: dict[int, dict[int, int]] = field(default_factory=dict)


def assignment_row(pilot: Pilot, customer_id: int, held: Assigned) -> Row:
    p, m = pilot.protocol, held.mapped
    return row(
        "pos_pilot_assignment",
        pos_pilot_assignment_id=row_id("pos_pilot_assignment", pilot_revision=p.pilot_revision, customer=customer_id),
        pilot_revision=p.pilot_revision,
        customer_id=customer_id,
        region_id=row_id("region", code=m.region),
        pair_id=m.pair_id,
        arm=m.arm,
        outside_reason=m.outside_reason,
        portfolio_key=m.portfolio_key,
        assigned_on=p.assigned_on,
        baseline_from=p.baseline_from,
        baseline_until=p.pos_free_until,
        window_from=p.window_from,
        window_until=p.window_until,
        baseline_customer_margin=held.baseline / 100,
        baseline_days=p.baseline_days,
        assignment_digest=pilot.digest,
    )


def decision_row(pilot: Pilot, day: date, d: Decision) -> Row:
    p, point = pilot.protocol, d.delivery_point_id
    return row(
        "pos_placement_decision",
        pos_placement_decision_id=row_id(
            "pos_placement_decision", pilot_revision=p.pilot_revision, decision_day=day, delivery_point=point
        ),
        pilot_revision=p.pilot_revision,
        policy_revision=p.policy_revision,
        decision_day=day,
        customer_id=d.customer_id,
        delivery_point_id=d.delivery_point_id,
        region_id=row_id("region", code=d.region),
        manager_id=d.manager_id,
        forecast_material_id=row_id("pos_material", code=d.material) if d.material else None,
        outcome=d.outcome,
        prior56_line_margin=d.prior56_line / 100,
        prior56_handout_cost=d.prior56_handouts / 100,
        asset_cost=d.asset_cost,
        forecast_gain=d.gain,
        forecast_cost=d.cost,
        catalogue_fit=d.fit,
        score=d.score,
        forecast_days=d.forecast_days,
        loan_id=d.loan_id,
    )


def state_rows(pilot: Pilot | None) -> tuple[list[Row], list[Row]]:
    """The two audit tables' rows, rendered from the state (`setup.opening.state_rows`): the assignments in the map's
    order, the decisions by day then the order they were made; none outside a pilot-mode build."""
    if pilot is None:
        return [], []
    assigned = [assignment_row(pilot, customer_id, held) for customer_id, held in pilot.assigned.items()]
    return assigned, [decision_row(pilot, day, d) for (day, _), d in pilot.decisions.items()]
