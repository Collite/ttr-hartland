"""G2 S2.6 T4 — S5's pilot audit tables (design08 §9, NH-D22): exactly the declared fields, keys and outcomes in the
inventory, the DDL and the rows the state renders; no oracle column."""

from __future__ import annotations

from datetime import date

import pytest

from hartland_sim.config import DATA_DIR, inventory
from hartland_sim.io.csv_writer import cells
from hartland_sim.loop.pilot import ARMS, OUTCOMES, Assigned, Decision, Mapped, Pilot, state_rows
from hartland_sim.protocol import Protocol

ASSIGNMENT = {  # design08 §9: name → (type, nullable)
    "pos_pilot_assignment_id": ("bigint", False),
    "pilot_revision": ("text", False),
    "customer_id": ("bigint", False),
    "region_id": ("bigint", False),
    "pair_id": ("bigint", True),
    "arm": ("text", False),
    "outside_reason": ("text", True),
    "portfolio_key": ("text", False),
    "assigned_on": ("date", False),
    "baseline_from": ("date", False),
    "baseline_until": ("date", False),
    "window_from": ("date", False),
    "window_until": ("date", False),
    "baseline_customer_margin": ("numeric(14,2)", False),
    "baseline_days": ("integer", False),
    "assignment_digest": ("text", False),
}
DECISION = {
    "pos_placement_decision_id": ("bigint", False),
    "pilot_revision": ("text", False),
    "policy_revision": ("text", False),
    "decision_day": ("date", False),
    "customer_id": ("bigint", False),
    "delivery_point_id": ("bigint", False),
    "region_id": ("bigint", False),
    "manager_id": ("bigint", True),
    "forecast_material_id": ("bigint", True),
    "outcome": ("text", False),
    "prior56_line_margin": ("numeric(14,2)", False),
    "prior56_handout_cost": ("numeric(14,2)", False),
    "asset_cost": ("numeric(14,2)", True),
    "forecast_gain": ("numeric(14,2)", True),
    "forecast_cost": ("numeric(14,2)", True),
    "catalogue_fit": ("numeric(18,8)", False),
    "score": ("numeric(18,8)", False),
    "forecast_days": ("integer", False),
    "loan_id": ("bigint", True),
}


@pytest.mark.parametrize(
    ("name", "declared", "key", "refs"),
    [
        ("pos_pilot_assignment", ASSIGNMENT, ("pilot_revision", "customer"), {"customer_id", "region_id"}),
        (
            "pos_placement_decision",
            DECISION,
            ("pilot_revision", "decision_day", "delivery_point"),
            {"customer_id", "delivery_point_id", "region_id", "manager_id", "forecast_material_id", "loan_id"},
        ),
    ],
)
def test_the_audit_tables_are_the_declared_ones(
    name: str, declared: dict[str, tuple[str, bool]], key: tuple[str, ...], refs: set[str]
) -> None:
    (table,) = [t for t in inventory().tables if t.name == name]
    assert {c.name: (c.type, c.nullable) for c in table.columns} == declared
    assert table.id_key == key and table.pk == (f"{name}_id",)
    assert {c.name for c in table.columns if c.ref} == refs
    parents = {t.name: t.load_order for t in inventory().tables}
    assert all(parents[c.ref.split(".")[0]] < table.load_order for c in table.columns if c.ref)


def test_the_six_outcomes_agree_across_the_inventory_and_the_ddl() -> None:
    held = inventory()
    assert tuple(held.codes["pilot_outcome"]) == OUTCOMES
    assert OUTCOMES[-1] == "cancelled" and len(OUTCOMES) == 6  # NH-D22
    assert tuple(held.codes["pilot_arm"]) == ARMS
    schema = (DATA_DIR / "ddl" / "schema.sql").read_text(encoding="utf-8")
    assert "outcome text NOT NULL CHECK (outcome IN ({}))".format(", ".join(f"'{o}'" for o in OUTCOMES)) in schema
    assert "loan_id bigint,\n" in schema  # nullable: a cancelled decision links no loan


def test_the_state_renders_rows_the_writer_takes() -> None:
    protocol = Protocol.load()
    pilot = Pilot("story", protocol, digest="d" * 64)
    pilot.assigned[7] = Assigned(Mapped("PHA", "programme", 11, None, "convenience/residential/S"), 123456)
    pilot.assigned[8] = Assigned(Mapped("PHA", "outside", None, "unpaired", "village/village/S"), 0)
    held = Decision(7, 70, "PHA", None, None, 0, 0, None, None, None, 0.0, 0.0, 183, "no_asset")
    pilot.decisions[(date(2023, 2, 28), 70)] = held
    assigned, decided = state_rows(pilot)
    tables = {t.name: t for t in inventory().tables}
    assert [cells(tables["pos_pilot_assignment"], r)[5] for r in assigned] == ["programme", "outside"]
    (row,) = decided
    assert cells(tables["pos_placement_decision"], row)[9] == "no_asset" and row["loan_id"] is None
    assert row["decision_day"] == date(2023, 2, 28) and row["forecast_material_id"] is None
    assert state_rows(None) == ([], [])
