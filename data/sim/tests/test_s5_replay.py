"""S5's no-PoS reference, isolation and development gate (G2 S2.7 T3–T5, `checks.s5_replay`; design08 §§7–8) on
conftest's hand-built pilot: a known effect on the displayed line and on another, which the reference counts whole —
never the displayed line's margin divided by its uplift; isolation as Bora ruled it (prompts and visit notes reported,
anything else outside the programme a failure); the gate's conditions; and the whole gate over three seeds × five
modes, where a failed isolation scores nothing."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.checks import s5_pilot, s5_replay
from hartland_sim.checks.s5_replay import Readout, Scored, develop, isolation, reference
from hartland_sim.checks.stories import Build
from hartland_sim.protocol import Protocol

PilotRows = Callable[..., Any]  # conftest's `pilot_rows`
EFFECTS = {"PHA": 8000.0, "JMV": 6000.0, "SJC": 4000.0, "ZSC": 2000.0, "VCH": -1000.0, "SMS": 1000.0}
FLAT = dict.fromkeys(EFFECTS, 2000.0)
NONE = dict.fromkeys(EFFECTS, 0.0)
OTHER = 500.0  # each programme customer's other line, sold only where the display is (basket spillover)
PROTOCOL = Protocol.load()


def _catalogue(case: Any) -> Any:
    """A second product line, in every build of the tree (a setup table)."""
    case.rows["product_line"].append({"product_line_id": 2, "code": "L2"})
    case.rows["product"].append({"product_id": 2, "product_line_id": 2})
    case.rows["sku"].append({"sku_id": 2, "product_id": 2})
    return case


def _treated(case: Any, other: float = OTHER) -> Any:
    """The programme's other-line gain: a line of the second product line on each programme customer's first order."""
    _catalogue(case)
    programme = {r["customer_id"] for r in case.rows["pos_pilot_assignment"] if r["arm"] == "programme"}
    for order in case.rows["sales_order"]:
        if order["customer_id"] in programme and order["ordered_on"] == s5_first():
            case.rows["sales_order_line"].append(
                {"sales_order_line_id": 10 * order["sales_order_id"] + 9, "sales_order_id": order["sales_order_id"]}
                | {"line_no": 9, "sku_id": 2, "qty_cases": 1.0, "list_price_case": other, "is_free_goods": False}
                | {"delivery_cost_case": None, "warehouse_cost_case": None}
            )
            quarter = f"{order['ordered_on'].year}-Q1"
            case.quarters[quarter]["lines"] += 1
    return case


def s5_first() -> Any:
    return PROTOCOL.window_from


def _replay(case: Any) -> Any:
    """The case's replay: no display, no give-away (`pilot_rows` with no effect, then the PoS rows dropped)."""
    _catalogue(case)
    for table in ("pos_loan", "pos_asset", "pos_handout"):
        case.rows[table] = []
    case.rows["visit"] = []
    return case


def _pair(pilot_rows: PilotRows, tmp_path: Path, effects: dict[str, float] = EFFECTS) -> tuple[Path, Path]:
    actual, replay = tmp_path / "actual", tmp_path / "replay"
    _treated(pilot_rows(effects)).write(actual)
    _replay(pilot_rows(NONE, mode="no-pos")).write(replay)
    return actual, replay


# ── the reference ──────────────────────────────────────────────────────────────


def test_the_reference_counts_the_other_lines_too(pilot_rows: PilotRows, tmp_path: Path) -> None:
    """The programme raises the displayed line by the region's effect and another line by 500.00: the reference is
    both, whole customer, per pair — not the displayed line alone (what dividing its margin by 1 + uplift would see) —
    and the observed paired gain finds the same, so its normalized error is 0."""
    actual, replay = _pair(pilot_rows, tmp_path)
    built = Build(actual)
    codes: dict[Any, str] = dict(built["region"][["region_id", "code"]].itertuples(index=False))
    found = reference(s5_pilot.outcomes(built), s5_pilot.outcomes(Build(replay)), codes)
    for region, effect in EFFECTS.items():
        assert found.at[region, "reference_gain"] == pytest.approx(2 * (effect + OTHER), abs=0.005)
        assert found.at[region, "reference_gain"] != pytest.approx(2 * effect, abs=1.0)  # not the displayed line only
        assert (found.at[region, "holdout_change"], found.at[region, "holdouts_changed"]) == (0.0, 0)
    observed = s5_pilot.regions(built)
    assert s5_replay.errors(observed, found).abs().max() < 1e-9


def test_a_holdout_that_moved_shows_in_the_reference(pilot_rows: PilotRows, tmp_path: Path) -> None:
    actual, replay = tmp_path / "actual", tmp_path / "replay"
    case = _treated(pilot_rows(EFFECTS))
    line = next(r for r in case.rows["sales_order_line"] if r["sales_order_line_id"] // 10 in _orders(case, "SJC", 2))
    line["list_price_case"] += 250.0
    case.write(actual)
    _replay(pilot_rows(NONE, mode="no-pos")).write(replay)
    built = Build(actual)
    codes: dict[Any, str] = dict(built["region"][["region_id", "code"]].itertuples(index=False))
    found = reference(s5_pilot.outcomes(built), s5_pilot.outcomes(Build(replay)), codes)
    assert (found.at["SJC", "holdout_change"], found.at["SJC", "holdouts_changed"]) == (250.0, 1)
    errors = s5_replay.errors(s5_pilot.regions(built), found)
    assert errors["SJC"] == pytest.approx(-250.0 / 9000.0)  # its rise counts against the gain, ÷ the reference 9 000


def _orders(case: Any, region: str, n: int) -> set[int]:
    return {o["sales_order_id"] for o in case.rows["sales_order"] if o["customer_id"] == case.customer(region, n)}


# ── isolation ──────────────────────────────────────────────────────────────────


def _append(run: Path, table: str, row: dict[str, Any]) -> None:
    path = run / f"{table}.csv"
    with path.open(encoding="utf-8", newline="") as source:
        header = next(csv.reader(source))
    with path.open("a", encoding="utf-8", newline="") as out:
        csv.DictWriter(out, fieldnames=header, lineterminator="\n").writerow(row)


def test_isolation_holds_where_only_the_programme_moved(pilot_rows: PilotRows, tmp_path: Path) -> None:
    actual, replay = _pair(pilot_rows, tmp_path)
    held = isolation(actual, replay)
    assert held.clean and not held.reported
    assert held.treatment["pos_loan"] == 12 and held.treatment["sales_order_line"] > 0


@pytest.mark.parametrize(
    ("change", "clean", "reported"),
    [
        ("a holdout's prompt", True, {"rep_prompt:holdout": 1}),
        ("a holdout's visit note", True, {"visit.note:holdout": 2}),
        ("a holdout's visit time", False, {}),
        ("a holdout's order", False, {}),
        ("a rep's discount budget", False, {}),
        ("the PoS ledger's register row", True, {}),
        ("another register row", False, {}),
        ("a table in the actual build only", False, {}),
    ],
)
def test_isolation_as_ruled(
    pilot_rows: PilotRows, tmp_path: Path, change: str, clean: bool, reported: dict[str, int]
) -> None:
    """Rep prompts and visit notes outside the programme are reported (Bora's ruling, 2026-11-17); any other
    difference outside the programme and the PoS tables fails, named by its table."""
    actual, replay = _pair(pilot_rows, tmp_path)
    holdout = str(100 * 2 + 2)  # JMV's first holdout
    if change == "a holdout's prompt":
        _append(actual, "rep_prompt", {"rep_prompt_id": "1", "customer_id": holdout, "kind": "ai_tip"})
    if change.startswith("a holdout's visit"):
        column = "note" if change.endswith("note") else "time_in"
        for run, value in ((actual, "Nabídl jsem Gold."), (replay, "Nabídl jsem Classic.")):
            text = (
                value if column == "note" else ("2025-02-03 09:00:00+01" if run == actual else "2025-02-03 10:00:00+01")
            )
            _append(run, "visit", {"visit_id": "77", "customer_id": holdout, "day": "2025-02-03", column: text})
            _append(run, "visit", {"visit_id": "78", "customer_id": holdout, "day": "2025-02-04", column: text})
    if change == "a holdout's order":
        _append(actual, "sales_order", {"sales_order_id": "88", "customer_id": holdout, "ordered_on": "2025-02-03"})
    if change == "a rep's discount budget":
        _append(actual, "discount_budget", {"employee_id": "5", "quarter": "2025-01-01", "budget_amount": "100.00"})
    if change.endswith("register row"):
        code = "MAJ-POS-PHA-2025" if "PoS" in change else "MAJ-IT-2025"
        _append(actual, "asset_register", {"asset_register_id": "9", "code": code, "cost": "1.00"})
    if change == "a table in the actual build only":
        (actual / "shipment.csv").write_text("shipment_id\n1\n", encoding="utf-8")
    held = isolation(actual, replay)
    assert held.clean is clean, held.failures
    assert held.reported == reported


# ── the gate ───────────────────────────────────────────────────────────────────


def _scored(mode: str, status: str, rank_holds: bool, group_holds: bool, truth: str | None = None) -> Scored:
    verdict = {"status": status, "rank_holds": rank_holds, "group_holds": group_holds}
    held = {"status": truth or status, "rank_holds": rank_holds, "group_holds": group_holds}
    return Scored(1, mode, status, [], verdict, held, {}, {}, dict.fromkeys(s5_pilot.SKILL, 0.0), {}, 0)


def _readout(scored: list[Scored], error: float = 0.0) -> Readout:
    bias = {mode: {r: {"mean": error, "rms": abs(error)} for r in s5_pilot.SKILL} for mode in s5_pilot.SCORED}
    return Readout("-", "", [], {}, [], scored, bias)


GOOD = [
    _scored("story", "PASS", True, True),
    _scored("skill-neutral", "REJECTED", False, False),
    _scored("full-null", "REJECTED", False, False),
]


@pytest.mark.parametrize(
    ("scored", "error", "go"),
    [
        (GOOD, 0.05, True),
        (GOOD, 0.11, False),  # a region's bias past 0.10
        ([_scored("story", "REJECTED", True, False, truth="PASS"), *GOOD[1:]], 0.0, False),  # the reference alone
        ([GOOD[0], _scored("skill-neutral", "REJECTED", True, False), GOOD[2]], 0.0, False),  # one half rejected
        ([GOOD[0], GOOD[1], _scored("full-null", "INCONCLUSIVE", False, False)], 0.0, False),  # no valid rejection
    ],
    ids=["go", "bias", "oracle-only success", "control rejects one half", "control inconclusive"],
)
def test_the_gate(scored: list[Scored], error: float, go: bool) -> None:
    reasons = s5_replay._gate(_readout(scored, error), PROTOCOL)
    assert (not reasons) is go, reasons


# ── the whole gate ─────────────────────────────────────────────────────────────


def _tree(pilot_rows: PilotRows, root: Path, spill: bool = False) -> None:
    """Three seeds × five modes of the hand-built pilot: the story's effects steep, the controls' flat, the replays
    without the programme; `spill`: one holdout's order differs in seed 1's story."""
    for seed in PROTOCOL.development:
        for mode, effects in (("story", EFFECTS), ("skill-neutral", FLAT), ("full-null", FLAT)):
            case = _treated(pilot_rows(effects, mode=mode))
            if spill and seed == PROTOCOL.development[0] and mode == "story":
                line = next(
                    r
                    for r in case.rows["sales_order_line"]
                    if r["sales_order_line_id"] // 10 in _orders(case, "PHA", 2)
                )
                line["list_price_case"] += 10.0
            case.meta["master_seed"] = str(seed)
            case.write(root / str(seed) / mode)
        for mode in ("no-pos", "no-pos-null"):
            case = _replay(pilot_rows(NONE, mode=mode))
            case.meta["master_seed"] = str(seed)
            case.write(root / str(seed) / mode)


def test_develop_goes_on_a_story_that_holds(pilot_rows: PilotRows, tmp_path: Path) -> None:
    _tree(pilot_rows, tmp_path)
    readout = develop(tmp_path, PROTOCOL)
    assert readout.decision == "GO", readout.reasons
    assert len(readout.isolation) == 9 and len(readout.scored) == 9
    assert all(s.errors is not None and max(abs(e) for e in s.errors.values()) < 1e-9 for s in readout.scored)
    assert s5_replay.main([str(tmp_path)]) == 0 and (tmp_path / "s5-development.json").is_file()


def test_a_failed_isolation_scores_nothing(pilot_rows: PilotRows, tmp_path: Path) -> None:
    _tree(pilot_rows, tmp_path, spill=True)
    readout = develop(tmp_path, PROTOCOL)
    assert readout.decision == "NO-GO" and not readout.scored
    assert any("isolation: sales_order_line" in reason for reason in readout.reasons)


def test_develop_refuses_builds_of_another_source_or_seed(pilot_rows: PilotRows, tmp_path: Path) -> None:
    _tree(pilot_rows, tmp_path)
    meta = tmp_path / str(PROTOCOL.development[1]) / "full-null" / "_sim_meta.csv"
    meta.write_text(meta.read_text(encoding="utf-8").replace(str(PROTOCOL.development[1]), "1", 1), encoding="utf-8")
    readout = develop(tmp_path, PROTOCOL)
    assert readout.decision == "NO-GO" and not readout.isolation
    assert any("a build of seed 1" in reason for reason in readout.reasons)
