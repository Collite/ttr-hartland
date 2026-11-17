"""S5's observed pilot evaluation (G2 S2.7 T1–T2, `checks.s5_pilot`; design08 §§6, 8) on conftest's hand-built pilot
(`pilot_rows`): a known gain under any common drift with the baseline's coefficient fixed at 1; acquisition, loss and
zero sales kept in the outcome; repeated loans that neither repeat the outcome nor the cost; the six regions, coverage,
contamination and missing rows failing closed; negative gains, tied ranks and the economic sign. SQL/twin parity on
the same rows is test_s5_pilot_component's."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.checks import s5_pilot
from hartland_sim.checks.s5_pilot import judge, regions, verdict
from hartland_sim.checks.stories import Build
from hartland_sim.config import STORIES_DIR, StoriesConfig, WorldConfig
from hartland_sim.protocol import Protocol

PilotRows = Callable[..., Any]  # conftest's `pilot_rows`
EFFECTS = {"PHA": 8000.0, "JMV": 6000.0, "SJC": 4000.0, "ZSC": 2000.0, "VCH": -1000.0, "SMS": 1000.0}
PROTOCOL = Protocol.load()


def _judged(case: Any, run: Path) -> s5_pilot.Judged:
    case.write(run)
    return judge(run, PROTOCOL)


# ── T1 (a): the known gain under a common drift, the baseline's coefficient fixed ──────────────────────────────────


@pytest.mark.parametrize("drift", [-20000.0, 0.0, 15000.0])
def test_a_known_gain_whatever_the_common_drift(pilot_rows: PilotRows, tmp_path: Path, drift: float) -> None:
    """Every customer's margin drifts alike from its own baseline × 731: a region's gain is its pairs' effects, to the
    haléř, whatever the drift, and its cost the programme's displays and give-aways."""
    judged = _judged(pilot_rows(EFFECTS, drift=drift), tmp_path)
    frame = judged.frame
    assert frame is not None and judged.status == "PASS"
    for region, effect in EFFECTS.items():
        assert frame.at[region, "gain"] == pytest.approx(2 * effect, abs=0.005)
        assert frame.at[region, "cost"] == pytest.approx(8000.0, abs=0.005)
        assert frame.at[region, "asset_cost"] == pytest.approx(7320.0, abs=0.005)
    assert (frame["baseline_adjustment"] == 0).all()  # each pair's two baselines are equal


def test_the_baseline_coefficient_is_one_and_the_stored_row_is_read(pilot_rows: PilotRows, tmp_path: Path) -> None:
    """Raise one holdout's stored baseline by 56.00 (1.00 a day): its expected margin rises by exactly 731.00, so
    its region's gain rises by 731.00 — the stored row, never a baseline rebuilt from the exported sales (which are
    unchanged), with W's days × the baseline a day and no fitted coefficient."""
    case = pilot_rows(EFFECTS)
    holdout = next(r for r in case.rows["pos_pilot_assignment"] if r["customer_id"] == case.customer("JMV", 2))
    holdout["baseline_customer_margin"] += 56.0
    frame = _judged(case, tmp_path).frame
    assert frame is not None
    assert frame.at["JMV", "baseline_adjustment"] == pytest.approx(-731.0, abs=1e-6)
    assert frame.at["JMV", "gain"] == pytest.approx(2 * EFFECTS["JMV"] + 731.0, abs=0.005)


def test_the_margin_is_sales_pl_s_to_the_haler(pilot_rows: PilotRows, tmp_path: Path) -> None:
    """Each customer's second order holds 10.10 × 0.250 = 2.525, which `sales_pl` rounds half away from zero to
    2.53 (a float's round gives 2.52): the outcomes are exact, so the holdouts' sum is 2 × 73 100 + 2 × 146 200 + 4
    drifts of 1 000 to the haléř."""
    frame = _judged(pilot_rows(EFFECTS), tmp_path).frame
    assert frame is not None
    assert (frame["holdout_margin"] == 73100.0 + 146200.0 + 2 * 1000.0).all()


# ── T1 (b): acquisition, loss and zero sales stay in the outcome ───────────────────────────────────────────────────


def test_acquisition_loss_and_zero_sales_stay_in_the_outcome(pilot_rows: PilotRows, tmp_path: Path) -> None:
    """A holdout that never orders in W counts with margin 0 (no order is a zero-sales day, not a missing row); a
    programme customer potential at entry and one lost during W keep their arms and their days: the gain moves by
    exactly their margins, and the cohort counts them."""
    case = pilot_rows(EFFECTS)
    silent = case.customer("SJC", 4)  # pair 2's holdout: 73 100 × 2 + 1 000 expected from its sales
    orders = {o["sales_order_id"] for o in case.rows["sales_order"] if o["customer_id"] == silent}
    case.rows["sales_order"] = [o for o in case.rows["sales_order"] if o["sales_order_id"] not in orders]
    gone = {r["sales_order_line_id"] for r in case.rows["sales_order_line"] if r["sales_order_id"] in orders}
    case.rows["sales_order_line"] = [r for r in case.rows["sales_order_line"] if r["sales_order_line_id"] not in gone]
    case.rows["sales_order_line_discount"] = [
        r for r in case.rows["sales_order_line_discount"] if r["sales_order_line_id"] not in gone
    ]
    for key in ("orders", "lines"):  # the build wrote them so: the manifest agrees
        case.quarters["2024-Q1"][key] -= 1
        case.quarters["2025-Q4"][key] -= 2 if key == "lines" else 1
    customers = {c["customer_id"]: c for c in case.rows["customer"]}
    customers[case.customer("ZSC", 1)]["since"] = None  # potential at entry, never won
    customers[case.customer("ZSC", 3)]["until"] = date(2024, 7, 31)  # lost in W: its later zeros count
    frame = _judged(case, tmp_path).frame
    assert frame is not None and (frame["integrity"] == "valid").all()
    assert frame.at["SJC", "gain"] == pytest.approx(2 * EFFECTS["SJC"] + 147200.0, abs=0.005)
    assert frame.at["SJC", "pairs"] == 2 and frame.at["SJC", "holdout_margin"] == 74100.0
    assert (frame.at["ZSC", "potential_at_entry"], frame.at["ZSC", "lost"]) == (1, 1)


# ── T1 (c): repeated loans repeat neither the outcome nor the cost ─────────────────────────────────────────────────


def test_repeated_loans_add_their_days_not_the_outcome(pilot_rows: PilotRows, tmp_path: Path) -> None:
    """PHA's first programme customer borrows the same asset again on the day it came back, to W's end: the asset's
    days are counted once each (its first loan's end is the day before it came back), the outcome once per
    customer, and a placement pending past the last day costs nothing."""
    case = pilot_rows(EFFECTS)
    c = case.customer("PHA", 1)
    loan = next(r for r in case.rows["pos_loan"] if r["customer_id"] == c)
    case.rows["pos_loan"].append(loan | {"pos_loan_id": 9001, "lent_on": date(2025, 1, 1), "returned_on": None})
    case.rows["pos_loan"].append(loan | {"pos_loan_id": 9002, "lent_on": date(2026, 1, 1), "returned_on": None})
    frame = _judged(case, tmp_path).frame
    assert frame is not None
    assert frame.at["PHA", "asset_cost"] == pytest.approx(7320.0 + 3650.0, abs=0.005)  # 366 + 365 days of 10.00
    assert frame.at["PHA", "gain"] == pytest.approx(2 * EFFECTS["PHA"], abs=0.005)
    assert frame.at["PHA", "integrity"] == "valid"


def test_a_programme_customer_without_a_display_keeps_its_giveaways(pilot_rows: PilotRows, tmp_path: Path) -> None:
    """Every programme give-away in W is a cost, at a customer with no loan as at one with: drop VCH's first display
    and its asset cost falls by its 3 660.00, its give-away cost stays 680.00."""
    case = pilot_rows(EFFECTS)
    case.rows["pos_loan"] = [r for r in case.rows["pos_loan"] if r["customer_id"] != case.customer("VCH", 1)]
    frame = _judged(case, tmp_path).frame
    assert frame is not None
    assert frame.at["VCH", "asset_cost"] == pytest.approx(3660.0, abs=0.005)
    assert frame.at["VCH", "handout_cost"] == pytest.approx(680.0, abs=0.005)


def test_an_asset_out_twice_on_one_day_is_an_integrity_failure(pilot_rows: PilotRows, tmp_path: Path) -> None:
    """A second loan of the same asset that overlaps the first by one day is a duplicate, never deduplicated: every
    region says `overlap` and the build is INCONCLUSIVE."""
    case = pilot_rows(EFFECTS)
    loan = case.rows["pos_loan"][0]
    case.rows["pos_loan"].append(loan | {"pos_loan_id": 9001, "lent_on": date(2024, 12, 31), "returned_on": None})
    judged = _judged(case, tmp_path)
    assert judged.status == "INCONCLUSIVE" and judged.frame is not None
    assert set(judged.frame["integrity"]) == {"overlap"} and math.isnan(judged.frame.pipe(s5_pilot.rank))


# ── T1 (d): six regions, coverage, contamination and missing data fail closed ──────────────────────────────────────


def _drop(case: Any, table: str, keep: Callable[[dict[str, Any]], bool]) -> None:
    case.rows[table] = [r for r in case.rows[table] if keep(r)]


def _region_without_pairs(case: Any) -> None:
    """SMS assigned no one (and so lending and handing out nothing): five regions' rows are not six."""
    sms = {r["customer_id"] for r in case.rows["pos_pilot_assignment"] if r["region_id"] == 6}
    _drop(case, "pos_pilot_assignment", lambda r: r["customer_id"] not in sms)
    _drop(case, "pos_loan", lambda r: r["customer_id"] not in sms)
    _drop(case, "visit", lambda r: r["customer_id"] not in sms)
    _drop(case, "pos_handout", lambda r: r["visit_id"] not in sms)


def _holdout_with_a_display(case: Any) -> None:
    holdout = case.customer("PHA", 2)
    case.rows["pos_asset"].append({"pos_asset_id": 9001, "region_id": 1, "cost": 10944.0})
    case.rows["pos_loan"].append(
        {"pos_loan_id": 9001, "pos_asset_id": 9001, "customer_id": holdout, "store_id": holdout}
        | {"lent_on": date(2024, 3, 1), "returned_on": None}
    )


def _outside_with_a_giveaway(case: Any) -> None:
    alone = 9002  # an unpaired customer of PHA
    case.rows["customer"].append({"customer_id": alone, "channel": "small_retail", "since": None, "until": None})
    case.rows["store"].append({"store_id": alone, "area_id": 1})
    row = case.rows["pos_pilot_assignment"][0]
    case.rows["pos_pilot_assignment"].append(
        row
        | {"pos_pilot_assignment_id": alone, "customer_id": alone, "pair_id": None, "arm": "outside"}
        | {"outside_reason": "unpaired"}
    )
    case.rows["visit"].append({"visit_id": alone, "customer_id": alone, "store_id": alone, "day": date(2024, 6, 3)})
    case.rows["pos_handout"].append({"pos_handout_id": alone, "visit_id": alone, "cost": 340.0})


def _assignment_row_deleted(case: Any) -> None:
    _drop(case, "pos_pilot_assignment", lambda r: r["customer_id"] != case.customer("VCH", 2))


def _key_account_display(case: Any) -> None:
    """A gas station's display on a ZSC asset all through W: no contamination (another channel), but most of the
    region's loan cost is outside its pairs."""
    case.rows["customer"].append({"customer_id": 9003, "channel": "gas_ka", "since": None, "until": None})
    case.rows["pos_asset"].append({"pos_asset_id": 9003, "region_id": 4, "cost": 50000.0})
    case.rows["pos_loan"].append(
        {"pos_loan_id": 9003, "pos_asset_id": 9003, "customer_id": 9003, "store_id": None}
        | {"lent_on": date(2024, 1, 1), "returned_on": None}
    )


@pytest.mark.parametrize(
    ("spoil", "integrity"),
    [
        (_region_without_pairs, "regions"),
        (_holdout_with_a_display, "contamination"),
        (_outside_with_a_giveaway, "contamination"),
        (_assignment_row_deleted, "pairs"),
        (_key_account_display, "coverage"),
    ],
)
def test_the_frame_fails_closed(
    pilot_rows: PilotRows, tmp_path: Path, spoil: Callable[[Any], None], integrity: str
) -> None:
    """Each spoils the pilot one way: the build is INCONCLUSIVE, never a pass, the frame says why, and both metrics
    are NaN (the SQL's NULL)."""
    case = pilot_rows(EFFECTS)
    spoil(case)
    judged = _judged(case, tmp_path)
    assert judged.status == "INCONCLUSIVE"
    assert judged.frame is not None and integrity in set(judged.frame["integrity"])
    assert judged.verdict is not None and math.isnan(judged.verdict.rank) and math.isnan(judged.verdict.ratio)


@pytest.mark.parametrize(
    ("name", "spoil", "status"),
    [
        ("an order deleted", "order", "INCONCLUSIVE"),
        ("an order and its lines deleted", "order+lines", "INCONCLUSIVE"),
        ("the give-aways' table missing", "file", "INCONCLUSIVE"),
        ("a build short of W", "short", "NOT_CERTIFIED"),
        ("another protocol", "stale", "NOT_CERTIFIED"),
        ("a no-PoS reference", "no-pos", "NOT_CERTIFIED"),
    ],
)
def test_missing_or_stale_sources_never_pass(
    pilot_rows: PilotRows, tmp_path: Path, name: str, spoil: str, status: str
) -> None:
    """A missing export is never a zero: a deleted order is caught by its lines or by `build.json`'s manifest, a
    missing file by name; a short or stale build, or a reference mode, is not certified."""
    case = pilot_rows(EFFECTS, mode="no-pos" if spoil == "no-pos" else "story")
    holdout = case.customer("JMV", 2)
    order = next(o["sales_order_id"] for o in case.rows["sales_order"] if o["customer_id"] == holdout)
    if spoil.startswith("order"):
        _drop(case, "sales_order", lambda r: r["sales_order_id"] != order)
    if spoil == "order+lines":
        _drop(case, "sales_order_line", lambda r: r["sales_order_id"] != order)
    if spoil == "short":
        case.meta["last_day_done"] = "2025-06-30"
    if spoil == "stale":
        case.meta["pilot_protocol_sha256"] = "0" * 64
    case.write(tmp_path)
    if spoil == "file":
        (tmp_path / "pos_handout.csv").unlink()
    judged = judge(tmp_path, PROTOCOL)
    assert judged.status == status, name
    assert judged.reasons, name


# ── T1 (e): negative gains, tied ranks and the economic sign ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("effects", "status", "rho", "ratio"),
    [
        (EFFECTS, "PASS", 0.98561076, 10.5),
        (
            {"PHA": -1000.0, "JMV": -2000.0, "SJC": 1000.0, "ZSC": 3000.0, "VCH": 4000.0, "SMS": 4000.0},
            "REJECTED",
            None,
            None,
        ),
        (dict.fromkeys(EFFECTS, 2000.0), "REJECTED", 0.0, 1.0),
        (
            {"PHA": 9000.0, "JMV": 8000.0, "SJC": 4000.0, "ZSC": 0.0, "VCH": -2000.0, "SMS": -1000.0},
            "REJECTED",
            None,
            math.nan,
        ),
    ],
    ids=["story", "negative gains in the high group", "six equal returns", "weak group not positive"],
)
def test_ranks_ties_and_the_economic_sign(
    pilot_rows: PilotRows,
    tmp_path: Path,
    effects: dict[str, float],
    status: str,
    rho: float | None,
    ratio: float | None,
) -> None:
    """Average ranks (VCH and SMS share their skill, and here their return too); six equal returns are no association,
    0; a weak group whose mean gain ÷ cost is not positive has no ratio (NaN, the SQL's NULL) and the economic claim
    is rejected, however steep the rank."""
    judged = _judged(pilot_rows(effects), tmp_path)
    found = judged.verdict
    assert found is not None and judged.status == status
    if rho is not None:
        assert found.rank == pytest.approx(rho, abs=1e-8)
    if ratio is not None and not math.isnan(ratio):
        assert found.ratio == pytest.approx(ratio)
    if ratio is not None and math.isnan(ratio):
        assert math.isnan(found.ratio) and not found.economic and not found.group_holds and found.rank_holds


def test_the_claim_needs_both_halves_and_the_high_group_s_net_return(pilot_rows: PilotRows, tmp_path: Path) -> None:
    """Steep rank and ratio, but the high group's gain ÷ cost ≤ 1 (a negative net return): REJECTED."""
    effects = {"PHA": 3600.0, "JMV": 3400.0, "SJC": 2000.0, "ZSC": 1000.0, "VCH": 800.0, "SMS": 900.0}
    found = _judged(pilot_rows(effects), tmp_path).verdict
    assert found is not None and found.rank_holds and found.ratio > 1.35
    assert found.e_high < 1 and not found.economic and found.status == "REJECTED"


# ── T2: the observed tables only; the constants are the protocol's ─────────────────────────────────────────────────


def test_the_twin_reads_the_observed_tables_only(pilot_rows: PilotRows, tmp_path: Path) -> None:
    """The judge reads `TABLES` (by whole or by column), `_sim_meta.csv` and `build.json` — no skill, uplift,
    decision, checkpoint or other world; with every other file removed it gives the same frame."""
    case = pilot_rows(EFFECTS)
    case.write(tmp_path)
    for path in tmp_path.glob("*.csv"):
        if path.stem not in (*s5_pilot.TABLES, "_sim_meta"):
            path.unlink()
    build = Build(tmp_path)
    judged = judge(tmp_path, PROTOCOL, build)
    assert judged.status == "PASS"
    assert build._frames is not None
    assert {key.split("[")[0] for key in build._frames} <= set(s5_pilot.TABLES)


def test_the_constants_are_the_protocol_s_and_the_story_s() -> None:
    world, stories = WorldConfig.load(), StoriesConfig.load()
    bars = PROTOCOL.acceptance
    assert s5_pilot.REVISION == PROTOCOL.pilot_revision and s5_pilot.COVERAGE == bars["coverage"]
    assert s5_pilot.LIFE_DAYS == world.field.pos_asset_life_months * PROTOCOL.asset_days_per_month
    assert s5_pilot.SKILL == stories.story("S5").parameters["pos_skill"]
    assert (list(s5_pilot.HIGH), list(s5_pilot.LOW)) == (bars["high"], bars["low"])
    assert s5_pilot.SCORED == tuple(m for m in PROTOCOL.modes if not m.startswith("no-pos"))


def test_verdict_takes_any_frame(pilot_rows: PilotRows, tmp_path: Path) -> None:
    """`verdict` judges a frame whose gains were replaced (the reference's truth-based scoring) the same way."""
    frame = regions(Build(_written(pilot_rows(EFFECTS), tmp_path)))
    truth = frame.assign(gain=frame["gain"] * -1)
    truth["gain_per_cost"] = truth["gain"] / truth["cost"]
    assert verdict(frame, PROTOCOL).status == "PASS" and verdict(truth, PROTOCOL).status == "REJECTED"


def _written(case: Any, run: Path) -> Path:
    case.write(run)
    return run


def test_the_pilot_sql_shares_one_block_and_the_twin_s_constants() -> None:
    """Both pilot files read one region return, byte for byte; their literals are the twin's (and so, above, the
    protocol's and the story's): the revision, the asset days, the coverage, the skills and the two groups."""
    rank, group = ((STORIES_DIR / name).read_text(encoding="utf-8") for name in s5_pilot.SQL)

    def block(sql: str) -> str:
        found = re.search(r"^-- region return: begin\n.*^-- region return: end\n", sql, re.S | re.M)
        assert found
        return found.group(0)

    assert block(rank) == block(group)
    revision, life, coverage = s5_pilot.REVISION, "36 * 30.4", s5_pilot.COVERAGE
    assert f"VALUES ('{revision}', {life}, {coverage:.2f})" in block(rank) and s5_pilot.LIFE_DAYS == 36 * 30.4
    skills = dict(re.findall(r"\('([A-Z]{3})', ([0-9.]+)\)", rank.split("-- region return: begin")[0]))
    assert {code: float(value) for code, value in skills.items()} == s5_pilot.SKILL
    groups = re.findall(r"region IN \(([^)]*)\)", group.split("-- region return: end")[1])
    assert [tuple(code.strip(" '") for code in found.split(",")) for found in groups] == [s5_pilot.HIGH, s5_pilot.LOW]
