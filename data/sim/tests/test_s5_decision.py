"""S5's placement decision quality (G2 S2.9 T1–T4, `checks.s5_decision`; NH-D25 design09 §§3–5) on conftest's
hand-built decisions — the five behaviour groups: the exact chosen/unchosen comparison, ties at half; a cancelled
selection kept as chosen and W's rounds at their boundaries; complete audit and cost reconciliation, failing closed;
unsupported rounds and regions never passing; observed rows only, the SQL's block the same in both files. Then the
economic disclosure, the verdicts with the controls' rejection, the evaluation's identity and its Czech readout."""

from __future__ import annotations

import itertools
import json
import math
import re
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hartland_sim import evaluation
from hartland_sim.checks import s5_decision
from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.checks.stories import Build
from hartland_sim.config import STORIES_DIR, InventoryConfig
from hartland_sim.evaluation import Evaluation

DecisionRows = Callable[..., Any]  # conftest's `decision_rows`
DAYS = [date(2024 + m // 12, m % 12 + 1, 1) - timedelta(days=1) for m in range(24)]  # W's rounds' decision days
REGIONS = ("PHA", "JMV", "SJC", "ZSC", "VCH", "SMS")
EVALUATION = Evaluation.load()


def _graded(region: str, k: int) -> list[tuple[int, str]]:
    """Scores 1…10, the five ending at PHA's 10, JMV's 9, … SMS's 5 chosen: q 1, 0.8, 0.6, 0.4, 0.2, 0."""
    best = 10 - REGIONS.index(region)
    return [(s, "selected" if best - 4 <= s <= best else "no_budget") for s in range(1, 11)]


def _run(case: Any, tmp_path: Path, audited: bool = True) -> Path:
    run = tmp_path / "run"
    case.write(run, audited)
    return run


def _frames(case: Any, tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    build = Build(_run(case, tmp_path))
    return s5_decision.regions(build), s5_decision.rounds(build)


def _q(rounds: pd.DataFrame, region: str, day: date) -> float:
    return float(rounds["q"].loc[(region, pd.Timestamp(day))])


# ── 1. the exact chosen/unchosen comparison, ties at half ───────────────────


def _brute(pool: list[tuple[int, str]]) -> float:
    chosen = [s for s, o in pool if o in ("selected", "cancelled")]
    unchosen = [s for s, o in pool if o not in ("selected", "cancelled")]
    if not chosen or not unchosen:
        return math.nan
    points = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(chosen, unchosen))
    return points / (len(chosen) * len(unchosen))


def test_q_is_the_share_of_chosen_over_unchosen_comparisons_a_tie_half(
    decision_rows: DecisionRows, tmp_path: Path
) -> None:
    """Hand-counted rounds and 144 random ones with many ties: the twin's q is the pairwise definition's."""
    hand = {
        0: [(3, "selected"), (2, "selected"), (2, "no_budget"), (1, "no_budget")],  # 3.5 of 4
        1: [(4, "selected"), (4, "no_budget"), (4, "no_asset")],  # all tied, both arms: 0.5
        2: [(1, "selected"), (2, "no_budget"), (3, "no_budget")],  # reversed: 0
        3: [(5, "selected"), (6, "selected")],  # nothing unchosen: not comparable
        4: [],  # an empty round: not comparable
    }
    rng = np.random.default_rng(29)
    drawn = {
        (region, k): [
            (int(rng.integers(1, 5)), "selected" if rng.random() < 0.4 else "no_budget")
            for _ in range(int(rng.integers(2, 11)))
        ]
        for region in REGIONS
        for k in range(24)
    }

    def pool(region: str, k: int) -> list[tuple[int, str]]:
        return hand[k] if region == "PHA" and k in hand else drawn[(region, k)]

    _, rounds = _frames(decision_rows(pool), tmp_path)
    assert [_q(rounds, "PHA", DAYS[k]) for k in range(3)] == [0.875, 0.5, 0.0]
    assert math.isnan(_q(rounds, "PHA", DAYS[3])) and math.isnan(_q(rounds, "PHA", DAYS[4]))
    for (region, k), held in drawn.items():
        if region == "PHA" and k in hand:
            continue
        mine, theirs = _q(rounds, region, DAYS[k]), _brute(held)
        assert (math.isnan(mine) and math.isnan(theirs)) or mine == pytest.approx(theirs, abs=1e-12), (region, k)


def test_q_is_an_equal_weight_mean_of_the_comparable_rounds(decision_rows: DecisionRows, tmp_path: Path) -> None:
    def pool(region: str, k: int) -> list[tuple[int, str]]:
        if k % 2:  # a small reversed round: q 0
            return [(1, "selected"), (2, "no_budget")]
        return _graded("PHA", k) if region == "PHA" else _graded(region, k)  # ten candidates: q of the region

    frame, _ = _frames(decision_rows(pool), tmp_path)
    assert frame.loc["PHA", "quality"] == pytest.approx(0.5)  # (12 × 1 + 12 × 0) ÷ 24, not weighted by size
    assert frame.loc["JMV", "quality"] == pytest.approx(0.4)


# ── 2. a cancellation stays chosen; W's rounds at their boundaries ──────────


def test_a_cancelled_selection_stays_chosen_with_its_score(decision_rows: DecisionRows, tmp_path: Path) -> None:
    def pool(region: str, k: int) -> list[tuple[int, str]]:
        if region == "VCH" and k == 6:  # the worst choice cancelled: dropped, q would be 1; unchosen, 1 too
            return [(1, "cancelled"), *((s, "no_budget") for s in range(2, 9)), (9, "selected"), (10, "selected")]
        return _graded(region, k)

    frame, rounds = _frames(decision_rows(pool), tmp_path)
    assert _q(rounds, "VCH", DAYS[6]) == pytest.approx(14 / 21)
    assert frame.loc["VCH", "cancelled"] == 1 and frame.loc["VCH", "selected"] == 117
    assert frame.loc["VCH", "chosen"] == 118 and frame.loc["VCH", "integrity"] == "valid"


def test_the_rounds_are_w_s_24_month_ends_and_the_boundaries_hold(decision_rows: DecisionRows, tmp_path: Path) -> None:
    """2023-12-31's decisions count (effective 2024-01-01); 2023-11-30's and 2025-12-31's do not; every region shows all
    24 rounds, empty or not."""
    case = decision_rows()
    case.rows["pos_placement_decision"] = [
        d for d in case.rows["pos_placement_decision"] if not (d["region_id"] == 2 and d["decision_day"] == DAYS[9])
    ]
    case.rows["pos_loan"] = [
        loan
        for loan in case.rows["pos_loan"]
        if not (200 <= loan["customer_id"] < 300 and loan["lent_on"] == DAYS[9] + timedelta(days=1))
    ]
    frame, rounds = _frames(case, tmp_path)
    assert len(rounds) == 6 * 24 and rounds.index.get_level_values("decision_day").min() == pd.Timestamp(2023, 12, 31)
    assert rounds.index.get_level_values("decision_day").max() == pd.Timestamp(2025, 11, 30)
    assert rounds["decisions"].loc[("JMV", pd.Timestamp(DAYS[9]))] == 0 and math.isnan(_q(rounds, "JMV", DAYS[9]))
    assert frame.loc["PHA", "decisions"] == 24 * 13  # not the 2023-11-30 or 2025-12-31 selections
    assert (frame["rounds"] == 24).all() and frame.loc["JMV", "comparable_rounds"] == 23
    assert set(frame["integrity"]) == {"valid"}


def test_the_pool_keeps_drawn_no_asset_and_no_budget_and_counts_the_rest_apart(
    decision_rows: DecisionRows, tmp_path: Path
) -> None:
    def pool(region: str, k: int) -> list[tuple[int, str]]:
        return [(s, "no_asset" if o == "no_budget" and s <= 2 else o) for s, o in _graded(region, k)]

    frame, rounds = _frames(decision_rows(pool), tmp_path)
    assert frame.loc["PHA", "no_asset"] == 48 and frame.loc["PHA", "no_budget"] == 72  # still unchosen
    assert (frame.loc["PHA", "viable"], frame.loc["PHA", "unchosen"]) == (240, 120)
    for name in ("no_asset_initial", "nonpositive_score", "ineligible"):
        assert frame.loc["PHA", name] == 24  # never in the pool
    assert _q(rounds, "PHA", DAYS[0]) == 1.0


# ── 3. complete audit and cost reconciliation, failing closed ───────────────


def test_the_disclosure_is_the_observed_economics(decision_rows: DecisionRows, tmp_path: Path) -> None:
    """The fixture's figures (DecisionCase's docstring): a warm-up loan's 31 days in W, a loan pending past the last
    day none, a cancellation none, a give-away to a customer with no display counted, the fleet and its idle part."""
    frame, _ = _frames(decision_rows(), tmp_path)
    expected = {
        "programme_customers": 13,
        "observed_margin": 24000.0,
        "asset_use_cost": 7372.0,  # 5 slots × 731 days × 2.00 + 31 × 2.00 (warm-up); the pending loan 0
        "handout_cost": 650.0,  # 13 × 50.00, customer 13's without any display among them
        "programme_cost": 8022.0,
        "no_sales_customers": 1,
        "no_sales_cost": 50.0,
        "fleet_cost": 11696.0,
        "idle_cost": 4324.0,
        "idle_assets": 2,
    }
    for name, value in expected.items():
        assert frame.loc[:, name].tolist() == pytest.approx([value] * 6), name


def _unaudited(case: Any, tmp_path: Path) -> s5_decision.Judged:
    return s5_decision.judge(_run(case, tmp_path, audited=False))


@pytest.mark.parametrize(
    ("tamper", "reason"),
    [
        (lambda run: _drop_line(run / "pos_placement_decision.csv", 7), "pos_placement_decision rows"),
        (
            lambda run: _edit(run / "pos_placement_decision.csv", ",no_budget,", ",selected,"),
            "pos_placement_decision sha256",
        ),
        (lambda run: _drop_line(run / "pos_pilot_assignment.csv", 3), "pos_pilot_assignment rows"),
        (lambda run: (run / "pos_loan.csv").unlink(), "no pos_loan.csv"),
    ],
    ids=["a decision deleted", "a decision changed", "an assignment deleted", "a table missing"],
)
def test_a_changed_export_is_inconclusive(
    decision_rows: DecisionRows, tmp_path: Path, tamper: Callable[[Path], None], reason: str
) -> None:
    run = _run(decision_rows(_graded), tmp_path)
    assert s5_decision.judge(run).status == "PASS"
    tamper(run)
    judged = s5_decision.judge(run)
    assert judged.status == "INCONCLUSIVE" and judged.frame is None
    assert any(reason in why for why in judged.reasons), judged.reasons


def _drop_line(path: Path, n: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:n] + lines[n + 1 :]), encoding="utf-8")


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_a_build_without_its_audit_manifest_is_inconclusive(decision_rows: DecisionRows, tmp_path: Path) -> None:
    judged = _unaudited(decision_rows(_graded), tmp_path)
    assert judged.status == "INCONCLUSIVE" and judged.reasons == ("no audit manifest in build.json",)


def test_a_build_without_build_json_is_inconclusive(decision_rows: DecisionRows, tmp_path: Path) -> None:
    """An interrupted build or a copied run without its `build.json`: a status, named once, never a traceback."""
    run = _run(decision_rows(_graded), tmp_path)
    (run / "build.json").unlink()
    judged = s5_decision.judge(run)
    assert judged.status == "INCONCLUSIVE" and judged.reasons == ("no build.json",)


def test_an_assignment_twice_fails_closed(decision_rows: DecisionRows, tmp_path: Path) -> None:
    """design09 §3's duplicate business key on the assignment: a programme customer's row copied under a new id, which
    the disclosure would count twice. A build fault, so every region names it."""
    case = decision_rows(_graded)
    held = next(a for a in case.rows["pos_pilot_assignment"] if a["customer_id"] == case.customer("PHA", 1))
    case.rows["pos_pilot_assignment"].append(held | {"pos_pilot_assignment_id": 900_101})
    judged = s5_decision.judge(_run(case, tmp_path))
    assert judged.status == "INCONCLUSIVE" and sorted(judged.reasons) == sorted(f"{r}: duplicate" for r in REGIONS)
    assert judged.frame is not None  # the customer counted once all the same
    pha = judged.frame.loc["PHA"]
    assert (pha["programme_customers"], pha["observed_margin"]) == (13, 24000.0)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda c: c.rows["sales_order"][0].update(delivery_point_id=987_654), "an order without its delivery point"),
        (lambda c: c.rows["delivery_point"][0].update(okres_id=987_654), "a delivery point without its okres"),
        (lambda c: c.rows["sales_order_line"][0].update(sku_id=987_654), "an order line without its sku"),
        (lambda c: c.rows["sku"][0].update(product_id=987_654), "a sku without its product"),
        (lambda c: c.rows["product"][0].update(product_line_id=987_654), "a product without its line"),
    ],
    ids=["delivery point", "okres", "sku", "product", "product line"],
)
def test_a_row_sales_pl_drops_is_inconclusive(
    decision_rows: DecisionRows, tmp_path: Path, change: Callable[[Any], None], reason: str
) -> None:
    """`sales_pl` inner-joins each order line's sku, product and line and its order's delivery point and okres: an
    orphan would leave the SQL's observed margin silently, not the twin's."""
    judged = s5_decision.judge(_run(_spoiled(decision_rows(_graded), change), tmp_path))
    assert judged.status == "INCONCLUSIVE" and judged.reasons == (f"{reason} (1)",)


def _spoiled(case: Any, change: Callable[[Any], None]) -> Any:
    change(case)
    return case


def _first(case: Any, region: str = "SJC", day: date = DAYS[4]) -> dict[str, Any]:
    held: dict[str, Any] = case.decisions(region, day)[0]
    return held


@pytest.mark.parametrize(
    ("change", "integrity"),
    [
        (lambda c: _first(c).update(pilot_revision="s5-pilot-v0"), "revision"),
        (lambda c: c.rows["pos_placement_decision"].append(dict(_first(c))), "duplicate"),
        (lambda c: _first(c).update(outcome="chosen"), "outcome"),
        (lambda c: _first(c).update(outcome="selected"), "fields"),  # selected without a loan
        (lambda c: _first(c).update(forecast_gain=366.25), "fields"),  # not its 56 days' margin × fit × H
        (lambda c: _first(c).update(forecast_days=182), "fields"),
        (lambda c: _first(c).update(decision_day=date(2024, 5, 30)), "fields"),  # not a month-end
        (lambda c: _first(c).update(manager_id=777), "links"),
        (lambda c: _first(c).update(customer_id=c.customer("SJC", 51)), "links"),  # a holdout's decision
        (lambda c: c.rows["pos_loan"].pop(), "links"),  # the pending selection's loan gone
        (lambda c: _first(c, day=DAYS[5]).update(outcome="cancelled", score=0.0), "fields"),
        (lambda c: c.rows["pos_asset"].append(c.rows["pos_asset"][0] | {"pos_asset_id": 77}), "valid"),
    ],
    ids=[
        "another revision",
        "a decision twice",
        "an unknown outcome",
        "selected without a loan",
        "a gain not its inputs'",
        "another H",
        "not a month-end",
        "an unknown manager",
        "a holdout's decision",
        "a selection's loan missing",
        "a cancellation without its pool's score",
        "an unused asset is only idle",
    ],
)
def test_an_impossible_decision_fails_closed(
    decision_rows: DecisionRows, tmp_path: Path, change: Callable[[Any], None], integrity: str
) -> None:
    frame, _ = _frames(_spoiled(decision_rows(_graded), change), tmp_path)
    assert set(frame["integrity"]) == {integrity}
    if integrity != "valid":
        assert math.isnan(s5_decision.rank(frame)) and math.isnan(s5_decision.quality(frame)[2])
        assert s5_decision.verdict(frame, EVALUATION).status == "INCONCLUSIVE"


def test_a_generator_decision_with_its_rounding_is_sound(decision_rows: DecisionRows, tmp_path: Path) -> None:
    """The generator rounds floats half to even and the stored values carry two (gain, cost) or eight (score)
    decimals: a value within half its last digit of its inputs is sound, one a cent further is not."""
    case = decision_rows()
    held = _first(case, "PHA", DAYS[0])  # score 1: gain 365.25 = 112.00 ÷ 56 × 1 × 182.625
    held.update(prior56_line_margin=112.03)  # 365.34784… → its gain may be 365.35, not 365.25
    frame, _ = _frames(case, tmp_path)
    assert frame.loc["PHA", "integrity"] == "fields"
    held.update(forecast_gain=365.35, score=round(365.35 / 365.25, 8))
    frame, _ = _frames(case, tmp_path / "rounded")
    assert frame.loc["PHA", "integrity"] == "valid"


# ── 4. unsupported rounds and regions never pass ────────────────────────────


@pytest.mark.parametrize(
    "spoil",
    ["eleven comparable rounds", "99 chosen", "99 unchosen", "a region without programme", "five regions"],
)
def test_an_unsupported_region_or_build_cannot_pass(decision_rows: DecisionRows, tmp_path: Path, spoil: str) -> None:
    def pool(region: str, k: int) -> list[tuple[int, str]]:
        held = _graded(region, k)  # SMS: 1…5 chosen, 6…10 not
        if region != "SMS":
            return held
        if spoil == "eleven comparable rounds" and k >= 11:
            return held[:5]
        if spoil == "99 chosen" and k <= 20:
            return held[1:]
        if spoil == "99 unchosen" and k <= 20:
            return held[:-1]
        return held

    case = decision_rows(pool)
    if spoil == "a region without programme":
        case.rows["pos_pilot_assignment"] = [
            a | {"arm": "holdout"} if a["region_id"] == 6 and a["arm"] == "programme" else a
            for a in case.rows["pos_pilot_assignment"]
        ]
        for table in ("pos_placement_decision", "pos_loan", "visit"):
            case.rows[table] = [row for row in case.rows[table] if not 600 <= row["customer_id"] < 700]
        case.rows["pos_handout"] = [row for row in case.rows["pos_handout"] if not 600 <= row["visit_id"] < 700]
    if spoil == "five regions":
        case.rows["region"] = case.rows["region"][:5]
    frame, _ = _frames(case, tmp_path)
    found = s5_decision.verdict(frame, EVALUATION)
    assert found.status == "INCONCLUSIVE" and not found.control_rejects
    assert math.isnan(found.rank) and math.isnan(found.difference)
    short = {"eleven comparable rounds": "comparable_rounds", "99 chosen": "chosen", "99 unchosen": "unchosen"}
    if spoil in short:
        assert frame.loc["SMS", "integrity"] == "support" and (frame.drop(index="SMS")["integrity"] == "valid").all()
        assert frame.loc["SMS", short[spoil]] == {"comparable_rounds": 11}.get(short[spoil], 99)
    if spoil == "a region without programme":
        assert set(frame["integrity"]) == {"regions"} and frame.loc["SMS", "programme_customers"] == 0


# ── 5. observed rows only; one block ────────────────────────────────────────


def _block(name: str) -> str:
    text = (STORIES_DIR / name).read_text(encoding="utf-8")
    return text[text.index("-- region quality: begin") : text.index("-- region quality: end")]


def test_the_two_files_share_one_block_that_reads_observed_rows_only() -> None:
    rank, difference = (_block(name) for name in s5_decision.SQL)
    assert rank == difference
    code = "\n".join(line for line in rank.splitlines() if not line.startswith("--")).lower()
    for hidden in ("skill", "uplift", "truth", "replay", "no_pos", "oracle"):
        assert hidden not in code, hidden
    inventory = InventoryConfig.load()
    known = {table.name for table in inventory.tables} | {view.name for view in inventory.views}
    named = set(re.findall(r"\b(?:from|join)\s+(\w+)\b(?!\.|\()", code))
    ctes = set(re.findall(r"(?:with|,)\s*(\w+)(?: \([^)]*\))? as \(", code)) | {"evaluation"}
    assert named - ctes <= known
    assert set(s5_decision.TABLES) <= known
    source = Path(s5_decision.__file__).read_text(encoding="utf-8")
    assert "StoriesConfig" not in source and "display_uplift" not in source and "pos_skill" not in source


def test_the_twin_s_constants_are_the_evaluation_s_and_the_sql_s() -> None:
    held = EVALUATION
    assert (held.pilot_revision, held.policy_revision) == (s5_decision.REVISION, s5_decision.POLICY)
    assert (held.window_from, held.window_until) == (s5_decision.FIRST.date(), s5_decision.LAST.date())
    assert held.support == {"comparable_rounds": 12, "chosen": 100, "unchosen": 100}
    assert held.rounds == len(DAYS) == 24
    assert (held.acceptance["high"], held.acceptance["low"]) == (list(s5_decision.HIGH), list(s5_decision.LOW))
    row = "('s5-pilot-v1', 'nh-d21-b-r1', DATE '2024-01-01', DATE '2025-12-31', 56, 182.625, 183, 36 * 30.4, "
    row += "12, 100, 100)"
    assert row in _block(s5_decision.SQL[0])
    skills = re.findall(r"\('(\w+)', ([\d.]+)\)", (STORIES_DIR / s5_decision.SQL[0]).read_text(encoding="utf-8"))
    assert {code: float(v) for code, v in skills} == s5_decision.SKILL
    codes = "', '".join(s5_decision.SKILL)  # the difference's validity: the six regions, not any six
    assert f"region IN ('{codes}')" in (STORIES_DIR / s5_decision.SQL[1]).read_text(encoding="utf-8")


# ── the verdicts ─────────────────────────────────────────────────────────────


def test_a_graded_story_passes_and_its_q_are_the_choices(decision_rows: DecisionRows, tmp_path: Path) -> None:
    judged = s5_decision.judge(_run(decision_rows(_graded), tmp_path))
    assert judged.status == "PASS" and judged.reasons == ()
    assert judged.frame is not None and judged.verdict is not None
    assert judged.frame["quality"].tolist() == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    v = judged.verdict
    assert (v.q_high, v.q_low, v.difference) == pytest.approx((0.9, 0.2, 0.7))
    assert v.rank == pytest.approx(0.9856, abs=1e-4) and not v.control_rejects


def _verdict(q: dict[str, float]) -> s5_decision.Verdict:
    frame = pd.DataFrame({"quality": pd.Series(q), "integrity": "valid"})
    return s5_decision.verdict(frame, EVALUATION)


@pytest.mark.parametrize(
    ("q", "status", "control_rejects"),
    [
        ({"PHA": 0.9, "JMV": 0.8, "SJC": 0.7, "ZSC": 0.6, "VCH": 0.6, "SMS": 0.6}, "PASS", False),
        ({"PHA": 0.69, "JMV": 0.69, "SJC": 0.6, "ZSC": 0.5, "VCH": 0.4, "SMS": 0.4}, "REJECTED", False),  # Q_H
        ({"PHA": 0.8, "JMV": 0.7, "SJC": 0.72, "ZSC": 0.7, "VCH": 0.69, "SMS": 0.66}, "REJECTED", False),  # D < 0.10
        ({"PHA": 0.6, "JMV": 0.9, "SJC": 0.5, "ZSC": 0.8, "VCH": 0.4, "SMS": 0.3}, "REJECTED", False),  # D but rank
        ({"PHA": 0.7, "JMV": 0.7, "SJC": 0.7, "ZSC": 0.7, "VCH": 0.7, "SMS": 0.7}, "REJECTED", True),  # rank 0
        ({"PHA": 0.5, "JMV": 0.6, "SJC": 0.7, "ZSC": 0.5, "VCH": 0.6, "SMS": 0.55}, "REJECTED", True),
    ],
)
def test_a_story_needs_every_bar_and_a_control_rejects_on_rank_and_difference(
    q: dict[str, float], status: str, control_rejects: bool
) -> None:
    found = _verdict(q)
    assert (found.status, found.control_rejects) == (status, control_rejects)


def test_an_invalid_frame_neither_passes_nor_rejects() -> None:
    frame = pd.DataFrame({"quality": pd.Series(dict.fromkeys(REGIONS, 0.9)), "integrity": "valid"})
    frame.loc["SMS", "integrity"] = "support"
    found = s5_decision.verdict(frame, EVALUATION)
    assert found.status == "INCONCLUSIVE" and not found.control_rejects and math.isnan(found.rank)


# ── the evaluation's identity and its Czech readout ─────────────────────────


@pytest.mark.parametrize(
    ("key", "value", "why"),
    [
        ("s5_evaluation_sha256", "0" * 64, "stale s5_evaluation_sha256"),
        ("s5_claim", "incremental_return", "stale s5_claim"),
        ("s5_evaluation_revision", None, "no s5_evaluation_revision"),
        ("pilot_mode", "no-pos", "lends nothing"),
        ("last_day_done", "2025-12-30", "short"),
    ],
)
def test_a_build_of_another_evaluation_or_short_is_not_certified(
    decision_rows: DecisionRows, tmp_path: Path, key: str, value: str | None, why: str
) -> None:
    case = decision_rows(_graded)
    if value is None:
        del case.meta[key]
    else:
        case.meta[key] = value
    judged = s5_decision.judge(_run(case, tmp_path))
    assert judged.status == "NOT_CERTIFIED" and judged.frame is None
    assert any(why in reason for reason in judged.reasons), judged.reasons


def test_the_evaluation_identity_is_the_file_s() -> None:
    held = dict(evaluation.identity())
    assert held["s5_evaluation_revision"] == "NH-D25/v1" and held["s5_claim"] == "placement_decision_quality"
    assert len(held["s5_evaluation_sha256"]) == 64 and list(held) == list(evaluation.META_KEYS)


def test_the_readout_keeps_forecast_and_observed_apart_in_czech(decision_rows: DecisionRows, tmp_path: Path) -> None:
    run = _run(decision_rows(_graded), tmp_path)
    text = s5_decision.markdown(s5_decision.judge(run))
    labels = EVALUATION.labels
    assert text.startswith(f"# {labels['title_cs']}: PASS")
    for name in ("disclosure_cs", "forecast_cs", "observed_cs", "unavailable_cs"):
        assert labels[name] in text
    assert "24 000,00" in text and "7 372,00" in text  # Czech money
    assert_czech_all(list(labels.values()), what="the evaluation's labels")
    held = json.loads((run / "build.json").read_text(encoding="utf-8"))
    assert set(held["audit"]) == {"pos_pilot_assignment", "pos_placement_decision"}
