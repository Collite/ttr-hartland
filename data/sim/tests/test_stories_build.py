"""The stories on the build tier (`pytest -m build`, its own CI job): G1 S1.2's DONE line (T7) and S1.4's S6 (T6) on
the story and the null world, `build-world` from the warm-up to 2025-12-31 (`conftest.story_world`, `null_world`; S1's
window is 2024–2025, S4's 2025). The warm-up writes nothing before 2024-01-01 and 2024 starts with a year of history;
the S1, S2 and S4 twins pass on the story world and fail on the null one.

**S6** (S1.4 T6, re-cut in S1.5 T5): per rep, the Customer Margin gained per agreed-discount crown
(`checks.stories.gains`, s6.sql's measure) ranks with its hidden `discount_discipline` (the answer key: the story
world's checkpoint) — Spearman ≥ 0.8 over all reps, the null world's |ρ| < 0.3 — and `04` S6's bounds on the pinned reps
(review-196 ⚑D): the disciplined ones' ratio > 1, the undisciplined ones' < 0.5; s6.sql's twin passes on the story world
and fails on the null one. The story world falls short of three of the four bars (the pinned undisciplined pass, 0.37 on
G2 S2.1's build after review-231, 0.45 on S1.5's, inside the noise either way): the lift — a ratio of the few cases an
agreement's windows hold — scatters far wider than the discount moves it, so strict expected failures hold the numbers
and the PR's Found line carries them (the list's stop condition: S1.5 does not solve it; review-225 ⚑R225-A).
**The field** (G2 S2.1 T7): A-7's and A-8's probes find the pinned reps, S3's first rate — the sales visits'
conversion — ranks with the hidden skill, the field's probes pass and find nothing in the null world, the visits' notes
read as Czech, and small retail's own orders keep the trade week (a visit's order stands in for the store's next one:
review-231 ⚑R231-A)."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.checks.stories import S6_FROM, Build, Verdict, a6_arc, a6_bars, check, gains, s6_reps
from hartland_sim.cli import main
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.reference import read as read_reference
from hartland_sim.loop.build import Built
from hartland_sim.loop.field import PLANTED_SIGHTINGS
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint

pytestmark = pytest.mark.build

JUDGED = ("detectors/s1.sql", "detectors/s1_weather.sql", "detectors/s2_affluence.sql", "detectors/s4.sql")
WARM_UP_END = date(2023, 12, 31)


def _first(path: Path, column: str) -> str:
    with path.open(encoding="utf-8", newline="") as source:
        return min(row[column] for row in csv.DictReader(source))


def _judged(verdicts: list[Verdict]) -> dict[str, Verdict]:
    return {v.check: v for v in verdicts if v.check in JUDGED}


def test_the_warm_up_writes_nothing_before_2024(story_world: Built) -> None:
    assert _first(story_world.run / "sales_order.csv", "ordered_on") == "2024-01-01"
    assert _first(story_world.run / "weather_daily.csv", "day") == "2024-01-01"
    assert _first(story_world.run / "price_agreement.csv", "valid_from") >= "2024-01-01"


def test_2024_starts_with_a_year_of_history(story_world: Built, opening: Setup) -> None:
    kept = Checkpoint.load(story_world.run, WARM_UP_END).restore()
    potential = {c.customer_id for c in opening.world.customers.values() if c.status == "potential"}
    gained = sum(c.customer_id in potential and c.status != "potential" for c in kept.customers.values())
    lost = sum(c.until is not None and c.until.year == 2023 for c in kept.customers.values())
    assert gained > 1000  # 0.03 a month of ~5 000 potential stores
    assert lost > 300  # 0.012 a month of ~3 500 customers
    demand = kept.demand
    assert demand is not None
    today = date(2024, 1, 1).toordinal()
    buying = (demand.customer_since <= today) & (today <= demand.customer_until)
    assert float((demand.basket[buying[demand.point_customer]].sum(axis=1) > 0).mean()) > 0.9
    assert {c.phase_on(WARM_UP_END) for c in kept.customers.values() if c.channel == "small_retail"} >= {
        "occasional",
        "regular",
        "lost",
    }
    with (story_world.run / "sales_order.csv").open(encoding="utf-8") as source:
        first_day = sum(1 for row in csv.DictReader(source) if row["ordered_on"] == "2024-01-01")
    assert first_day > 500  # the trade orders on the first day as on any Monday


def test_the_s1_s2_and_s4_twins_pass_on_the_story_world(story_world: Built, capsys: pytest.CaptureFixture[str]) -> None:
    main(["stories-check", "--csv", str(story_world.run)])
    text = capsys.readouterr().out
    for check_sql in JUDGED:
        (row,) = [line for line in text.splitlines() if f"`{check_sql}`" in line]
        assert row.split(" | ")[-1].startswith("pass"), row  # the result, and a note after it (S1: each line's ratio)
    assert "| S4 | `kmeans` |" in text and (story_world.run / "stories.md").read_text(encoding="utf-8") == text


def test_the_null_world_fails_them(null_world: Built) -> None:
    found = _judged(check(null_world.run, StoriesConfig.load()))
    assert set(found) == set(JUDGED)
    assert not any(v.passed for v in found.values()), [(v.check, v.value) for v in found.values()]
    # each gives the null world's side, but S5's spread (its strict xfail below, transferred to G2 S2.4)
    verdicts = check(null_world.run, StoriesConfig.load(), null=True)
    assert all(v.passed is not False for v in verdicts if v.check != "detectors/s5_best_worst.sql"), [
        (v.check, v.value) for v in verdicts if v.passed is False
    ]


# ── S6 (G1 S1.4 T6, S1.5 T5) ──────────────────────────────────────────────────


def _reps(story_world: Built) -> dict[int, float]:
    """Each rep's hidden discount discipline: the answer key, the story world's (the null world's are all 1.0)."""
    world = Checkpoint.load(story_world.run).restore()
    return {e: a.skills.discount_discipline for e, a in world.actors.items() if a.role == "rep"}


def by_rep(run: Path, reps: dict[int, float]) -> pd.DataFrame:
    """Per rep: its agreed discounts' margin gained, their crowns, the gain a crown, its discipline."""
    found = gains(Build(run), S6_FROM)
    out = found.groupby("agreed_by_employee_id")[["gain", "discount"]].sum()
    out = out[out.index.isin(list(reps))]
    out["ratio"] = out["gain"] / out["discount"]
    out["discipline"] = [reps[int(e)] for e in out.index]
    return out


def spearman(frame: pd.DataFrame) -> float:
    return float(frame["ratio"].rank().corr(frame["discipline"].rank()))


@pytest.mark.xfail(
    strict=True,
    reason="Found (S1.5 T5, redrawn G2 S2.1 review-231): Spearman 0.10 on 2024–2025; residual SD 1.32 vs T2",
)
def test_s6_the_reps_gain_a_crown_ranks_with_their_discipline(story_world: Built) -> None:
    assert spearman(by_rep(story_world.run, _reps(story_world))) >= 0.8


def test_s6_in_the_null_world_the_ranking_is_gone(story_world: Built, null_world: Built) -> None:
    frame = by_rep(null_world.run, _reps(story_world))
    assert len(frame) == 50 and abs(spearman(frame)) < 0.3


@pytest.mark.xfail(
    strict=True,
    reason="Found (S1.5 T5, redrawn G2 S2.1 review-231): the pinned disciplined −0.31 a crown, not > 1 (T2: 1.59)",
)
def test_s6_the_pinned_disciplined_reps_discounts_pay(story_world: Built) -> None:
    assert s6_reps(Build(story_world.run), StoriesConfig.load())[True] > 1


def test_s6_the_pinned_undisciplined_reps_lose_on_their_discounts(story_world: Built) -> None:
    assert s6_reps(Build(story_world.run), StoriesConfig.load())[False] < 0.5


@pytest.mark.xfail(
    strict=True,
    reason="Found (S1.5 T5, redrawn G2 S2.1 review-231): s6.sql's pinned difference −0.68 on 2024–2025, not ≥ 0.5",
)
def test_s6_s_twin_passes_on_the_story_world(story_world: Built) -> None:
    (s6,) = [v for v in check(story_world.run, StoriesConfig.load()) if v.check == "detectors/s6.sql"]
    assert s6.passed, s6


def test_s6_s_twin_fails_on_the_null_world(null_world: Built) -> None:
    (s6,) = [v for v in check(null_world.run, StoriesConfig.load()) if v.check == "detectors/s6.sql"]
    assert s6.value is not None and not np.isnan(s6.value) and s6.passed is False


# ── the field (G2 S2.1 T7) ────────────────────────────────────────────────────

FIELD_PROBES = ("A-6", "A-7", "A-8", "A-13")  # A-13: G2 S2.2


def _actors(story_world: Built) -> dict[int, Any]:
    """The reps of the story world's checkpoint, the answer key (their hidden skills and traits)."""
    world = Checkpoint.load(story_world.run).restore()
    return {e: a for e, a in world.actors.items() if a.role == "rep"}


def test_a7_and_a8_find_the_pinned_reps(story_world: Built) -> None:
    reps = _actors(story_world)
    skipping = {e for e, a in reps.items() if a.skills.skips_visits}
    silent = {e for e, a in reps.items() if a.skills.records_nothing}
    assert len(skipping) == 3 and len(silent) == 1
    build = Build(story_world.run)
    plans = build["visit_plan"]
    plans = plans[plans["planned_day"].dt.year == 2025]
    adherence = (
        plans["visit_plan_id"].isin(build["visit"]["visit_plan_id"].dropna()).groupby(plans["employee_id"]).mean()
    )
    assert set(adherence[adherence <= 0.6].index) == skipping
    assert adherence.drop(list(skipping)).min() > 0.8
    visits = build["visit"]
    recorded = visits["recorded"].astype(bool).groupby(visits["employee_id"]).mean()
    assert set(recorded[recorded <= 0.35].index) == silent


def _conversion(run: Path) -> pd.Series:
    """Per rep, S3's first rate: its sales visits at small-retail customers that took an order ÷ those visits."""
    build = Build(run)
    visits = build["visit"].merge(build["customer"][["customer_id", "channel"]], on="customer_id")
    sales = visits[(visits["kind"] == "sales") & (visits["channel"] == "small_retail")]
    return (sales["outcome"] == "order").groupby(sales["employee_id"]).mean()


def test_s3_s_conversion_ranks_with_the_hidden_skill(story_world: Built, null_world: Built) -> None:
    skills = {e: a.skills.conversion for e, a in _actors(story_world).items()}
    for world, bar in ((story_world, 0.7), (null_world, None)):
        rates = _conversion(world.run)
        rates = rates[rates.index.isin(list(skills))]
        rho = rates.rank().corr(pd.Series({e: skills[int(e)] for e in rates.index}).rank())
        assert len(rates) == 50
        assert rho >= bar if bar is not None else abs(rho) < 0.3, rho


def test_the_field_probes_pass_on_the_story_world_and_find_nothing_in_the_null(
    story_world: Built, null_world: Built
) -> None:
    stories = StoriesConfig.load()
    found = {v.story: v for v in check(story_world.run, stories) if v.story in FIELD_PROBES}
    assert all(v.passed for v in found.values()), [(v.story, v.value) for v in found.values()]
    null = {v.story: v for v in check(null_world.run, stories, null=True) if v.story in FIELD_PROBES}
    assert all(v.passed and v.value == 0 for v in null.values()), [(v.story, v.value) for v in null.values()]


def test_the_notes_are_czech(story_world: Built) -> None:
    visits = Build(story_world.run)["visit"]
    notes = visits["note"].dropna()
    assert len(notes) > 100_000
    names = [*(r["name_cs"] for r in read_reference("product")), *(r["name"] for r in read_reference("competitor"))]
    names += [n for r in read_reference("product") for n in (r["nicknames_cs"] or "").split(";") if n]
    assert_czech_all(notes.sample(1_000, random_state=0).tolist(), what="visit.note", allow=names)


def test_small_retail_s_own_orders_keep_the_trade_week(story_world: Built) -> None:
    """review-231 R1: the phone and portal orders' Saturday and Sunday no less than 3 pp under the trade week's
    (`s1.yaml` `weekday.trade`, 11.7 %) — a visit that restarted the store's cycle put its order days on its calls
    (1.7 %). Above it there is room to 6 pp (re-review 1's RR1): the excess is the dry trigger's, which orders on any
    weekday after the order day a visit's order stood in for (⚑R231-A, NH-D16; 14.3 % on 2024–2025)."""
    orders = Build(story_world.run)["sales_order"]
    own = orders[(orders["channel"] == "small_retail") & orders["source"].isin(["phone", "portal"])]
    weekend = float((pd.to_datetime(own["ordered_on"]).dt.day_of_week >= 5).mean())
    trade = np.array(StoriesConfig.load().story("S1").parameters["weekday"]["trade"], dtype=float)
    share = trade[5:].sum() / trade.sum()
    assert share - 0.03 <= weekend <= share + 0.06, weekend


# ── the prompts (G2 S2.2 T7) ───────────────────────────────────────────────────


def test_the_prompts_of_a_year_their_kinds_caps_and_czech(story_world: Built) -> None:
    """On 2025: ~4 prompts a planned call on a small-retail store, each with its rule, its JSON payload and a Czech
    explanation; R-7's listing prompts the most, then the tips, then the mandatory promotions, and the display prompts
    since G2 S2.3 fills `World.display_uplift` and the budgets; every outcome class met, the classified ones with a
    reason where R-11 gives one; on no visit more prompts discussed than its customer's phase allows
    (`policies.prompt_caps`); and over 2024–2025 no prompt `not_discussed` whose item went on its visit's order
    (review-233 R1)."""
    build = Build(story_world.run)
    prompts = build["rep_prompt"]
    prompts = prompts[prompts["generated_on"].dt.year == 2025]
    plans = build["visit_plan"]
    plans = plans[plans["planned_day"].dt.year == 2025]
    assert 3.0 <= len(prompts) / len(plans) <= 5.0, len(prompts) / len(plans)
    assert prompts["prompt_rule_id"].notna().all() and (prompts["generated_by"] == "rules").all()
    assert all(isinstance(json.loads(payload), dict) for payload in prompts["payload"])
    shares = prompts["kind"].value_counts(normalize=True)
    assert set(shares.index) == {"unsupplied_items", "ai_tip", "mandatory_promotion", "pos_placement"}, shares
    assert shares["unsupplied_items"] > shares["ai_tip"] > shares["mandatory_promotion"], shares
    names = [r["name_cs"] for r in read_reference("product")] + [r["name_cs"] for r in read_reference("catalogue")]
    sample = prompts["explanation"].sample(1_000, random_state=0).tolist()
    assert_czech_all(sample, what="rep_prompt.explanation", allow=names)
    assert prompts["visit_id"].dropna().isin(build["visit"]["visit_id"]).all()
    assert not (prompts["kind"].eq("mandatory_promotion") & prompts["outcome"].eq("rejected")).any()
    outcomes = prompts["outcome"].value_counts()
    assert set(outcomes.index) == {"open", "not_discussed", "unsuccessful", "successful", "rejected"}, outcomes
    reasoned = prompts["outcome"].isin(["unsuccessful", "rejected"])
    assert prompts.loc[reasoned, "outcome_reason"].notna().all()
    assert prompts.loc[~reasoned, "outcome_reason"].isna().all()
    assert (prompts["classified_on"].notna() == (prompts["outcome"] != "open")).all()
    phases = build["lifecycle_phase"].sort_values("valid_from")
    discussed = prompts[prompts["outcome"] != "open"].sort_values("generated_on")
    held = phases[["customer_id", "valid_from", "phase"]]
    at = pd.merge_asof(discussed, held, left_on="generated_on", right_on="valid_from", by="customer_id")
    caps = WorldConfig.load().policies.prompt_caps
    per_visit = at.groupby(["visit_id", "phase"]).size().reset_index(name="n")
    assert (per_visit["n"] <= per_visit["phase"].map(caps)).all()
    every = build["rep_prompt"]
    undiscussed = every[every["outcome"] == "not_discussed"]
    codes = {code: int(sku) for code, sku in zip(build["sku"]["sku_code"], build["sku"]["sku_id"], strict=True)}
    offered = pd.DataFrame(
        [
            (int(visit), codes[code])
            for visit, payload in zip(undiscussed["visit_id"], undiscussed["payload"], strict=True)
            for code in json.loads(payload).get("skus", [])
        ],
        columns=["visit_id", "sku_id"],
    )
    orders = build["sales_order"]
    visited = orders.loc[orders["source"] == "visit", ["sales_order_id", "visit_id"]]
    lines = build["sales_order_line"][["sales_order_id", "sku_id"]].merge(visited, on="sales_order_id")
    lines = lines.astype({"visit_id": "int64", "sku_id": "int64"})
    assert len(offered) > 10_000 and offered.merge(lines, on=["visit_id", "sku_id"]).empty


# ── the point of sale (G2 S2.3) ───────────────────────────────────────────────


def _s5(world: Built, null: bool = False) -> dict[str, Verdict]:
    return {v.check: v for v in check(world.run, StoriesConfig.load(), null=null) if v.story == "S5"}


def test_s5_s_rank_passes_on_the_story_world_and_not_on_the_null(story_world: Built, null_world: Built) -> None:
    """T6: s5.sql's Spearman of the regions' returns against their managers' skill, ≥ 0.8 on the story world (0.90 on
    2024–2025), below it on the null world (0.78)."""
    found, null = _s5(story_world)["detectors/s5.sql"], _s5(null_world, null=True)["detectors/s5.sql"]
    assert found.passed and null.passed, (found.value, null.value)


S5_SPREAD = (
    "G2 S2.3 T6's acceptance, transferred to G2 S2.4 (NH-D17): s5_best_worst.sql's PHA ÷ VCH gain ÷ cost reads 1.27"
    " on the story world and 1.53 on the null one against the 1.35 derived on paper (1.54 and 1.21) — the display's"
    " lift is 11–14 % of the lent stores' margin and moves little with skill, while s5.sql's before/during gain"
    " carries each store's own drift (4–10 % of it by region); S2.4 corrects the gain by the untreated stores of the"
    " same windows and derives its threshold anew"
)


@pytest.mark.xfail(strict=True, reason=S5_SPREAD)
def test_s5_s_spread_passes_on_the_story_world(story_world: Built) -> None:
    found = _s5(story_world)["detectors/s5_best_worst.sql"]
    assert found.passed, found.value


@pytest.mark.xfail(strict=True, reason=S5_SPREAD)
def test_s5_s_spread_fails_on_the_null_world(null_world: Built) -> None:
    found = _s5(null_world, null=True)["detectors/s5_best_worst.sql"]
    assert found.passed, found.value


def test_the_loans_are_consistent_with_their_assets(story_world: Built) -> None:
    """S2.3's DONE: an asset is out on one loan at a time, lent after it was bought and before it was written off; one
    written off on a return came back damaged or past its life; each region lends and holds its pools; a loan is
    placed by the store's rep and approved by its region's manager."""
    build = Build(story_world.run)
    assets, loans = build["pos_asset"], build["pos_loan"]
    joined = loans.merge(assets, on="pos_asset_id").sort_values(["pos_asset_id", "lent_on"])
    assert len(loans) > 2000 and joined["region_id"].nunique() == 6
    returned = joined["returned_on"].fillna(pd.Timestamp("2099-12-31"))
    after = joined.groupby("pos_asset_id")["lent_on"].shift(-1)
    assert (after.isna() | (after >= returned)).all()
    assert (joined["lent_on"] >= joined["acquired_on"]).all()
    assert (joined["written_off_on"].isna() | (joined["written_off_on"] >= returned)).all()
    off = joined[joined["written_off_on"].notna() & (joined["written_off_on"] == joined["returned_on"])]
    life = off["acquired_on"] + pd.DateOffset(months=WorldConfig.load().field.pos_asset_life_months)
    assert ((off["condition_on_return"] == "damaged") | (life <= off["written_off_on"])).all()
    held = assets[assets["written_off_on"].isna()].groupby("region_id").size()
    pools = sum(m["assets_per_region"] for m in read_reference("pos_material"))
    assert (held <= pools).all() and (held > 0.8 * pools).all()
    managers = build["region"].set_index("region_id")["area_manager_employee_id"]
    assert (joined["approved_by_employee_id"] == joined["region_id"].map(managers)).all()
    reps = set(build["assignment"].query("kind == 'rep_area'")["employee_id"])
    assert joined["placed_by_employee_id"].isin(reps).all()
    register = build["asset_register"]
    # the purchases of 2021–2026: setup's three, then each January's, 2026's at 2025-12-31's month-end, dated its day
    assert set(register["code"].str[:8]) == {"MAJ-POS-"} and len(register) == 6 * 6


def test_every_loan_is_lent_on_a_day_its_customer_is_active(story_world: Built, null_world: Built) -> None:
    """G2 S2.5: in both worlds a loan's `lent_on` lies in its customer's `since`…`until`, both included
    (`Customer.active_on`) — a month-end lends from the 1st before the 1st's losses run, and the 1st's `placements`
    step cancels a placement they leave at a lost customer. A build ending on a month-end renders that month-end's
    placements, pending (lent the day after `last_day_done`): they pass against `last_day_done`'s dates, and their own
    day revalidates them in a prolong (`loop.pos`)."""
    for world in (story_world, null_world):
        build = Build(world.run)
        customers = build["customer"][["customer_id", "since", "until"]]
        loans = build["pos_loan"].merge(customers, on="customer_id", validate="many_to_one")
        lent = loans["lent_on"]
        active = (loans["since"] <= lent) & (loans["until"].isna() | (lent <= loans["until"]))
        assert len(loans) == len(build["pos_loan"]) and active.all(), loans[~active][["pos_loan_id", "lent_on"]]


def test_the_give_aways_follow_the_managers(story_world: Built) -> None:
    """T4: at 35 % of the recorded visits that dealt with someone (SMS's generous manager × 1.5), 30 % fewer in PHA and
    JMV (skilled); one item each, its cost `unit_cost × qty`."""
    build = Build(story_world.run)
    visits = build["visit"]
    visits = visits[visits["recorded"].astype(bool) & visits["outcome"].isin(["order", "no_order"])]
    regions = (
        build["store"][["store_id", "area_id"]]
        .merge(build["area"][["area_id", "region_id"]], on="area_id")
        .merge(build["region"][["region_id", "code"]], on="region_id")
    )
    visits = visits.merge(regions, on="store_id")
    given = build["pos_handout"].merge(build["pos_material"][["pos_material_id", "unit_cost"]], on="pos_material_id")
    assert (given["cost"] == (given["unit_cost"] * given["qty"]).round(2)).all() and given["qty"].between(1, 5).all()
    assert given["visit_id"].is_unique and given["visit_id"].isin(visits["visit_id"]).all()
    share = visits["visit_id"].isin(given["visit_id"]).groupby(visits["code"]).mean()
    base = WorldConfig.load().field.pos_handouts.share
    assert share[["SJC", "ZSC", "VCH"]].between(base - 0.03, base + 0.03).all(), share
    assert abs(share["SMS"] - 1.5 * base) < 0.04 and share[["PHA", "JMV"]].between(0.6 * base, 0.8 * base).all(), share


def test_a6_s_arc_end_to_end(story_world: Built) -> None:
    """T7: in East Bohemia Kavalír's promotion seen from week 45, the competitors' facings beside Gold up over weeks
    45–50, Gold's cases down in weeks 46–50 against the same customers' 8 weeks before, the season corrected — each
    past a6.sql's bar — and the planted hand-written notes in the visits' notes verbatim."""
    stories = StoriesConfig.load()
    build = Build(story_world.run)
    seen, lift, dip = a6_arc(build, stories)
    rose, fell = a6_bars(stories)
    assert seen >= 10 and lift >= rose and dip <= fell, (seen, lift, dip)
    sightings = build["competitor_sighting"]
    planted = sightings[sightings["description"].isin(PLANTED_SIGHTINGS)]
    notes = planted.merge(build["visit"][["visit_id", "note"]], on="visit_id").dropna(subset=["note"])
    assert len(notes) >= 10 and all(d in n for d, n in zip(notes["description"], notes["note"], strict=True))
