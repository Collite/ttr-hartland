"""S5's placement decision quality (G2 S2.9; NH-D25 design09 §§2–5): the pandas twin of the pilot's decision-quality
SQL (`data/stories/pilot/s5_quality_rank.sql` and `s5_quality_difference.sql`) and the judge of a pilot build.

**The claim** is forecast-based: which area managers place displays at the better forecast opportunities of their own
monthly candidate pool — „Kvalita výběru umístění stojanů“, „Podle předpovědi před rozhodnutím; nejde o prokázaný
přírůstek marže.“ It reads the stored pre-decision forecasts only, never a later sale, a skill, an uplift or another
world: no measured gain, payback or ROI, and the regional comparison is descriptive, not a causal manager effect.

**The frames**: `rounds`, the SQL's `round_quality` row for row — each region × each of W's 24 month-end decision
days, its decisions, its pool (viable, chosen, unchosen) and q — and `regions`, its `region_quality`: the rounds,
comparable rounds and Q, the decisions by outcome, the economic disclosure and `integrity` (the block's comment says
what each column means and the order the failures are judged in). Scores compare exactly, as integers of 1e-8 (the
stored eight decimals); q comes from average ranks, so a tie is half a comparison. Money: the Customer Margin in
haléře as `sales_pl` makes it (`margins`, G2 S2.7's twin of it, ttr-hartland#22), an asset's use and the fleet over
`LIFE_DAYS`, the give-aways as stored; the loan days are #22's — lent_on to the day before returned_on, from W's first
day for a warm-up loan, none for a loan pending past the last day — and an asset out twice on one day fails.
`faults` gives the build's failure counts the integrity is judged from.

**The metrics** (`rank`, `quality`): Spearman's rho of the six regions' Q against the story's declared skills
(`SKILL`, read only after Q; a control world is ranked against them too), 0 when the six Q are equal; the high
group's mean Q (`HIGH`), the weak group's (`LOW`) and their difference D. NaN unless every region is valid.

**The verdict** (`verdict`, design09 §5): PASS when rank ≥ 0.8, Q_H ≥ 0.70 and D ≥ 0.10, REJECTED for a valid frame
below them, INCONCLUSIVE for an invalid one; `control_rejects` when a valid frame has rank < 0.8 AND D < 0.10, the
rejection a control world must show — a control that is merely not a PASS has not rejected.

**The judge** (`judge`): NOT_CERTIFIED for no pilot build of this protocol in a scored mode (`SCORED`), a stale one (the
protocol's design, revision, window or file, or world.yaml, differs), one of another evaluation (its `_sim_meta`
revision, sha256 or claim is not the judging evaluation's) or one short of W; INCONCLUSIVE when a source check fails — a
table the twin reads missing, or a column of it; `build.json` or its audit manifest absent, or that manifest not the
audit tables' rows, sha256 and decisions by day; its written orders or lines of a quarter not the CSVs'; a row
`sales_pl` would drop (an order line without its order or sku, an order without its customer or delivery point, a
delivery point without its okres, a sku without its product, a product without its line) — or a region is not valid;
else the verdict's PASS or REJECTED. The SQL alone sees neither `build.json` nor the rows `sales_pl` drops: only these
source checks do. The judge does not check the build's source either: its git sha, its `stories_sha256` (s5.yaml) and
its `build_id` (the generator's revision) are not compared with the checkout's, so a build is the frozen head's only
when its runner built it there (G2 S2.10's manifest-driven runner checks them). A PASS here is one development build's
claim, never S5's certification (S2.11's protocol).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hartland_sim.checks.stories import Build, Missing
from hartland_sim.config import WORLD_FILE, WorldConfig, inventory
from hartland_sim.dates import quarter_start
from hartland_sim.evaluation import Evaluation
from hartland_sim.loop.build import AUDITED, BUILD_JSON, audit
from hartland_sim.protocol import PROTOCOL_FILE, Protocol
from hartland_sim.world import META_FILE, Meta

Frame = pd.DataFrame

SQL = ("pilot/s5_quality_rank.sql", "pilot/s5_quality_difference.sql")  # S2.11 makes them s5.sql, s5_best_worst.sql
# the SQL's `evaluation` row: the revisions, W, the protocol's baseline days, `H` (1461/8) and its forecast days, an
# asset's life (pos_asset_life_months × asset_days_per_month) and the support bars of s5_evaluation.yaml
REVISION, POLICY = "s5-pilot-v1", "nh-d21-b-r1"
FIRST, LAST = pd.Timestamp(2024, 1, 1), pd.Timestamp(2025, 12, 31)
BASELINE_DAYS, TERM_DAYS, FORECAST_DAYS = 56, 182.625, 183
LIFE_DAYS = 36 * 30.4
LEAST_ROUNDS, LEAST_CHOSEN, LEAST_UNCHOSEN = 12, 100, 100
SCORE_SLACK, MONEY_SLACK = 0.0000000050001, 0.005000001  # half the stored last digit, and a float's error
SKILL = {"PHA": 1.4, "JMV": 1.3, "SJC": 1.0, "ZSC": 0.6, "VCH": 0.5, "SMS": 0.5}  # s5_quality_rank.sql's: s5.yaml's
HIGH, LOW = ("PHA", "JMV"), ("ZSC", "VCH", "SMS")  # s5_quality_difference.sql's: the evaluation's high and low
SCORED = ("story", "skill-neutral", "full-null")  # the modes judged; the no-PoS ones lend nothing
OUTCOMES = ("selected", "no_budget", "no_asset", "nonpositive_score", "ineligible", "cancelled")
FAULTS = (  # the build's failures, in the order `integrity` names the first
    "revision",
    "duplicate",
    "outcome",
    "fields",
    "links",
    "loans",
    "ownership",
    "asset",
    "visit",
    "overlap",
    "astray",
)
TABLES = (  # the observed tables the twin reads, and the parents `sales_pl` joins that `margins` does not
    "pos_pilot_assignment",
    "pos_placement_decision",
    "region",
    "area",
    "store",
    "delivery_point",
    "okres",
    "sku",
    "product",
    "product_line",
    "employee",
    "pos_material",
    "customer",
    "sales_order",
    "sales_order_line",
    "sales_order_line_discount",
    "pos_loan",
    "pos_asset",
    "pos_handout",
    "visit",
)
ORDERS = ("sales_order_id", "customer_id", "delivery_point_id", "ordered_on")  # the columns read, each file parsed once
LINES = (
    "sales_order_line_id",
    "sales_order_id",
    "sku_id",
    "qty_cases",
    "list_price_case",
    "warehouse_cost_case",
    "delivery_cost_case",
)
ROUND_COLUMNS = ("decisions", "viable", "chosen", "unchosen", "q")  # `round_quality`'s but its keys
COLUMNS = (  # `region_quality`'s but its index
    "rounds",
    "comparable_rounds",
    "decisions",
    "viable",
    "chosen",
    "unchosen",
    "selected",
    "cancelled",
    "no_budget",
    "no_asset",
    "no_asset_initial",
    "nonpositive_score",
    "ineligible",
    "quality",
    "programme_customers",
    "observed_margin",
    "asset_use_cost",
    "handout_cost",
    "programme_cost",
    "no_sales_customers",
    "no_sales_cost",
    "fleet_cost",
    "idle_cost",
    "idle_assets",
    "integrity",
)
ONE_DAY = pd.Timedelta(days=1)


# ── the observed rows ────────────────────────────────────────────────────────


def _haler(values: pd.Series[Any]) -> np.ndarray[Any, Any]:
    """CZK with two decimals as integer haléře (NULL as 0)."""
    return np.asarray(np.round(values.astype(float).fillna(0.0).to_numpy() * 100), dtype=np.int64)


def _rounded(price: np.ndarray[Any, Any], cases: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """price (haléře) × cases (thousandths) to the haléř, half away from zero: SQL's ROUND(price × cases, 2)."""
    raw = price * cases
    return np.asarray(np.sign(raw) * ((np.abs(raw) + 500) // 1000), dtype=np.int64)


def margins(build: Build) -> Frame:
    """Every order line's Customer Margin in haléře as `sales_pl` makes it — list × cases, less its discounts, its
    delivery and its warehouse cost, each rounded half away from zero (#22's twin) — with its order's customer and
    day."""
    lines = build.narrow("sales_order_line", LINES)
    cases = np.asarray(np.round(lines["qty_cases"].astype(float).to_numpy() * 1000), dtype=np.int64)
    gross = _rounded(_haler(lines["list_price_case"]), cases)
    costs = _rounded(_haler(lines["delivery_cost_case"]), cases) + _rounded(_haler(lines["warehouse_cost_case"]), cases)
    discounts = build.narrow("sales_order_line_discount", ("sales_order_line_id", "amount_line"))
    off = (
        pd.Series(_haler(discounts["amount_line"]), index=discounts.index)
        .groupby(discounts["sales_order_line_id"])
        .sum()
    )
    taken = lines["sales_order_line_id"].map(off).fillna(0).to_numpy(dtype=np.int64)
    out = pd.DataFrame({"sales_order_id": lines["sales_order_id"], "margin": gross - taken - costs})
    orders = build.narrow("sales_order", ORDERS)
    return out.merge(orders[["sales_order_id", "customer_id", "ordered_on"]], on="sales_order_id")


@dataclass(frozen=True)
class Scope:
    """The block's `w`, `as_of` and `programme`: whether the revision's assignments have W as their one window, the
    assignment day, the last order day, the revision's assignment rows and its programme customers' region, each
    customer once (one assigned twice is a `duplicate` fault)."""

    one: bool
    assigned_on: pd.Timestamp
    as_of: pd.Timestamp
    assigned: Frame
    programme: Frame  # customer_id, region_id


def scope(build: Build) -> Scope:
    every = build["pos_pilot_assignment"]
    a = every[every["pilot_revision"] == REVISION]
    one = bool(
        a["window_from"].nunique() == 1
        and a["window_until"].nunique() == 1
        and a["window_from"].min() == FIRST
        and a["window_until"].min() == LAST
    )
    as_of = build.narrow("sales_order", ORDERS)["ordered_on"].max()
    programme = a.loc[a["arm"] == "programme", ["customer_id", "region_id"]].drop_duplicates()
    return Scope(one, a["assigned_on"].min(), as_of, a, programme)


def _distinct(a: pd.Series[Any], b: pd.Series[Any] | Any) -> pd.Series[Any]:
    """SQL's IS DISTINCT FROM, elementwise (`b` a series or one value)."""
    b = b if isinstance(b, pd.Series) else pd.Series([b] * len(a), index=a.index, dtype=object)
    left, right = a.isna().to_numpy(), b.isna().to_numpy()
    both = ~left & ~right
    differ = np.zeros(len(a), dtype=bool)
    differ[both] = a.to_numpy(dtype=object)[both] != b.to_numpy(dtype=object)[both]
    return pd.Series(differ | (left != right), index=a.index)


def decided(build: Build) -> Frame:
    """The block's `decided`: every decision with `viable`, `inside` (its effective day in W) and `sound`."""
    return build.joined("s5_decision_decided", _decided)


def _decided(build: Build) -> Frame:
    d = build["pos_placement_decision"].copy()
    outcome, score = d["outcome"], d["score"]
    gain, cost, asset_cost = d["forecast_gain"], d["forecast_cost"], d["asset_cost"]
    material, manager, loan = d["forecast_material_id"].notna(), d["manager_id"].notna(), d["loan_id"].notna()
    positive, zero = score > 0, score == 0
    d["viable"] = (gain > 0) & (cost > 0) & positive & material & manager
    effective = d["decision_day"] + ONE_DAY
    d["inside"] = (effective >= FIRST) & (effective <= LAST)
    rule = np.select(
        [
            outcome.isin(["selected", "cancelled", "no_budget"]),
            outcome == "no_asset",
            outcome == "nonpositive_score",
            outcome == "ineligible",
        ],
        [
            positive & manager & (loan == (outcome == "selected")),
            ~loan & manager & (positive | (zero & ~material)),
            ~loan & manager & zero & (gain == 0) & (cost > 0),
            ~loan & zero & (~manager | (cost <= 0)),
        ],
        default=False,
    )
    forecast_gain = np.maximum(d["prior56_line_margin"] / BASELINE_DAYS, 0) * d["catalogue_fit"] * TERM_DAYS
    forecast_cost = asset_cost * TERM_DAYS / LIFE_DAYS + d["prior56_handout_cost"] / BASELINE_DAYS * TERM_DAYS
    rounded = ~material | (
        ((gain - forecast_gain).abs() <= MONEY_SLACK) & ((cost - forecast_cost).abs() <= MONEY_SLACK)
    )
    d["sound"] = (
        rule.astype(bool)
        & (material == gain.notna())
        & (gain.notna() == cost.notna())
        & (cost.notna() == asset_cost.notna())
        & (score >= 0)
        & (zero | ((cost > 0) & ((score - gain / cost).abs() <= SCORE_SLACK)))
        & rounded
        & (d["forecast_days"] == FORECAST_DAYS).fillna(False).astype(bool)
        & (effective.dt.day == 1)
    )
    d["units"] = np.round(score.fillna(0).to_numpy() * 1e8).astype(np.int64)  # the stored score, exact
    return d


def calendar(build: Build) -> Frame:
    """The block's `calendar`: each region × each month-end whose next day is in W."""
    days = pd.date_range(FIRST, LAST, freq="MS") - ONE_DAY
    regions = build["region"][["region_id", "code"]].rename(columns={"code": "region"})
    return regions.merge(pd.DataFrame({"decision_day": days}), how="cross")


def rounds(build: Build) -> Frame:
    """The SQL's `round_quality` (module docstring), by region code and decision day."""
    return build.joined("s5_decision_rounds", _rounds)


def _rounds(build: Build) -> Frame:
    d = decided(build)
    keys = ["region_id", "decision_day"]
    pool = d.loc[d["inside"] & d["viable"], [*keys, "outcome", "units"]].copy()
    pool["chosen"] = pool["outcome"].isin(["selected", "cancelled"])
    pool["twice"] = np.round(2 * pool.groupby(keys)["units"].rank(method="average")).astype(np.int64)
    pooled = pool.groupby(keys).agg(viable=("chosen", "size"), chosen=("chosen", "sum"))
    pooled["twice_ranks"] = pool[pool["chosen"]].groupby(keys)["twice"].sum()
    counted = d[d["inside"]].groupby(keys).size().rename("decisions")
    out = (
        calendar(build)
        .merge(pooled.reset_index(), on=keys, how="left")
        .merge(counted.reset_index(), on=keys, how="left")
    )
    for name in ("viable", "chosen", "decisions", "twice_ranks"):
        out[name] = out[name].fillna(0).astype(np.int64)
    out["unchosen"] = out["viable"] - out["chosen"]
    k, n = out["chosen"].to_numpy(dtype=float), out["viable"].to_numpy(dtype=float)
    comparable = (k > 0) & (n > k)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (out["twice_ranks"] - k * (k + 1)) / (2 * k * (n - k))
    out["q"] = q.where(comparable)
    return out.set_index(["region", "decision_day"])[["region_id", *ROUND_COLUMNS]]


def _used(build: Build, s: Scope) -> Frame:
    """The block's `used`: every loan with its asset (NaN when missing), its days in W and their cost."""
    loans = build["pos_loan"][["pos_loan_id", "pos_asset_id", "customer_id", "lent_on", "returned_on"]]
    assets = build["pos_asset"][["pos_asset_id", "region_id", "cost"]].rename(columns={"region_id": "asset_region_id"})
    loans = loans.merge(assets, on="pos_asset_id", how="left", indicator=True)
    loans["has_asset"] = loans.pop("_merge") == "both"
    ends = pd.DataFrame({"back": loans["returned_on"] - ONE_DAY, "as_of": s.as_of, "last": LAST})
    ends["back"] = ends["back"].fillna(ends["as_of"])
    loans["first_day"] = pd.DataFrame({"lent": loans["lent_on"], "first": FIRST}).max(axis=1)
    loans["last_day"] = ends.min(axis=1)
    days = (loans["last_day"] - loans["first_day"]).dt.days + 1
    loans["days"] = days.clip(lower=0).fillna(0).astype(np.int64)
    loans["use_cost"] = loans["cost"].astype(float).fillna(0.0) * loans["days"] / LIFE_DAYS
    return loans


def _handed(build: Build) -> Frame:
    """The block's `handed`: every give-away with its visit's customer and day (NaN when the visit is missing)."""
    visits = build["visit"][["visit_id", "customer_id", "day"]]
    handed = build["pos_handout"][["pos_handout_id", "visit_id", "cost"]].merge(
        visits, on="visit_id", how="left", indicator=True
    )
    handed["has_visit"] = handed.pop("_merge") == "both"
    handed["in_w"] = handed["has_visit"] & (handed["day"] >= FIRST) & (handed["day"] <= LAST)
    return handed


def _spending(build: Build) -> tuple[Frame, Frame]:
    """`_used` and `_handed`, made once for the faults and the disclosure."""
    return build.joined("s5_decision_used", lambda b: _used(b, scope(b))), build.joined("s5_decision_handed", _handed)


def faults(build: Build) -> dict[str, int]:
    """The block's `fault`: each failure's count (module docstring); all 0 for a sound build."""
    held = build.joined("s5_decision_faults", lambda b: pd.DataFrame([_faults(b)]))
    return {name: int(held[name].iloc[0]) for name in FAULTS}


def _faults(build: Build) -> dict[str, int]:
    s, d = scope(build), decided(build)
    used, handed = _spending(build)
    out: dict[str, int] = {}
    assignments = build["pos_pilot_assignment"]
    out["revision"] = int(
        (_distinct(d["pilot_revision"], REVISION) | _distinct(d["policy_revision"], POLICY)).sum()
        + _distinct(assignments["pilot_revision"], REVISION).sum()
    )
    keyed = d.groupby(["pilot_revision", "decision_day", "delivery_point_id"], dropna=False).size()
    out["duplicate"] = int(
        len(d)
        - d["pos_placement_decision_id"].nunique()
        + (keyed > 1).sum()
        + d["loan_id"].notna().sum()
        - d["loan_id"].nunique()
        + len(s.assigned)
        - s.assigned["customer_id"].nunique()
    )
    out["outcome"] = int((~d["outcome"].isin(OUTCOMES)).sum())
    out["fields"] = int((~d["sound"]).sum())
    out["links"] = _links(build, s, d)
    selected = set(d.loc[d["outcome"] == "selected", "loan_id"].dropna())
    loans = build["pos_loan"]
    since = loans["lent_on"] >= s.assigned_on if not pd.isna(s.assigned_on) else pd.Series(False, index=loans.index)
    cancelled = d.loc[d["outcome"] == "cancelled", ["customer_id", "decision_day"]]
    lent = set(zip(loans["customer_id"], loans["lent_on"], strict=True))
    holding = [
        (c, day + ONE_DAY) in lent for c, day in zip(cancelled["customer_id"], cancelled["decision_day"], strict=True)
    ]
    out["loans"] = int((since & ~loans["pos_loan_id"].isin(selected)).sum() + sum(holding))
    spending = pd.concat(
        [used.loc[used["days"] > 0, "customer_id"], handed.loc[handed["in_w"], "customer_id"]], ignore_index=True
    )
    customers = build["customer"][["customer_id", "channel"]]
    known = spending.isin(customers["customer_id"])
    small = spending.map(customers.set_index("customer_id")["channel"]).eq("small_retail")
    out["ownership"] = int((~known | (small & ~spending.isin(s.programme["customer_id"]))).sum())
    out["asset"] = int((~used["has_asset"]).sum())
    out["visit"] = int((~handed["has_visit"]).sum())
    lending = used[used["days"] > 0]
    twice = lending.merge(lending, on="pos_asset_id", suffixes=("", "_y"))
    twice = twice[
        (twice["pos_loan_id_y"] > twice["pos_loan_id"])
        & (twice["first_day"] <= twice["last_day_y"])
        & (twice["first_day_y"] <= twice["last_day"])
    ]
    out["overlap"] = len(twice)
    away = lending[lending["has_asset"]].merge(s.programme, on="customer_id")
    out["astray"] = int((away["asset_region_id"] != away["region_id"]).sum())
    return out


def _links(build: Build, s: Scope, d: Frame) -> int:
    """The block's `fault.links`: decisions not linked as their programme customer, region, delivery point, manager,
    material and loan say."""
    x = d[["customer_id", "region_id", "delivery_point_id", "manager_id", "forecast_material_id", "loan_id"]]
    x = x.assign(decision_day=d["decision_day"])
    x = x.merge(s.programme.rename(columns={"region_id": "p_region"}).assign(p=True), on="customer_id", how="left")
    x["r"] = x["region_id"].isin(build["region"]["region_id"])
    points = build["delivery_point"][["delivery_point_id", "customer_id", "store_id"]]
    x = x.merge(points.rename(columns={"customer_id": "dp_customer"}), on="delivery_point_id", how="left")
    places = build["store"][["store_id", "area_id"]].merge(build["area"][["area_id", "region_id"]], on="area_id")
    x = x.merge(places[["store_id", "region_id"]].rename(columns={"region_id": "a_region"}), on="store_id", how="left")
    loans = build["pos_loan"][["pos_loan_id", "customer_id", "lent_on", "pos_asset_id"]]
    assets = build["pos_asset"][["pos_asset_id", "region_id"]].rename(columns={"region_id": "l_region"})
    loans = loans.merge(assets, on="pos_asset_id", how="left").rename(columns={"customer_id": "l_customer"})
    x = x.merge(loans.drop(columns="pos_asset_id"), left_on="loan_id", right_on="pos_loan_id", how="left")
    lent = x["loan_id"].notna()
    broken = (
        x["p"].isna()
        | _distinct(x["p_region"], x["region_id"])
        | ~x["r"]
        | _distinct(x["dp_customer"], x["customer_id"])
        | _distinct(x["a_region"], x["region_id"])
        | (x["manager_id"].notna() & ~x["manager_id"].isin(build["employee"]["employee_id"]))
        | (
            x["forecast_material_id"].notna()
            & ~x["forecast_material_id"].isin(build["pos_material"]["pos_material_id"])
        )
        | (
            lent
            & (
                x["pos_loan_id"].isna()
                | _distinct(x["l_customer"], x["customer_id"])
                | _distinct(x["lent_on"], x["decision_day"] + ONE_DAY)
                | (x["l_region"].notna() & _distinct(x["l_region"], x["region_id"]))
            )
        )
    )
    return int(broken.sum())


def regions(build: Build) -> Frame:
    """The SQL's `region_quality` (module docstring), by region code."""
    return build.joined("s5_decision_regions", _regions)


def _regions(build: Build) -> Frame:
    s, d, r = scope(build), decided(build), rounds(build)
    used, handed = _spending(build)
    comparable = r["q"].notna()
    by = r["region_id"]
    sums = pd.DataFrame(
        {
            "rounds": r.groupby(by).size(),
            "comparable_rounds": comparable.groupby(by).sum(),
            "decisions": r["decisions"].groupby(by).sum(),
            "viable": r["viable"].groupby(by).sum(),
            "chosen": r["chosen"].where(comparable, 0).groupby(by).sum(),
            "unchosen": r["unchosen"].where(comparable, 0).groupby(by).sum(),
            "quality": r["q"].groupby(by).mean(),
        }
    )
    inside = d[d["inside"]]
    initial = (inside["outcome"] == "no_asset") & ~(inside["score"] > 0)
    kinds = {
        "selected": inside["outcome"] == "selected",
        "cancelled": inside["outcome"] == "cancelled",
        "no_budget": inside["outcome"] == "no_budget",
        "no_asset": (inside["outcome"] == "no_asset") & (inside["score"] > 0),
        "no_asset_initial": initial,
        "nonpositive_score": inside["outcome"] == "nonpositive_score",
        "ineligible": inside["outcome"] == "ineligible",
    }
    outcomes = pd.DataFrame({name: flag.groupby(inside["region_id"]).sum() for name, flag in kinds.items()})

    lines = margins(build)
    sold = lines[(lines["ordered_on"] >= FIRST) & (lines["ordered_on"] <= LAST)].groupby("customer_id")["margin"]
    spent = s.programme.copy()
    spent["margin"] = spent["customer_id"].map(sold.sum()).fillna(0).astype(np.int64)
    spent["lines"] = spent["customer_id"].map(sold.size()).fillna(0).astype(np.int64)
    spent["asset_use_cost"] = spent["customer_id"].map(used.groupby("customer_id")["use_cost"].sum()).fillna(0.0)
    given = handed[handed["in_w"]].groupby("customer_id")["cost"].sum()
    spent["handout_cost"] = spent["customer_id"].map(given).fillna(0.0)
    spent["none"] = spent["lines"] == 0
    spent["none_cost"] = (spent["asset_use_cost"] + spent["handout_cost"]).where(spent["none"], 0.0)
    economics = spent.groupby("region_id").agg(
        programme_customers=("customer_id", "size"),
        observed_margin=("margin", "sum"),
        asset_use_cost=("asset_use_cost", "sum"),
        handout_cost=("handout_cost", "sum"),
        no_sales_customers=("none", "sum"),
        no_sales_cost=("none_cost", "sum"),
    )
    economics["observed_margin"] = economics["observed_margin"] / 100

    assets = build["pos_asset"][["pos_asset_id", "region_id", "cost", "acquired_on", "written_off_on"]].copy()
    ends = pd.DataFrame({"off": assets["written_off_on"] - ONE_DAY, "as_of": s.as_of, "last": LAST})
    ends["off"] = ends["off"].fillna(ends["as_of"])
    starts = pd.DataFrame({"acquired": assets["acquired_on"], "first": FIRST}).max(axis=1)
    assets["days"] = ((ends.min(axis=1) - starts).dt.days + 1).clip(lower=0).fillna(0).astype(np.int64)
    assets["carry"] = assets["cost"].astype(float) * assets["days"] / LIFE_DAYS
    lent_days = assets["pos_asset_id"].map(used.groupby("pos_asset_id")["days"].sum()).fillna(0)
    assets["idle"] = (assets["days"] > 0) & (lent_days == 0)
    fleet = assets.groupby("region_id").agg(fleet_cost=("carry", "sum"), idle_assets=("idle", "sum"))
    fleet_used = used[used["has_asset"]].groupby("asset_region_id")["use_cost"].sum()

    ids = build["region"][["region_id", "code"]]
    frame = pd.DataFrame(index=pd.Index(ids["region_id"], name="region_id"))
    frame = frame.join([sums, outcomes, economics, fleet])
    counts = [name for name in COLUMNS[:13] if name != "quality"] + ["programme_customers", "no_sales_customers"]
    for name in [*counts, "idle_assets"]:
        frame[name] = frame[name].fillna(0).astype(np.int64)
    for name in ("observed_margin", "asset_use_cost", "handout_cost", "no_sales_cost", "fleet_cost"):
        frame[name] = frame[name].astype(float).fillna(0.0)
    frame["quality"] = frame["quality"].astype(float)
    frame["programme_cost"] = frame["asset_use_cost"] + frame["handout_cost"]
    frame["idle_cost"] = frame["fleet_cost"] - fleet_used.reindex(frame.index).astype(float).fillna(0.0)
    frame.index = pd.Index(ids["code"].to_numpy(), name="region")
    frame["integrity"] = _integrity(frame, s, faults(build))
    return frame[list(COLUMNS)]


def _integrity(frame: Frame, s: Scope, found: dict[str, int]) -> list[str]:
    """The block's `integrity` per region (module docstring): the build's failures first, then the region's."""
    if not s.one:
        build = "window"
    elif pd.isna(s.as_of) or s.as_of < LAST:
        build = "short"
    else:
        build = next((name for name in FAULTS if found[name] > 0), "")
    if not build and (frame["programme_customers"] == 0).any():
        build = "regions"
    out = []
    counts = (frame[name].to_numpy(dtype=np.int64) for name in ("comparable_rounds", "chosen", "unchosen"))
    for comparable, chosen, unchosen in zip(*counts, strict=True):
        if build:
            out.append(build)
        elif comparable >= LEAST_ROUNDS and chosen >= LEAST_CHOSEN and unchosen >= LEAST_UNCHOSEN:
            out.append("valid")
        else:
            out.append("support")
    return out


# ── the metrics ──────────────────────────────────────────────────────────────


def _valid(frame: Frame) -> bool:
    return len(frame) == len(SKILL) and set(frame.index) == set(SKILL) and bool((frame["integrity"] == "valid").all())


def rank(frame: Frame) -> float:
    """s5_quality_rank.sql (module docstring)."""
    if not _valid(frame):
        return math.nan
    quality = frame["quality"].astype(float).rank(method="average")
    if quality.min() == quality.max():
        return 0.0
    skills = pd.Series(SKILL)[frame.index].rank(method="average")
    return float(np.corrcoef(quality.to_numpy(), skills.to_numpy())[0, 1])


def quality(frame: Frame) -> tuple[float, float, float]:
    """s5_quality_difference.sql's parts (module docstring): Q_H, Q_L and D = Q_H − Q_L (NaN unless valid)."""
    q = frame["quality"].astype(float).reindex(list(SKILL))
    high, low = float(q[list(HIGH)].mean()), float(q[list(LOW)].mean())
    if not _valid(frame):
        return high, low, math.nan
    return high, low, high - low


@dataclass(frozen=True)
class Verdict:
    """A frame's claim against the evaluation's bars (`verdict`): PASS or REJECTED for a valid frame, else
    INCONCLUSIVE; the metrics, whether each bar holds and whether a control's rejection holds."""

    status: str
    rank: float
    q_high: float
    q_low: float
    difference: float
    rank_holds: bool
    q_high_holds: bool
    difference_holds: bool
    control_rejects: bool  # valid, rank < its bar AND D < its bar: what a control world must show


def verdict(frame: Frame, held: Evaluation) -> Verdict:
    bars = held.acceptance
    rho, (high, low, difference) = rank(frame), quality(frame)
    rank_holds = rho >= float(bars["rank"])
    q_high_holds = high >= float(bars["q_high"])
    difference_holds = difference >= float(bars["difference"])
    valid = _valid(frame) and not math.isnan(rho) and not math.isnan(difference)
    if not valid:
        status = "INCONCLUSIVE"
    else:
        status = "PASS" if rank_holds and q_high_holds and difference_holds else "REJECTED"
    control = valid and rho < float(bars["rank"]) and difference < float(bars["difference"])
    return Verdict(status, rho, high, low, difference, rank_holds, q_high_holds, difference_holds, control)


# ── the judge ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Judged:
    """A build's judgement (`judge`): its status and why, its verdict, frames (None when NOT_CERTIFIED or a source
    check fails) and identifiers."""

    run: Path
    mode: str | None
    status: str
    reasons: tuple[str, ...]
    verdict: Verdict | None
    frame: Frame | None
    rounds: Frame | None
    identity: dict[str, Any]


def identity(run: Path) -> dict[str, Any]:
    """The build's design, source, evaluation and window identifiers: `_sim_meta`'s and `build.json`'s."""
    out: dict[str, Any] = {}
    if (run / META_FILE).is_file():
        meta = Meta.load(run)
        out |= {
            "build_id": meta.build_id,
            "master_seed": meta.master_seed,
            "stories_sha256": meta.stories_sha256,
            "world_sha256": meta.world_sha256,
            "last_day_done": meta.last_day_done.isoformat(),
            **dict(meta.pilot),
        }
    if (run / BUILD_JSON).is_file():
        held = json.loads((run / BUILD_JSON).read_text(encoding="utf-8"))
        out |= {"git_sha": held.get("git_sha"), "git_dirty": held.get("git_dirty")}
    return out


def stale(run: Path, protocol: Protocol, judged_by: Evaluation | None = None) -> list[str]:
    """Why `run` is no complete scored pilot build of this protocol and evaluation (module docstring); empty when it
    is one. Its source — git sha, `stories_sha256`, `build_id` — is not compared here, but by its runner (module
    docstring)."""
    if not (run / META_FILE).is_file():
        return ["no _sim_meta.csv"]
    meta = Meta.load(run)
    held = dict(meta.pilot)
    mode = held.get("pilot_mode")
    if mode is None:
        return ["not a pilot build"]
    if mode not in SCORED:
        return [f"a {mode} build lends nothing: not scored"]
    window = f"{protocol.window_from.isoformat()}/{protocol.window_until.isoformat()}"
    expected = {
        "pilot_design": protocol.design,
        "pilot_revision": protocol.pilot_revision,
        "policy_revision": protocol.policy_revision,
        "pilot_window": window,
        "pilot_protocol_sha256": hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest(),
        **dict((judged_by or Evaluation.load()).identity()),
    }
    out = [f"stale {key}" if key in held else f"no {key}" for key, value in expected.items() if held.get(key) != value]
    if meta.world_sha256 != hashlib.sha256(WORLD_FILE.read_bytes()).hexdigest():
        out.append("stale world.yaml")
    if meta.last_day_done < protocol.window_until:
        out.append(f"short: done to {meta.last_day_done}")
    return out


def sources(build: Build, first_day: date) -> list[str]:
    """The source checks (module docstring); empty when they hold."""
    lacking = _headers(build)
    if lacking:
        return lacking
    try:
        for table in TABLES:
            if table not in ("sales_order", "sales_order_line", "sales_order_line_discount"):
                build[table]
        orders = build.narrow("sales_order", ORDERS)
        lines = build.narrow("sales_order_line", LINES)
        build.narrow("sales_order_line_discount", ("sales_order_line_id", "amount_line"))
    except Missing as error:
        return [str(error)]
    points, skus, products = build["delivery_point"], build["sku"], build["product"]
    out = []
    for what, ids, among in (  # `sales_pl` inner-joins each: an orphan would leave the SQL's margin, not the twin's
        ("an order line without its order", lines["sales_order_id"], orders["sales_order_id"]),
        ("an order without its customer", orders["customer_id"], build["customer"]["customer_id"]),
        ("an order without its delivery point", orders["delivery_point_id"], points["delivery_point_id"]),
        ("a delivery point without its okres", points["okres_id"], build["okres"]["okres_id"]),
        ("an order line without its sku", lines["sku_id"], skus["sku_id"]),
        ("a sku without its product", skus["product_id"], products["product_id"]),
        ("a product without its line", products["product_line_id"], build["product_line"]["product_line_id"]),
    ):
        missing = int((~ids.isin(among)).sum())
        if missing:
            out.append(f"{what} ({missing})")
    return out + _audited(build) + _manifest(build, orders, lines, first_day)


def _headers(build: Build) -> list[str]:
    """Each table's CSV header against its inventory columns, before any read: a column missing would make a narrow
    read or a twin's lookup raise, not name it."""
    out = []
    for table in TABLES:
        path = build.run / f"{table}.csv"
        if path.is_file():  # a file missing is named by the reads
            with path.open(encoding="utf-8", newline="") as handle:
                header = set(next(csv.reader(handle), []))
            lacking = [column.name for column in inventory().table(table).columns if column.name not in header]
            if lacking:
                out.append(f"{table}.csv lacks {', '.join(lacking)}")
    return out


def _audited(build: Build) -> list[str]:
    """`build.json`'s audit manifest — the last entry's (a prolong renders the audit tables again) — against the audit
    tables' files: their rows, sha256 and the decisions by day."""
    path = build.run / BUILD_JSON
    if not path.is_file():
        return ["no build.json"]
    held = json.loads(path.read_text(encoding="utf-8"))
    entries = [entry for entry in [held, *held.get("prolongs", [])] if "audit" in entry]
    if not entries:
        return ["no audit manifest in build.json"]
    expected, found = entries[-1]["audit"], audit(build.run)
    out = []
    for name in AUDITED:
        mine, theirs = expected.get(name, {}), found[name]
        for key in theirs:
            if mine.get(key) != theirs[key]:
                out.append(f"{name} {key}: build.json {_short(mine.get(key))}, CSV {_short(theirs[key])}")
    return out


def _short(value: Any) -> str:
    return f"{len(value)} days" if isinstance(value, dict) else str(value)


def _manifest(build: Build, orders: Frame, lines: Frame, first_day: date) -> list[str]:
    """`build.json`'s written orders and lines by quarter (a prolong's added) against the CSVs' (#22's check); none
    without `build.json`, which `_audited` names."""
    path = build.run / BUILD_JSON
    if not path.is_file():
        return []
    held = json.loads(path.read_text(encoding="utf-8"))
    counted: dict[str, dict[str, int]] = {}
    for entry in [held, *held.get("prolongs", [])]:
        for name, counts in entry.get("quarters", {}).items():
            mine = counted.setdefault(name, {"orders": 0, "lines": 0})
            mine["orders"] += int(counts.get("orders", 0))
            mine["lines"] += int(counts.get("lines", 0))
    quarter = orders["ordered_on"].dt.to_period("Q").astype(str).str.replace("Q", "-Q")
    of_order = pd.Series(quarter.to_numpy(), index=orders["sales_order_id"])
    found = {"orders": quarter.value_counts(), "lines": lines["sales_order_id"].map(of_order).value_counts()}
    written = quarter_start(first_day)
    out = []
    for name in sorted(set(counted) | set(found["orders"].index)):
        year, q = name.split("-Q")
        if date(int(year), 3 * int(q) - 2, 1) < written:
            continue
        for kind in ("orders", "lines"):
            expected, there = counted.get(name, {}).get(kind, 0), int(found[kind].get(name, 0))
            if expected != there:
                out.append(f"{name} {kind}: build.json {expected}, CSV {there}")
    return out


def judge(run: Path, protocol: Protocol | None = None, held: Evaluation | None = None) -> Judged:
    """`run`'s status (module docstring)."""
    protocol, held = protocol or Protocol.load(), held or Evaluation.load()
    ident = identity(run)
    mode = ident.get("pilot_mode")
    why = stale(run, protocol, held)
    if why:
        return Judged(run, mode, "NOT_CERTIFIED", tuple(why), None, None, None, ident)
    build = Build(run)
    broken = sources(build, WorldConfig.load().history.first_day)
    if broken:
        return Judged(run, mode, "INCONCLUSIVE", tuple(broken), None, None, None, ident)
    frame = regions(build)
    found = verdict(frame, held)
    reasons = tuple(f"{region}: {state}" for region, state in frame["integrity"].items() if state != "valid")
    return Judged(run, mode, found.status, reasons, found, frame, rounds(build), ident)


# ── the readout ──────────────────────────────────────────────────────────────


def markdown(judged: Judged, held: Evaluation | None = None) -> str:
    """The judgement in Czech, forecast and observed apart (design09 §§2, 4): the claim's label and its disclosure,
    each region's Q with its support, its observed economics unattributed, and the measures not made."""
    held = held or Evaluation.load()
    labels = held.labels
    lines = [f"# {labels['title_cs']}: {judged.status}", "", f"_{labels['disclosure_cs']}_", ""]
    lines += [f"- {reason}" for reason in judged.reasons]
    v, frame = judged.verdict, judged.frame
    if v is not None and frame is not None:
        lines += [
            "",
            f"Pořadí oblastí: {_fmt(v.rank)} · Q vyšší skupiny: {_fmt(v.q_high)} · rozdíl skupin: {_fmt(v.difference)}",
            "",
            f"## {labels['forecast_cs']}",
            "",
            "| oblast | Q | srovnatelná kola | vybrané | nevybrané | zrušené | stav |",
            "|---|---|---|---|---|---|---|",
        ]
        for region, row in frame.iterrows():
            lines.append(
                f"| {region} | {_fmt(row['quality'])} | {row['comparable_rounds']} | {row['chosen']} | "
                f"{row['unchosen']} | {row['cancelled']} | {row['integrity']} |"
            )
        lines += [
            "",
            f"## {labels['observed_cs']}",
            "",
            "| oblast | marže zákazníků (Kč) | náklad stojanů (Kč) | dárky (Kč) | bez prodeje (Kč) | flotila (Kč) | "
            "nevyužito (Kč) |",
            "|---|---|---|---|---|---|---|",
        ]
        for region, row in frame.iterrows():
            money = ("observed_margin", "asset_use_cost", "handout_cost", "no_sales_cost", "fleet_cost", "idle_cost")
            lines.append(f"| {region} | " + " | ".join(_czk(row[name]) for name in money) + " |")
    lines += ["", labels["unavailable_cs"]]
    return "\n".join(lines) + "\n"


def _czk(value: float) -> str:
    """Czech money: a space between thousands, a decimal comma."""
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def _fmt(value: Any) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.4f}"
