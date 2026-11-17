"""S5's observed pilot evaluation (G2 S2.7; NH-D21-B design08 §§6 and 8): the pandas twin of the pilot's S5 SQL
(`data/stories/pilot/s5_rank.sql` and `s5_group.sql`) and the judge of a pilot build.

**The frame** (`regions`): the SQL's shared block, `region return`, row for row: per region its complete pairs, the
programme and the holdout customers' whole Customer Margin in W, the baseline adjustment (W's days × each customer's
stored baseline a day, programme less holdout), the gain, the display and the give-away cost, both coverages, the
cohort counts and `integrity` (the block's comment says what each column means and the order the failures are judged
in). Money is summed in haléře: each order line's Customer Margin as `sales_pl` makes it — list × cases, rounded half
away from zero to the haléř, less its discounts, its delivery and its warehouse cost, each rounded the same way — so
the twin's sums are the SQL's to the haléř; only the baseline (× W's days ÷ its days) and an asset's cost (÷
`LIFE_DAYS`) divide. The baseline is the assignment's stored `baseline_customer_margin`: the PoS-free months are the
warm-up's, never written, so it is never rebuilt from the exported sales.

**The metrics**: `rank`, Spearman's rho of the six regions' returns against the story's declared skills (`SKILL`, read
only after the returns are computed; a control world is ranked against them too), 0 when the six returns are equal;
`claim`, the high group's mean gain ÷ cost (`HIGH`), the weak group's (`LOW`) and their ratio, NaN when the weak mean is
not positive. Both are NaN unless every region is valid, as the SQL's NULL.

**The judge** (`judge`): a build's S5 status. NOT_CERTIFIED: no pilot build of this protocol in a scored mode (`SCORED`;
the no-PoS modes are the references), a stale one (its design, revision, window, protocol file or world.yaml differs)
or one short of W. INCONCLUSIVE: a source check fails — a table the twin reads is missing; `build.json`'s written
orders or lines of a quarter are not the CSVs'; an order line without its order, a discount without its line, an order
or an assigned customer without its customer, a loan without its asset, a give-away without its visit — or a region is
not valid. Else PASS when the claim holds (`verdict`: rank ≥ the protocol's bar, the high group's net return positive,
E_H > 1, the weak group's gain ÷ cost positive, E_L > 0, and E_H ÷ E_L ≥ the group bar) and REJECTED when it does not.
A missing export is never a zero, and a short or broken build never a pass. A PASS here is one development build's
claim, never S5's certification: that is S2.8's protocol.

It reads the observed tables only (`TABLES`, `_sim_meta.csv`, `build.json`): never a skill, an uplift, a truth gain or
another world.
"""

from __future__ import annotations

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
from hartland_sim.config import WORLD_FILE, WorldConfig
from hartland_sim.dates import quarter_start
from hartland_sim.protocol import PROTOCOL_FILE, Protocol
from hartland_sim.world import META_FILE, Meta

Frame = pd.DataFrame

SQL = ("pilot/s5_rank.sql", "pilot/s5_group.sql")  # S2.8 makes them s5.sql and s5_best_worst.sql
REVISION = "s5-pilot-v1"  # the SQL's `pilot.revision`: the protocol's pilot_revision
LIFE_DAYS = 36 * 30.4  # the SQL's `pilot.life_days`: pos_asset_life_months × the protocol's asset_days_per_month
COVERAGE = 0.90  # the SQL's `pilot.coverage`: the protocol's acceptance coverage
SKILL = {"PHA": 1.4, "JMV": 1.3, "SJC": 1.0, "ZSC": 0.6, "VCH": 0.5, "SMS": 0.5}  # s5_rank.sql's: s5.yaml's pos_skill
HIGH, LOW = ("PHA", "JMV"), ("ZSC", "VCH", "SMS")  # s5_group.sql's: the protocol's acceptance high and low
SCORED = ("story", "skill-neutral", "full-null")  # the modes judged; the no-PoS ones are their references
STATUSES = ("PASS", "REJECTED", "INCONCLUSIVE", "NOT_CERTIFIED")
TABLES = (
    "pos_pilot_assignment",
    "region",
    "sales_order",
    "sales_order_line",
    "sales_order_line_discount",
    "pos_loan",
    "pos_asset",
    "pos_handout",
    "visit",
    "customer",
    "store",
    "area",
)
COLUMNS = (  # the frame's, the SQL's `region_return` but its index
    "pairs",
    "programme_margin",
    "holdout_margin",
    "baseline_adjustment",
    "gain",
    "asset_cost",
    "handout_cost",
    "cost",
    "loan_coverage",
    "cost_coverage",
    "unpaired",
    "cross_region",
    "potential_at_entry",
    "lost",
    "gain_per_cost",
    "net_return",
    "integrity",
)
ONE_DAY = pd.Timedelta(days=1)


# ── the frame ──────────────────────────────────────────────────────────────────


def _haler(values: pd.Series[Any]) -> np.ndarray[Any, Any]:
    """CZK with two decimals as integer haléře (NULL as 0)."""
    return np.asarray(np.round(values.astype(float).fillna(0.0).to_numpy() * 100), dtype=np.int64)


def _rounded(price: np.ndarray[Any, Any], cases: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """price (haléře) × cases (thousandths) to the haléř, half away from zero: SQL's ROUND(price × cases, 2)."""
    raw = price * cases
    return np.asarray(np.sign(raw) * ((np.abs(raw) + 500) // 1000), dtype=np.int64)


def margins(build: Build) -> Frame:
    """Every order line's Customer Margin in haléře (module docstring), with its order's customer and day."""
    lines = build.narrow(
        "sales_order_line",
        (
            "sales_order_line_id",
            "sales_order_id",
            "qty_cases",
            "list_price_case",
            "warehouse_cost_case",
            "delivery_cost_case",
        ),
    )
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
    orders = build.narrow("sales_order", ("sales_order_id", "customer_id", "ordered_on"))
    return out.merge(orders, on="sales_order_id")


@dataclass(frozen=True)
class Population:
    """The block's `w`, `assigned` and `paired` (module docstring): the window, the assignment day, whether the
    revision's rows share one window, the last order day, its assigned rows and those of its complete pairs."""

    first: pd.Timestamp
    last: pd.Timestamp
    assigned_on: pd.Timestamp
    one: bool
    as_of: pd.Timestamp
    assigned: Frame
    paired: Frame


def population(build: Build) -> Population:
    """The block's `w`, `assigned` and `paired` of `build`."""
    every = build["pos_pilot_assignment"]
    a = every[every["pilot_revision"] == REVISION]
    one = all(a[c].nunique() == 1 for c in ("window_from", "window_until", "assigned_on"))
    held = a[a["pair_id"].notna()].assign(
        programme=a["arm"] == "programme", holdout=a["arm"] == "holdout", base=a["baseline_customer_margin"].notna()
    )
    pairs = held.groupby("pair_id").agg(
        n=("customer_id", "size"),
        programme=("programme", "sum"),
        holdout=("holdout", "sum"),
        regions=("region_id", "nunique"),
        base=("base", "sum"),
        days=("baseline_days", "count"),
        least=("baseline_days", "min"),
    )
    whole = pairs[
        (pairs["n"] == 2)
        & (pairs["programme"] == 1)
        & (pairs["holdout"] == 1)
        & (pairs["regions"] == 1)
        & (pairs["base"] == 2)
        & (pairs["days"] == 2)
        & (pairs["least"] > 0)
    ]
    paired = a[a["pair_id"].isin(whole.index)]
    as_of = build.narrow("sales_order", ("sales_order_id", "customer_id", "ordered_on"))["ordered_on"].max()
    return Population(a["window_from"].min(), a["window_until"].min(), a["assigned_on"].min(), one, as_of, a, paired)


def outcomes(build: Build) -> Frame:
    """Each paired customer's outcome (the block's `outcome`), by customer_id: its arm, region_id, whole Customer
    Margin in W (`margin`, haléře) and W's days × its baseline a day (`expected`, haléře)."""
    return build.joined("s5_pilot_outcomes", _outcomes)


def _outcomes(build: Build) -> Frame:
    p = population(build)
    lines = margins(build)
    inside = lines[(lines["ordered_on"] >= p.first) & (lines["ordered_on"] <= p.last)]
    sold = inside.groupby("customer_id")["margin"].sum()
    out = p.paired[["customer_id", "arm", "region_id"]].copy()
    out["margin"] = out["customer_id"].map(sold).fillna(0).astype(np.int64)
    days = (p.last - p.first).days + 1 if not (pd.isna(p.first) or pd.isna(p.last)) else math.nan
    out["expected"] = days * _haler(p.paired["baseline_customer_margin"]) / p.paired["baseline_days"].astype(float)
    return out.set_index("customer_id")


def _used(build: Build, p: Population) -> Frame:
    """The block's `used`: every loan with its asset's region and cost, its days in W and their cost."""
    loans = build["pos_loan"][["pos_loan_id", "pos_asset_id", "customer_id", "lent_on", "returned_on"]]
    loans = loans.merge(build["pos_asset"][["pos_asset_id", "region_id", "cost"]], on="pos_asset_id")
    back = loans["returned_on"] - ONE_DAY
    ends = pd.DataFrame({"back": back.fillna(p.as_of), "as_of": p.as_of, "last": p.last})
    loans["first_day"] = pd.DataFrame({"lent": loans["lent_on"], "first": p.first}).max(axis=1)
    loans["last_day"] = ends.min(axis=1)
    days = (loans["last_day"] - loans["first_day"]).dt.days + 1
    loans["days"] = days.clip(lower=0).fillna(0).astype(np.int64)
    loans["asset_cost"] = loans["cost"].astype(float) * loans["days"] / LIFE_DAYS
    return loans


def _handed(build: Build, p: Population) -> Frame:
    """The block's `handed`: the give-aways in W at small-retail customers, with their store's region."""
    visits = build["visit"][["visit_id", "customer_id", "store_id", "day"]]
    handed = build["pos_handout"][["visit_id", "cost"]].merge(visits, on="visit_id")
    customers = build["customer"][["customer_id", "channel"]]
    handed = handed.merge(customers, on="customer_id")
    places = build["store"][["store_id", "area_id"]].merge(build["area"][["area_id", "region_id"]], on="area_id")
    handed = handed.merge(places, on="store_id")
    keep = (handed["channel"] == "small_retail") & (handed["day"] >= p.first) & (handed["day"] <= p.last)
    return handed.loc[keep, ["customer_id", "region_id", "cost"]]


def regions(build: Build) -> Frame:
    """The SQL's `region_return` (module docstring), by region code."""
    return build.joined("s5_pilot_regions", _regions)


def _regions(build: Build) -> Frame:
    p = population(build)
    out = outcomes(build)
    used, handed = _used(build, p), _handed(build, p)
    paired = p.paired.set_index("customer_id")
    programme = paired.index[paired["arm"] == "programme"]
    customers = build["customer"][["customer_id", "channel", "since", "until"]].set_index("customer_id")

    loose = p.assigned[(p.assigned["arm"].isin(["programme", "holdout"]) | p.assigned["pair_id"].notna())]
    faults = {"pairs": int((~loose["customer_id"].isin(paired.index)).sum())}
    lent = used.loc[(used["lent_on"] >= p.assigned_on) & (used["lent_on"] <= p.last), "customer_id"]
    treated = pd.concat([lent, handed["customer_id"]], ignore_index=True).to_frame()
    treated = treated.merge(customers["channel"], left_on="customer_id", right_index=True)
    treated = treated.merge(p.assigned[["customer_id", "arm"]], on="customer_id", how="left")
    faults["contamination"] = int(((treated["channel"] == "small_retail") & (treated["arm"] != "programme")).sum())
    out_days = used[used["days"] > 0]
    twice = out_days.merge(out_days, on="pos_asset_id", suffixes=("", "_y"))
    twice = twice[
        (twice["pos_loan_id_y"] > twice["pos_loan_id"])
        & (twice["first_day"] <= twice["last_day_y"])
        & (twice["first_day_y"] <= twice["last_day"])
    ]
    faults["overlap"] = len(twice)
    astray = out_days.merge(p.assigned[["customer_id", "region_id"]], on="customer_id", suffixes=("", "_a"))
    faults["astray"] = int((astray["region_id"] != astray["region_id_a"]).sum())

    mine = out["arm"] == "programme"
    gains = (
        out.assign(
            pairs=mine.astype(np.int64),
            programme_margin=out["margin"].where(mine, 0),
            holdout_margin=out["margin"].where(~mine, 0),
            adjustment=out["expected"].where(mine, -out["expected"]),
        )
        .groupby("region_id")[["pairs", "programme_margin", "holdout_margin", "adjustment"]]
        .sum()
    )
    region_of = paired["region_id"].rename("region_p")
    displayed = used[used["customer_id"].isin(programme)].merge(region_of, left_on="customer_id", right_index=True)
    displays = displayed.groupby("region_p")["asset_cost"].sum()
    given = handed[handed["customer_id"].isin(programme)].merge(region_of, left_on="customer_id", right_index=True)
    giveaways = given.groupby("region_p")["cost"].sum()
    held = used.assign(paired=used["customer_id"].isin(paired.index))
    loans = held.groupby("region_id")["asset_cost"].sum()
    loans_paired = held[held["paired"]].groupby("region_id")["asset_cost"].sum()
    out_handed = handed.assign(paired=handed["customer_id"].isin(paired.index))
    handouts = out_handed.groupby("region_id")["cost"].sum()
    handouts_paired = out_handed[out_handed["paired"]].groupby("region_id")["cost"].sum()
    cohort = _cohort(p, paired, customers)

    rows = []
    for region_id, code in build["region"][["region_id", "code"]].itertuples(index=False):
        g = gains.loc[region_id] if region_id in gains.index else None
        asset_cost = float(displays.get(region_id, 0.0))
        handout_cost = float(giveaways.get(region_id, 0.0))
        loan_total, loan_paired = float(loans.get(region_id, 0.0)), float(loans_paired.get(region_id, 0.0))
        whole = loan_total + float(handouts.get(region_id, 0.0))
        covered = loan_paired + float(handouts_paired.get(region_id, 0.0))
        k = cohort.loc[region_id] if region_id in cohort.index else None
        row: dict[str, Any] = {
            "region": code,
            "pairs": int(g["pairs"]) if g is not None else 0,
            "programme_margin": g["programme_margin"] / 100 if g is not None else math.nan,
            "holdout_margin": g["holdout_margin"] / 100 if g is not None else math.nan,
            "baseline_adjustment": g["adjustment"] / 100 if g is not None else math.nan,
            "asset_cost": asset_cost,
            "handout_cost": handout_cost,
            "cost": asset_cost + handout_cost,
            "loan_coverage": loan_paired / loan_total if loan_total else math.nan,
            "cost_coverage": covered / whole if whole else math.nan,
            **{name: int(k[name]) if k is not None else 0 for name in ("unpaired", "cross_region")},
            **{name: int(k[name]) if k is not None else 0 for name in ("potential_at_entry", "lost")},
        }
        row["gain"] = row["programme_margin"] - row["holdout_margin"] - row["baseline_adjustment"]
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(COLUMNS), index=pd.Index([], name="region"))
    frame = pd.DataFrame(rows).set_index("region")
    priced = frame["cost"] > 0
    frame["gain_per_cost"] = (frame["gain"] / frame["cost"]).where(priced)
    frame["net_return"] = frame["gain_per_cost"] - 1
    frame["integrity"] = _integrity(frame, p, faults)
    return frame[list(COLUMNS)]


def _cohort(p: Population, paired: Frame, customers: Frame) -> Frame:
    """The block's `cohort`, by region_id."""
    a = p.assigned.merge(customers[["since", "until"]], left_on="customer_id", right_index=True, how="left")
    inside = a["customer_id"].isin(paired.index)
    counts = pd.DataFrame(
        {
            "region_id": a["region_id"],
            "unpaired": a["outside_reason"] == "unpaired",
            "cross_region": a["outside_reason"] == "cross_region",
            "potential_at_entry": inside & (a["since"].isna() | (a["since"] >= a["assigned_on"])),
            "lost": inside & (a["until"] <= a["window_until"]),
        }
    )
    return counts.groupby("region_id").sum()


def _integrity(frame: Frame, p: Population, faults: dict[str, int]) -> list[str]:
    """The block's `integrity` per region (module docstring): the build's failures first, then the region's."""
    if not p.one:
        build = "window"
    elif pd.isna(p.as_of) or p.as_of < p.last:
        build = "short"
    else:
        build = next((name for name in ("pairs", "contamination", "overlap", "astray") if faults[name] > 0), "")
    if not build and (frame["pairs"] == 0).any():
        build = "regions"
    out = []
    costs, loans, totals = (frame[name].to_numpy(dtype=float) for name in ("cost", "loan_coverage", "cost_coverage"))
    for cost, loan, total in zip(costs, loans, totals, strict=True):
        if build:
            out.append(build)
        elif not cost > 0:
            out.append("cost")
        elif not (loan >= COVERAGE and total >= COVERAGE):
            out.append("coverage")
        else:
            out.append("valid")
    return out


# ── the metrics ────────────────────────────────────────────────────────────────


def _valid(frame: Frame) -> bool:
    return len(frame) == len(SKILL) and set(frame.index) == set(SKILL) and bool((frame["integrity"] == "valid").all())


def rank(frame: Frame) -> float:
    """s5_rank.sql (module docstring)."""
    if not _valid(frame):
        return math.nan
    returns = frame["gain_per_cost"].rank(method="average")
    if returns.min() == returns.max():
        return 0.0
    skills = pd.Series(SKILL)[frame.index].rank(method="average")
    return float(np.corrcoef(returns.to_numpy(), skills.to_numpy())[0, 1])


def claim(frame: Frame) -> tuple[float, float, float]:
    """s5_group.sql's parts (module docstring): E_H, E_L and E_H ÷ E_L (NaN unless valid and E_L > 0)."""
    ratio = frame["gain_per_cost"].reindex(list(SKILL))
    high, low = float(ratio[list(HIGH)].mean()), float(ratio[list(LOW)].mean())
    if not _valid(frame) or not low > 0:
        return high, low, math.nan
    return high, low, high / low


@dataclass(frozen=True)
class Verdict:
    """A frame's claim against the protocol's bars (`verdict`): PASS or REJECTED for a valid frame, else
    INCONCLUSIVE; the metrics and whether each half of the claim holds."""

    status: str
    rank: float
    e_high: float
    e_low: float
    ratio: float
    rank_holds: bool
    group_holds: bool
    economic: bool  # E_H > 1 and E_L > 0: the economic claim's signs


def verdict(frame: Frame, protocol: Protocol) -> Verdict:
    bars = protocol.acceptance
    rho, (high, low, ratio) = rank(frame), claim(frame)
    economic = high > 1 and low > 0
    rank_holds = rho >= float(bars["rank"])
    group_holds = economic and ratio >= float(bars["group_ratio"])
    if not _valid(frame) or math.isnan(rho):
        status = "INCONCLUSIVE"
    else:
        status = "PASS" if rank_holds and group_holds else "REJECTED"
    return Verdict(status, rho, high, low, ratio, rank_holds, group_holds, economic)


# ── the judge ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Judged:
    """A build's S5 judgement (`judge`): its status and why, its verdict and frame (None when NOT_CERTIFIED), its
    identifiers and the small-retail customers registered after the assignment (in no arm)."""

    run: Path
    mode: str | None
    status: str
    reasons: tuple[str, ...]
    verdict: Verdict | None
    frame: Frame | None
    identity: dict[str, Any]
    later_acquired: int


def identity(run: Path) -> dict[str, Any]:
    """The build's design, source and window identifiers: `_sim_meta`'s and `build.json`'s."""
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
    if (run / "build.json").is_file():
        held = json.loads((run / "build.json").read_text(encoding="utf-8"))
        out |= {"git_sha": held.get("git_sha"), "git_dirty": held.get("git_dirty")}
    return out


def stale(run: Path, protocol: Protocol) -> list[str]:
    """Why `run` is no complete scored pilot build of this protocol (module docstring); empty when it is one."""
    if not (run / META_FILE).is_file():
        return ["no _sim_meta.csv"]
    meta = Meta.load(run)
    held = dict(meta.pilot)
    mode = held.get("pilot_mode")
    if mode is None:
        return ["not a pilot build"]
    if mode not in SCORED:
        return [f"a {mode} build is a reference, not scored"]
    window = f"{protocol.window_from.isoformat()}/{protocol.window_until.isoformat()}"
    expected = {
        "pilot_design": protocol.design,
        "pilot_revision": protocol.pilot_revision,
        "pilot_window": window,
        "pilot_protocol_sha256": hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest(),
    }
    out = [f"stale {key}" for key, value in expected.items() if held.get(key) != value]
    if meta.world_sha256 != hashlib.sha256(WORLD_FILE.read_bytes()).hexdigest():
        out.append("stale world.yaml")
    if meta.last_day_done < protocol.window_until:
        out.append(f"short: done to {meta.last_day_done}")
    return out


def sources(build: Build, first_day: date) -> list[str]:
    """The source checks (module docstring); empty when they hold."""
    try:
        orders = build.narrow("sales_order", ("sales_order_id", "customer_id", "ordered_on"))
        lines = build.narrow("sales_order_line", ("sales_order_line_id", "sales_order_id"))
        discounts = build.narrow("sales_order_line_discount", ("sales_order_line_id", "amount_line"))
        customers = build["customer"]["customer_id"]
        assigned = build["pos_pilot_assignment"]["customer_id"]
        loans, assets = build["pos_loan"]["pos_asset_id"], build["pos_asset"]["pos_asset_id"]
        handouts, visits = build["pos_handout"]["visit_id"], build["visit"]["visit_id"]
        for table in ("region", "store", "area"):
            build[table]
    except Missing as error:
        return [str(error)]
    out = []
    for what, ids, among in (
        ("an order line without its order", lines["sales_order_id"], orders["sales_order_id"]),
        ("a discount without its line", discounts["sales_order_line_id"], lines["sales_order_line_id"]),
        ("an order without its customer", orders["customer_id"], customers),
        ("an assigned customer without its row", assigned, customers),
        ("a loan without its asset", loans, assets),
        ("a give-away without its visit", handouts, visits),
    ):
        missing = int((~ids.isin(among)).sum())
        if missing:
            out.append(f"{what} ({missing})")
    out += _manifest(build, orders, lines, first_day)
    return out


def _manifest(build: Build, orders: Frame, lines: Frame, first_day: date) -> list[str]:
    """`build.json`'s written orders and lines by quarter (a prolong's added) against the CSVs'."""
    path = build.run / "build.json"
    if not path.is_file():
        return ["no build.json"]
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
    names = sorted(set(counted) | set(found["orders"].index))
    for name in names:
        year, q = name.split("-Q")
        if date(int(year), 3 * int(q) - 2, 1) < written:
            continue
        for kind in ("orders", "lines"):
            expected, there = counted.get(name, {}).get(kind, 0), int(found[kind].get(name, 0))
            if expected != there:
                out.append(f"{name} {kind}: build.json {expected}, CSV {there}")
    return out


def judge(run: Path, protocol: Protocol | None = None, build: Build | None = None) -> Judged:
    """`run`'s S5 status (module docstring)."""
    protocol = protocol or Protocol.load()
    build = build or Build(run)
    ident = identity(run)
    mode = ident.get("pilot_mode")
    why = stale(run, protocol)
    if why:
        return Judged(run, mode, "NOT_CERTIFIED", tuple(why), None, None, ident, 0)
    broken = sources(build, WorldConfig.load().history.first_day)
    if broken:
        return Judged(run, mode, "INCONCLUSIVE", tuple(broken), None, None, ident, 0)
    frame = regions(build)
    found = verdict(frame, protocol)
    reasons = tuple(f"{region}: {state}" for region, state in frame["integrity"].items() if state != "valid")
    customers = build["customer"]
    small = customers.loc[customers["channel"] == "small_retail", "customer_id"]
    later = int((~small.isin(build["pos_pilot_assignment"]["customer_id"])).sum())
    return Judged(run, mode, found.status, reasons, found, frame, ident, later)
