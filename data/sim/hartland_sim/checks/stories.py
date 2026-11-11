"""`stories-check --csv` (contracts §3): the detectors' pandas twins over a CSV build, `build/<run>/*.csv`.

Each twin computes what its detector's SQL (`data/stories/detectors/*.sql`) computes, over the same tables, with the
same windows and the thresholds of its story's YAML; the weeks, lines and street types it reads come from the YAML,
as the SQL's copies do (a test keeps the two equal), and a NaN where the SQL gives a NULL. A twin whose tables are not
in the build yet is **not run**, and says which stage writes them; a story switched off (`enabled: false`) is not
planted, so its checks are not run either. The runner judges each enabled metric as the database runner will
(G5 S5.2): a story passes when `metric direction threshold` holds; a NULL (NaN) metric fails; with `--null` the build
is a null world and each metric must give the opposite (`null_world`).

**Built here (G1 S1.2):** S1 (`s1`, `s1_weather`), S2 (`s2` — not run until `promotion_result` exists, G4 S4.3 —
and `s2_affluence`), S4 (`s4`, and 04's k-means check `s4_kmeans`, which has no SQL: s4.yaml `kmeans`).
S2-affluence counts an order only where the promotion names its channel (ttr-hartland#8 item 1, the twin's default
until the SQL takes it). **G1 S1.4:** S6 (`s6`, and `s6_targets` — not run until `promotion_result` exists), its
measure re-cut in S1.5 (`measured`). **G2 S2.1:** the field's probes, A-6 (`a6`), A-7 (`a7`), A-8 (`a8`). **G2 S2.3:**
S5 (`s5`, `s5_best_worst`, their shared region block `s5_regions`) and A-6's whole arc (`a6_arc`: the sightings, the
facings, the dip). The others (S3, the other probes) arrive with their stages.

**The probes** (`anomalies.yaml`): a probe returns one count, judged by its `expect` (`>= n`, `= n`, or `rows` for
any row at all); with `--null` it must not hold — with the anomaly off the probe finds nothing. A disabled anomaly's
probe is not run, nor one without a twin.

**`sales_pl`** (S5, S6): the view's columns the twins read, as the view makes them — the store (the delivery point's),
the cases, the Customer Margin (gross revenue less every discount, the delivery and the warehouse cost, a NULL cost as
0; no COGS, contracts §4) and the agreed discount. Before G3/G4 write the costs, the Customer Margin is the net
revenue.

The report goes to `build/<run>/stories.md`: story · check · metric · threshold · result, and a note where a twin
has more to say (S1: each line's ratio, so the line that sets the margin shows).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hartland_sim.config import Detector, StoriesConfig, inventory
from hartland_sim.ids import row_id

S2_FROM = date(2024, 2, 26)  # s2_affluence.sql: promotions whose 8 weeks before lie inside the history
S6_FROM = date(2024, 2, 26)  # s6.sql: agreements whose 8 weeks before lie inside the history
S6_AFTER = 14  # s6.sql: the window runs a fortnight past the agreement, the small-retail cadence
S6_SLOT_DAY = date(2023, 1, 1)  # s6.sql: a pinned rep is the one holding its area on the warm-up's first day
WEEK_DAYS = 56  # s2_affluence.sql's 8 weeks before
KMEANS_BAR = 0.6  # 04 §S4's adjusted Rand index, reported beside S4 (s4.yaml `kmeans`), never judged
PLACED_CHANNELS = ("web", "own_shops")  # sold where delivered without a store (s1_weather.sql, s2_affluence.sql)
Frame = pd.DataFrame


class Missing(Exception):
    """A table the twin needs is not in the build; the message names the stage that writes it."""


WRITERS = {
    "promotion_result": "G4 S4.3",
    "visit": "G2 S2.1",
    "visit_plan": "G2 S2.1",
    "competitor_sighting": "G2 S2.1",
    "visit_observation": "G2 S2.1",
    "sales_order": "G1 S1.4",
    "sales_order_line": "G1 S1.4",
    "sales_order_line_discount": "G1 S1.4",
    "price_agreement": "G1 S1.4",
    "pos_loan": "G2 S2.3",
    "pos_asset": "G2 S2.3",
    "pos_handout": "G2 S2.3",
}


@dataclass
class Build:
    """The CSV build in `run`: each table read once, when a twin first asks for it, its columns typed by the
    inventory (dates as `datetime64`, booleans as `bool`); the joins several twins share are made once (`joined`)."""

    run: Path
    _frames: dict[str, Frame] | None = None
    _joined: dict[str, Frame] | None = None

    def joined(self, name: str, make: Callable[[Build], Frame]) -> Frame:
        if self._joined is None:
            self._joined = {}
        if name not in self._joined:
            self._joined[name] = make(self)
        return self._joined[name]

    def __getitem__(self, table: str) -> Frame:
        if self._frames is None:
            self._frames = {}
        if table not in self._frames:
            path = self.run / f"{table}.csv"
            if not path.is_file():
                raise Missing(f"no {table}.csv ({WRITERS.get(table, 'its stage')} writes it)")
            self._frames[table] = read(path, table)
        return self._frames[table]


def read(path: Path, table: str) -> Frame:
    """`path` as `table`'s rows (the CSV writer's format: ISO dates, `true`/`false`, empty = NULL)."""
    columns = {column.name: column.type for column in inventory().table(table).columns}
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype=str)
    for name in frame.columns:
        kind = columns.get(name, "text")
        if kind in ("bigint", "integer"):  # parsed as integers: a float would round a 19-digit id
            missing = frame[name].isna().to_numpy()
            values = frame[name].fillna("0").astype("int64").to_numpy()
            frame[name] = pd.arrays.IntegerArray(values, missing)
        elif kind.startswith("numeric"):
            frame[name] = pd.to_numeric(frame[name])
        elif kind == "date":
            frame[name] = pd.to_datetime(frame[name])
        elif kind == "boolean":
            frame[name] = frame[name].map({"true": True, "false": False})
    return frame


def _lines(build: Build) -> Frame:
    """The order lines with their order's day, channel and delivery point, and the SKU's product line code."""
    return build.joined("lines", _join_lines)


def _join_lines(build: Build) -> Frame:
    lines = build["sales_order_line"][["sales_order_id", "sku_id", "qty_cases", "is_free_goods"]]
    orders = build["sales_order"][["sales_order_id", "ordered_on", "channel", "delivery_point_id"]]
    sku = build["sku"][["sku_id", "product_id"]]
    product = build["product"][["product_id", "product_line_id"]]
    line = build["product_line"][["product_line_id", "code"]].rename(columns={"code": "line"})
    out = lines.merge(orders, on="sales_order_id").merge(sku, on="sku_id").merge(product, on="product_id")
    return out.merge(line, on="product_line_id")


def _weeks(ranges: Any) -> list[int]:
    out: list[int] = []
    for first, last in ranges:
        out += list(range(first, last + 1)) if first <= last else [*range(first, 54), *range(1, last + 1)]
    return out


# ── S1 ─────────────────────────────────────────────────────────────────────────


def _weekly(build: Build) -> Frame:
    """Each line's cases by ISO year and week, 2024–2025, free goods left out."""
    return build.joined("weekly", _join_weekly)


def _join_weekly(build: Build) -> Frame:
    lines = _lines(build)
    lines = lines[~lines["is_free_goods"].astype(bool)]
    iso = lines["ordered_on"].dt.isocalendar()
    lines = lines.assign(iso_year=iso["year"].astype(int), week=iso["week"].astype(int))
    lines = lines[lines["iso_year"].between(2024, 2025)]
    return lines.groupby(["line", "iso_year", "week"])["qty_cases"].sum().reset_index()


def s1_ratios(build: Build, stories: StoriesConfig) -> dict[str, float]:
    """s1.sql's per-line ratios: for each line with a curve's peak, in the YAML's order, the mean weekly cases of its
    peak weeks ÷ its trough weeks (NaN where the SQL's is NULL). A range that wraps the new year takes week 53 too, as
    the SQL's arrays."""
    weekly = _weekly(build)
    out = {}
    for code, spec in stories.story("S1").parameters["curves"].items():
        peak = _weeks(spec["peak_weeks"])
        if not peak:
            continue
        mine = weekly[weekly["line"] == code]
        up = mine.loc[mine["week"].isin(peak), "qty_cases"].mean()
        down = mine.loc[mine["week"].isin(_weeks(spec["trough_weeks"])), "qty_cases"].mean()
        out[code] = float(up / down) if down else math.nan
    return out


def s1(build: Build, stories: StoriesConfig) -> float:
    """s1.sql: the smallest of `s1_ratios`, a line's NULL skipped as SQL's MIN does."""
    found = [ratio for ratio in s1_ratios(build, stories).values() if not math.isnan(ratio)]
    return float(min(found)) if found else math.nan


def s1_note(build: Build, stories: StoriesConfig) -> str:
    """Each line's ratio, so the report shows which line sets S1's margin."""
    return " · ".join(
        f"{code} {'—' if math.isnan(ratio) else f'{ratio:.3f}'}" for code, ratio in s1_ratios(build, stories).items()
    )


def _placed(build: Build) -> Frame:
    """The order lines delivered where they are sold (a store, an own shop, a web consumer's town), with the place's
    okres and region."""
    return build.joined("placed", _join_placed)


def _join_placed(build: Build) -> Frame:
    lines = _lines(build)
    points = build["delivery_point"][["delivery_point_id", "store_id", "okres_id"]]
    lines = lines.merge(points, on="delivery_point_id")
    lines = lines[lines["store_id"].notna() | lines["channel"].isin(PLACED_CHANNELS)]
    area = build["area_okres"].merge(build["area"][["area_id", "region_id"]], on="area_id")
    return lines.merge(area[["okres_id", "region_id"]], on="okres_id")


def s1_weather(build: Build, stories: StoriesConfig) -> float:
    """s1_weather.sql: Pearson's r of each region's weekly dried-soup cases (÷ the region's mean) against its weekly
    mean temperature, 2024–2025."""
    lines = _placed(build)
    lines = lines[(lines["line"] == "POLEVKY") & lines["ordered_on"].between("2024-01-01", "2025-12-31")]
    lines = lines.assign(week=lines["ordered_on"].dt.to_period("W-SUN").dt.start_time)
    cases = lines.groupby(["region_id", "week"], as_index=False)["qty_cases"].sum()
    cases["cases"] = cases["qty_cases"] / cases.groupby("region_id")["qty_cases"].transform("mean")
    weather = build["weather_daily"]
    weather = weather[weather["day"].between("2024-01-01", "2025-12-31")]
    weather = weather.assign(week=weather["day"].dt.to_period("W-SUN").dt.start_time)
    temps = weather.groupby(["region_id", "week"], as_index=False)["temp_mean_c"].mean()
    joined = cases.merge(temps, on=["region_id", "week"])
    if len(joined) < 2:
        return math.nan
    return float(joined["cases"].corr(joined["temp_mean_c"]))


# ── S2 ─────────────────────────────────────────────────────────────────────────


def s2(build: Build, stories: StoriesConfig) -> float:
    """s2.sql: the TV-supported promotions' mean weekly uplift (`promotion_result`) on premium lines ÷ on the others,
    mechanic by mechanic, averaged over the mechanics both sides ran."""
    results = build["promotion_result"]
    promotions = build["promotion"].merge(build["campaign"][["campaign_id", "has_tv"]], on="campaign_id")
    promotions = promotions[promotions["has_tv"].astype(bool)]
    lines = build["product_line"][["product_line_id", "is_premium"]]
    promotions = promotions.merge(lines, left_on="target_product_line_id", right_on="product_line_id")
    each = results.groupby("promotion_id", as_index=False)["uplift_pct"].mean()
    each = each.merge(promotions[["promotion_id", "mechanic", "discount_pct", "is_premium"]], on="promotion_id")
    ratios = []
    for _, group in each.groupby(["mechanic", "discount_pct"], dropna=False):
        premium = group.loc[group["is_premium"].astype(bool), "uplift_pct"]
        basic = group.loc[~group["is_premium"].astype(bool), "uplift_pct"]
        if len(premium) and len(basic) and basic.mean():
            ratios.append(premium.mean() / basic.mean())
    return float(np.mean(ratios)) if ratios else math.nan


def _ntile(values: Frame, column: str, n: int) -> pd.Series[int]:
    """SQL's NTILE(n) OVER (ORDER BY column): the rows in order, cut into n groups whose sizes differ by at most one,
    the larger first (ties broken by the frame's order)."""
    order = values.sort_values(column, kind="stable").index
    groups = np.concatenate(
        [np.full(len(part), q + 1) for q, part in enumerate(np.array_split(np.arange(len(order)), n))]
    )
    return pd.Series(groups, index=order).reindex(values.index)


def s2_affluence(build: Build, stories: StoriesConfig) -> float:
    """s2_affluence.sql: within the TV-supported promotions, the promoted line's lift (its daily cases during the
    promotion ÷ in the 8 weeks before, clean days only) in the richest quarter of the leaf okresy ÷ the poorest's.
    The cases are those delivered where they are sold, of the channels the promotion names (ttr-hartland#8).

    As the SQL: the history ends on the last order's day, whatever its place; a day is clean when no other promotion
    of the line runs, with a campaign or without; a quarter's lift is NULL — skipped — without orders both during and
    before, or with no cases before."""
    lines = _placed(build)
    orders = build["sales_order"]["ordered_on"]
    as_of = orders.max() if len(orders) else pd.NaT
    if pd.isna(as_of):
        return math.nan
    every = build["promotion"]
    promotions = every.merge(build["campaign"][["campaign_id", "has_tv"]], on="campaign_id")
    tv = promotions[promotions["has_tv"].astype(bool) & promotions["valid_from"].between(pd.Timestamp(S2_FROM), as_of)]
    okres = build["okres"]
    leaves = okres[~okres["okres_id"].isin(okres["parent_okres_id"].dropna())].copy()
    leaves["q"] = _ntile(leaves, "affluence_index", 4)
    lines = lines.merge(leaves[["okres_id", "q"]], on="okres_id")
    channels = build["promotion_eligibility"]
    channels = channels[channels["scope_kind"] == "channel"]
    lifts = []
    for row in tv.to_dict("records"):
        start, line, promotion = pd.Timestamp(row["valid_from"]), row["target_product_line_id"], row["promotion_id"]
        days = pd.date_range(start - pd.Timedelta(days=WEEK_DAYS), min(pd.Timestamp(row["valid_to"]), as_of))
        others = every[(every["target_product_line_id"] == line) & (every["promotion_id"] != promotion)]
        clean = np.ones(len(days), dtype=bool)
        for other in others.to_dict("records"):
            clean &= ~((days >= pd.Timestamp(other["valid_from"])) & (days <= pd.Timestamp(other["valid_to"])))
        days = days[clean]
        during = int((days >= start).sum())
        before = len(days) - during
        named = set(channels.loc[channels["promotion_id"] == promotion, "scope_value"])
        mine = lines[(lines["product_line_id"] == line) & lines["ordered_on"].isin(days) & lines["channel"].isin(named)]
        is_during = (mine["ordered_on"] >= start).to_numpy()
        for q, group in mine.groupby("q"):
            in_q = is_during[mine.index.get_indexer(group.index)]
            if not (in_q.any() and (~in_q).any()):  # SUM … FILTER over no rows is NULL
                continue
            during_cases = float(group.loc[in_q, "qty_cases"].sum())
            before_cases = float(group.loc[~in_q, "qty_cases"].sum())
            if during and before and before_cases:
                lifts.append((q, (during_cases / during) / (before_cases / before)))
    frame = pd.DataFrame(lifts, columns=["q", "lift"])
    rich, poor = frame.loc[frame["q"] == 4, "lift"].mean(), frame.loc[frame["q"] == 1, "lift"].mean()
    return float(rich / poor) if poor else math.nan


# ── S4 ─────────────────────────────────────────────────────────────────────────


def _store_mix(build: Build) -> Frame:
    """(store × line) cases in 2025, free goods left out, over the stores that ordered."""
    return build.joined("store_mix", _join_store_mix)


def _join_store_mix(build: Build) -> Frame:
    lines = _lines(build)
    points = build["delivery_point"][["delivery_point_id", "store_id"]]
    lines = lines.merge(points, on="delivery_point_id")
    lines = lines[
        lines["ordered_on"].between("2025-01-01", "2025-12-31")
        & lines["store_id"].notna()
        & ~lines["is_free_goods"].astype(bool)
    ]
    return lines.pivot_table(index="store_id", columns="line", values="qty_cases", aggfunc="sum", fill_value=0.0)


def s4(build: Build, stories: StoriesConfig) -> float:
    """s4.sql: per store the line with the most cases in 2025 (a tie to the first line by code); per street type of
    `leading_line`, the share of its ordering stores that line leads; the smallest share — NaN (the SQL's NULL) when a
    street type has no ordering store, as a detector blind to a type proves nothing of it."""
    mix = _store_mix(build)
    if mix.empty:
        return math.nan
    leader = mix[sorted(mix.columns)].idxmax(axis=1)
    stores = build["store"].set_index("store_id")["street_type"]
    shares = []
    for street, line in stories.story("S4").parameters["leading_line"].items():
        mine = leader[stores.reindex(leader.index) == street]
        if not len(mine):
            return math.nan
        shares.append(float((mine == line).mean()))
    return min(shares)


def kmeans(points: np.ndarray[Any, Any], k: int, rounds: int = 100) -> np.ndarray[Any, Any]:
    """Lloyd's k-means with a deterministic start (the point farthest from the mean, then each next point farthest
    from the centres chosen): a check, not the world, so no random stream."""
    centres = [points[np.argmax(((points - points.mean(axis=0)) ** 2).sum(axis=1))]]
    while len(centres) < k:
        apart = np.min([((points - c) ** 2).sum(axis=1) for c in centres], axis=0)
        centres.append(points[np.argmax(apart)])
    centre = np.array(centres)
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(rounds):
        labels = np.argmin(((points[:, None, :] - centre[None, :, :]) ** 2).sum(axis=2), axis=1)
        moved = np.array([points[labels == j].mean(axis=0) if (labels == j).any() else centre[j] for j in range(k)])
        if np.allclose(moved, centre):
            break
        centre = moved
    return labels


def adjusted_rand(first: np.ndarray[Any, Any], second: np.ndarray[Any, Any]) -> float:
    """The adjusted Rand index of two labellings (Hubert & Arabie)."""
    _, a = np.unique(first, return_inverse=True)
    _, b = np.unique(second, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)

    def pairs(x: np.ndarray[Any, Any]) -> float:
        return float((x * (x - 1) / 2).sum())

    index, rows, cols = pairs(table), pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = rows * cols / pairs(np.array([len(a)]))
    top = (rows + cols) / 2
    return (index - expected) / (top - expected) if top != expected else math.nan


def s4_kmeans(build: Build, stories: StoriesConfig) -> float:
    """04's other S4 check: k-means (s4.yaml `kmeans.clusters`) over each store's line-mix shares in 2025, against its
    street type, as the adjusted Rand index."""
    mix = _store_mix(build)
    if mix.empty:
        return math.nan
    shares = mix.div(mix.sum(axis=1), axis=0).to_numpy()
    labels = kmeans(shares, stories.story("S4").parameters["kmeans"]["clusters"])
    stores = build["store"].set_index("store_id")["street_type"].reindex(mix.index).to_numpy()
    return adjusted_rand(labels, stores)


Twin = Callable[[Build, StoriesConfig], float]
# ── S6 ─────────────────────────────────────────────────────────────────────────


def _sales_pl(build: Build) -> Frame:
    """`sales_pl` (contracts §4): the columns S5 and S6 read (module docstring)."""
    return build.joined("sales_pl", _join_sales_pl)


def _join_sales_pl(build: Build) -> Frame:
    columns = ["sales_order_line_id", "sales_order_id", "sku_id", "qty_cases", "list_price_case"]
    lines = build["sales_order_line"][[*columns, "delivery_cost_case", "warehouse_cost_case"]]
    orders = build["sales_order"][["sales_order_id", "customer_id", "delivery_point_id", "ordered_on"]]
    orders = orders.merge(build["delivery_point"][["delivery_point_id", "store_id"]], on="delivery_point_id")
    sku = build["sku"][["sku_id", "product_id"]]
    product = build["product"][["product_id", "product_line_id"]]
    out = lines.merge(orders, on="sales_order_id").merge(sku, on="sku_id").merge(product, on="product_id")
    discounts = build["sales_order_line_discount"]
    total = discounts.groupby("sales_order_line_id")["amount_line"].sum()
    agreed = discounts[discounts["kind"] == "agreed"].groupby("sales_order_line_id")["amount_line"].sum()

    def money(per_case: pd.Series[float]) -> pd.Series[float]:
        return (per_case.fillna(0.0) * out["qty_cases"]).round(2)

    out["customer_margin"] = (
        money(out["list_price_case"])
        - out["sales_order_line_id"].map(total).fillna(0.0)
        - money(out["delivery_cost_case"])
        - money(out["warehouse_cost_case"])
    )
    out["discount_agreed"] = out["sales_order_line_id"].map(agreed).fillna(0.0)
    out = out.rename(columns={"qty_cases": "cases"})
    keep = ["customer_id", "store_id", "product_line_id", "ordered_on", "cases", "customer_margin", "discount_agreed"]
    return out[keep]


# Why `measured` leaves an agreement out (s6.sql's agreement and gain CTEs), each a label for the note, in the order
# they are judged: its window runs past the last order day; another agreement of the customer-line ends in its 8 weeks
# before, or starts in its fortnight after; the customer ordered nothing in the 8 weeks before; the line or the
# customer's other lines have no cases before or in the window.
LEFT_OUT = {
    "history": "past the history",
    "before": "another before",
    "after": "another after",
    "no order": "no order before",
    "no cases": "no cases",
}


def _days(column: pd.Series[Any]) -> np.ndarray[Any, Any]:
    return np.asarray(column.to_numpy("datetime64[D]").astype(np.int64))


def _sums(rows: Frame, keys: list[str], values: list[str], asks: Frame, first: str, last: str) -> Frame:
    """Per row of `asks`, the Σ of `rows`' `values` over the rows of its `keys` ordered from its `first` day to its
    `last` (columns of `asks`): one sort and two binary searches, never a join of the two."""
    keyed = pd.concat([rows[keys], asks[keys]], ignore_index=True)
    code = keyed.groupby(keys, sort=False).ngroup().to_numpy().astype(np.int64)
    span = np.int64(1 << 22)  # more days than 0001-01-01 … 9999-12-31 from 1970
    key = code[: len(rows)] * span + _days(rows["ordered_on"])
    order = np.argsort(key, kind="stable")
    key = key[order]
    totals = np.vstack([np.zeros(len(values)), np.cumsum(rows[values].to_numpy(dtype=float)[order], axis=0)])
    mine = code[len(rows) :] * span
    low = np.searchsorted(key, mine + _days(asks[first]), "left")
    high = np.searchsorted(key, mine + _days(asks[last]), "right")
    return pd.DataFrame(totals[high] - totals[low], columns=values, index=asks.index)


def measured(build: Build, since: date) -> Frame:
    """s6.sql's measure for every agreed discount from `since`, by `price_agreement_id` (Bora's S6 rulings, 11-04):
    who agreed it; the line's and the customer's other lines' cases in the 8 weeks before and in the window — its
    first day to `S6_AFTER` days past its last; the line's Customer Margin and agreed discount (`discount`) in the
    window; the lift, (line ÷ others) in the window ÷ (line ÷ others) before; the gain, the margin less what the line
    would have earned at list without the lift, margin − (margin + discount) ÷ lift; and `left_out`, why it is not
    counted (a key of `LEFT_OUT`, judged in that order), or "" when it is — the lift and the gain then NaN."""
    as_of = build["sales_order"]["ordered_on"].max()
    every = build["price_agreement"]
    found = every[
        (every["kind"] == "agreed_discount")
        & every["product_line_id"].notna()
        & (every["valid_from"] >= pd.Timestamp(since))
    ].reset_index(drop=True)
    one, week, after = pd.Timedelta(days=1), pd.Timedelta(days=WEEK_DAYS), pd.Timedelta(days=S6_AFTER)
    asks = found.assign(before_from=found["valid_from"] - week, before_to=found["valid_from"] - one)
    asks["window_to"] = found["valid_to"] + after
    pl = _sales_pl(build)
    mine, theirs = ["customer_id", "product_line_id"], ["customer_id"]
    line_before = _sums(pl, mine, ["cases"], asks, "before_from", "before_to")["cases"]
    window = _sums(pl, mine, ["cases", "customer_margin", "discount_agreed"], asks, "valid_from", "window_to")
    all_before = _sums(pl, theirs, ["cases"], asks, "before_from", "before_to")["cases"]
    all_window = _sums(pl, theirs, ["cases"], asks, "valid_from", "window_to")["cases"]
    out = pd.DataFrame(
        {
            "agreed_by_employee_id": found["agreed_by_employee_id"],
            "line_before": line_before,
            "others_before": all_before - line_before,
            "line_window": window["cases"],
            "others_window": all_window - window["cases"],
            "margin": window["customer_margin"],
            "discount": window["discount_agreed"],
        }
    )
    keys = ["price_agreement_id", "customer_id", "product_line_id", "valid_from", "valid_to"]
    pairs = found[keys].merge(every[keys], on=["customer_id", "product_line_id"], suffixes=("", "_other"))
    pairs = pairs[pairs["price_agreement_id"] != pairs["price_agreement_id_other"]]
    ends_before = pairs["valid_to_other"].between(pairs["valid_from"] - week, pairs["valid_from"] - one)
    starts_after = pairs["valid_from_other"].between(pairs["valid_to"] + one, pairs["valid_to"] + after)
    none = 0.0005  # below the cases' grain, 0.001: a sum of none, less a float's dust
    shares = ["line_before", "others_before", "line_window", "others_window"]
    reasons = {
        "history": asks["window_to"] > as_of,
        "before": found["price_agreement_id"].isin(pairs.loc[ends_before, "price_agreement_id"]),
        "after": found["price_agreement_id"].isin(pairs.loc[starts_after, "price_agreement_id"]),
        "no order": all_before < none,
        "no cases": (out[shares] < none).any(axis=1),
    }
    assert list(reasons) == list(LEFT_OUT)
    out["left_out"] = np.select([np.asarray(held, dtype=bool) for held in reasons.values()], list(reasons), "")
    counted = out["left_out"] == ""
    lift = (out["line_window"] / out["others_window"]) / (out["line_before"] / out["others_before"])
    out["lift"] = lift.where(counted)
    out["gain"] = (out["margin"] - (out["margin"] + out["discount"]) / lift).where(counted)
    return out.set_index(found["price_agreement_id"].rename("price_agreement_id"))


def gains(build: Build, since: date) -> Frame:
    """s6.sql's counted agreements (`measured`), by `price_agreement_id`: who agreed it, its margin gain and its
    discount crowns."""
    found = measured(build, since)
    return found.loc[found["left_out"] == "", ["agreed_by_employee_id", "gain", "discount"]]


def _pinned_reps(build: Build, stories: StoriesConfig) -> dict[int, bool]:
    """s6.yaml's pinned reps by employee_id — the reps holding the pinned areas on the warm-up's first day — each
    whether it is disciplined (`disciplined_from`)."""
    parameters = stories.story("S6").parameters
    assignment = build["assignment"]
    held = assignment[
        (assignment["kind"] == "rep_area")
        & (assignment["valid_from"] <= pd.Timestamp(S6_SLOT_DAY))
        & (assignment["valid_to"] >= pd.Timestamp(S6_SLOT_DAY))
    ].merge(build["area"][["area_id", "code"]], on="area_id")
    pinned = parameters["pinned_reps"]
    return {
        int(row["employee_id"]): pinned[row["code"]] >= parameters["disciplined_from"]
        for row in held.to_dict("records")
        if row["code"] in pinned
    }


def _pinned(build: Build, stories: StoriesConfig) -> Frame:
    """`measured` from `S6_FROM` (once a check: `s6` and `s6_note` read it), the pinned reps' agreements, each with
    whether its rep is disciplined."""
    pinned = _pinned_reps(build, stories)
    found = build.joined("s6_measured", lambda made: measured(made, S6_FROM))
    found = found[found["agreed_by_employee_id"].isin(list(pinned))]
    return found.assign(disciplined=found["agreed_by_employee_id"].map(pinned).astype(bool))


def s6_reps(build: Build, stories: StoriesConfig) -> dict[bool, float]:
    """The pinned reps' margin gained per discount crown, disciplined (True) and not (False)."""
    found = _pinned(build, stories)
    counted = found[found["left_out"] == ""]
    out = {}
    for side in (True, False):
        mine = counted[counted["disciplined"] == side]
        crowns = float(mine["discount"].sum())
        out[side] = float(mine["gain"].sum()) / crowns if crowns else math.nan
    return out


def s6(build: Build, stories: StoriesConfig) -> float:
    """s6.sql: the disciplined pinned reps' Customer Margin gained per agreed-discount crown minus the undisciplined
    ones' (a difference: the undisciplined gain may be negative)."""
    ratios = s6_reps(build, stories)
    return ratios[True] - ratios[False]


def s6_note(build: Build, stories: StoriesConfig) -> str:
    """Each side's ratio, and the pinned reps' agreements counted and left out (`LEFT_OUT`)."""
    ratios = s6_reps(build, stories)
    found = _pinned(build, stories)
    left = found["left_out"].value_counts()
    why = " · ".join(f"{int(left[code])} {label}" for code, label in LEFT_OUT.items() if code in left)
    note = f"disciplined {ratios[True]:.2f}, undisciplined {ratios[False]:.2f} a crown; "
    note += f"{int(left.get('', 0))} of {len(found)} agreements counted"
    return note + (f" (left out: {why})" if why else "")


def s6_targets(build: Build, stories: StoriesConfig) -> float:
    """s6_targets.sql: the share of the disciplined product manager's promotions whose uplift lands within ±10 % of
    its target, minus the undisciplined one's (s6.yaml `pinned_product_managers`)."""
    results = build["promotion_result"]
    parameters = stories.story("S6").parameters
    lines = build["product_line"][["product_line_id", "code"]]
    promotions = build["promotion"][["promotion_id", "target_product_line_id", "target_uplift_pct"]].merge(
        lines, left_on="target_product_line_id", right_on="product_line_id"
    )
    pinned = parameters["pinned_product_managers"]
    promotions = promotions[promotions["code"].isin(list(pinned))]
    uplift = results.groupby("promotion_id")["uplift_pct"].mean()
    promotions = promotions.join(uplift, on="promotion_id", how="inner")
    on_target = (promotions["uplift_pct"] - promotions["target_uplift_pct"]).abs() <= 0.10 * promotions[
        "target_uplift_pct"
    ]
    disciplined = promotions["code"].map(lambda code: pinned[code] >= parameters["disciplined_from"])
    return float(on_target[disciplined].mean() - on_target[~disciplined].mean())


# ── the probes (G2 S2.1, S2.2) ────────────────────────────────────────────────

S5_FROM = date(2024, 2, 26)  # s5.sql: loans whose 8 weeks before lie inside the history
S5_LIFE_DAYS = 36 * 30.4  # s5.sql: a display's cost spread over 36 months
S5_BEST, S5_WORST = "PHA", "VCH"  # s5_best_worst.sql: the most skilled manager's region, the least (not the generous)


def s5_regions(build: Build) -> Frame:
    """s5.sql's and s5_best_worst.sql's shared block (`region return`): per region with a loan lent from `S5_FROM` to
    the last order day, its displays' gain — each loan's Customer Margin at its store (at its customer, without a
    store) a day during the loan less that of the 56 days before, × the loan's days — and their cost: each display's
    over `S5_LIFE_DAYS` for the loan's days, plus the region's give-aways from `S5_FROM` on."""
    return build.joined("s5_regions", _join_s5_regions)


def _join_s5_regions(build: Build) -> Frame:
    day = build["sales_order"]["ordered_on"].max()
    first = pd.Timestamp(S5_FROM)
    regions = build["region"][["region_id", "code"]].rename(columns={"code": "region"})
    assets = build["pos_asset"][["pos_asset_id", "cost", "region_id"]]
    loans = build["pos_loan"].merge(assets, on="pos_asset_id").merge(regions, on="region_id")
    loans = loans[loans["lent_on"].between(first, day)].copy()
    loans["until"] = loans["returned_on"].fillna(day).clip(upper=day)
    loans["days"] = (loans["until"] - loans["lent_on"]).dt.days + 1
    sales = _sales_pl(build).groupby(["customer_id", "store_id", "ordered_on"], dropna=False, as_index=False)
    margin = sales["customer_margin"].sum()
    at = loans[["pos_loan_id", "customer_id", "store_id", "lent_on", "until"]].merge(
        margin, on="customer_id", suffixes=("", "_sold")
    )
    at = at[
        (at["store_id"].isna() | (at["store_id"] == at["store_id_sold"]))
        & (at["ordered_on"] >= at["lent_on"] - pd.Timedelta(days=WEEK_DAYS))
        & (at["ordered_on"] <= at["until"])
    ]
    during = at[at["ordered_on"] >= at["lent_on"]].groupby("pos_loan_id")["customer_margin"].sum()
    before = at[at["ordered_on"] < at["lent_on"]].groupby("pos_loan_id")["customer_margin"].sum()
    loans["gain"] = (
        loans["pos_loan_id"].map(during).fillna(0.0).astype(float) / loans["days"]
        - loans["pos_loan_id"].map(before).fillna(0.0).astype(float) / WEEK_DAYS
    ) * loans["days"]
    loans["cost"] = loans["cost"] * loans["days"] / S5_LIFE_DAYS
    visits = build["visit"][["visit_id", "store_id", "day"]]
    visits = visits[visits["day"].between(first, day)]
    places = build["store"][["store_id", "area_id"]].merge(build["area"][["area_id", "region_id"]], on="area_id")
    handed = build["pos_handout"][["visit_id", "cost"]].merge(visits, on="visit_id").merge(places, on="store_id")
    giveaways = handed.merge(regions, on="region_id").groupby("region")["cost"].sum()
    out = loans.groupby("region", as_index=False)[["gain", "cost"]].sum()
    out["cost"] = out["cost"] + out["region"].map(giveaways).fillna(0.0)
    return out


def s5(build: Build, stories: StoriesConfig) -> float:
    """s5.sql: Spearman's rho between the regions' returns, (gain − cost) ÷ cost, and their managers' `pos_skill`, on
    average ranks; NaN for fewer than two regions or no spread."""
    skill = stories.story("S5").parameters["pos_skill"]
    found = s5_regions(build)
    found = found[found["region"].isin(list(skill)) & (found["cost"] != 0)]
    if len(found) < 2:
        return math.nan
    returns = ((found["gain"] - found["cost"]) / found["cost"]).rank(method="average")
    skills = found["region"].map(skill).astype(float).rank(method="average")
    rho = float(np.corrcoef(returns, skills)[0, 1]) if returns.std() and skills.std() else math.nan
    return rho


def s5_best_worst(build: Build, stories: StoriesConfig) -> float:
    """s5_best_worst.sql: the best region's gain ÷ cost over the worst's (`S5_BEST`, `S5_WORST`); NaN where the
    worst's is ≤ 0 or either is missing, as the SQL's NULL."""
    found = s5_regions(build).set_index("region")
    ratio = (found["gain"] / found["cost"].where(found["cost"] != 0)).to_dict()
    best, worst = ratio.get(S5_BEST, math.nan), ratio.get(S5_WORST, math.nan)
    if not worst > 0 or math.isnan(best):
        return math.nan
    return float(best / worst)


A7_YEAR = 2025  # a7.sql: the reps' adherence over 2025
A7_SKIPPING = 0.65  # a7.sql: a rep who made this share of its planned visits or fewer skips them
A8_SILENT = 0.6  # a8.sql: a rep whose visits go unrecorded this share of the time or more records nothing


def a6_arc(build: Build, stories: StoriesConfig) -> tuple[float, float, float]:
    """a6.sql's three legs (G2 S2.3 T7): the sightings of the competitor's promotion at visits in the region from its
    sightings to its end; the competitors' facings beside its line in the region's observations over those days ÷
    those of the 8 weeks before; and the dip — the region's customers' cases of the line a day from its `from` to its
    end over the same customers' in the 8 weeks before, ÷ that ratio of the other regions' customers."""
    held = stories.anomaly("A-6").parameters
    first, start, last = (pd.Timestamp(held[key]) for key in ("sightings_from", "from", "until"))
    week8 = pd.Timedelta(days=WEEK_DAYS)
    sightings, observed = build["competitor_sighting"], build["visit_observation"]  # G2 S2.1's: not run before
    regions = build["area"][["area_id", "region_id"]].merge(build["region"][["region_id", "code"]], on="region_id")
    stores = build["store"][["store_id", "area_id"]].merge(regions, on="area_id")
    visits = build["visit"][["visit_id", "store_id", "day"]].merge(stores, on="store_id")
    competitor = row_id("competitor", code=held["competitor"])  # a reference table, loaded from data/reference/
    found = sightings[(sightings["competitor_id"] == competitor) & (sightings["activity_kind"] == "promotion")]
    at = visits.merge(found[["visit_id"]], on="visit_id")
    seen = float(((at["code"] == held["region"]) & at["day"].between(first, last)).sum())
    line = build["product_line"].loc[build["product_line"]["code"] == held["line"], "product_line_id"]
    observed = observed[observed["product_line_id"].isin(line)][["visit_id", "competitor_facings"]]
    observed = observed.merge(visits[visits["code"] == held["region"]], on="visit_id")
    facings = observed["competitor_facings"].astype("float64")  # an empty window's mean is NaN, not NA
    during = facings[observed["day"].between(first, last)].mean()
    before = facings[observed["day"].between(first - week8, first - pd.Timedelta(days=1))].mean()
    lift = float(during / before) if before > 0 else math.nan
    lines = _lines(build)
    lines = lines[(lines["line"] == held["line"]) & lines["ordered_on"].between(start - week8, last)]
    orders = build["sales_order"][["sales_order_id", "customer_id"]]
    points = build["delivery_point"][["delivery_point_id", "store_id", "okres_id"]]
    points = points.merge(build["store"][["store_id", "area_id"]], on="store_id", how="left")
    by_okres = build["area_okres"].set_index("okres_id")["area_id"]
    points["area_id"] = points["area_id"].fillna(points["okres_id"].map(by_okres))
    points = points.merge(regions, on="area_id")
    sold = lines.merge(orders, on="sales_order_id").merge(points[["delivery_point_id", "code"]], on="delivery_point_id")
    sold["east"] = sold["code"] == held["region"]
    days = (last - start).days + 1
    sold["during"] = sold["qty_cases"].where(sold["ordered_on"] >= start, 0.0) / days
    sold["before"] = sold["qty_cases"].where(sold["ordered_on"] < start, 0.0) / WEEK_DAYS
    each = sold.groupby(["east", "customer_id"])[["during", "before"]].sum().reset_index()
    each = each[each["before"] > 0]
    ratio = {east: g["during"].sum() / g["before"].sum() for east, g in each.groupby("east")}
    dip = float(ratio[True] / ratio[False]) if ratio.get(True) is not None and ratio.get(False) else math.nan
    return seen, lift, dip


def a6_bars(stories: StoriesConfig) -> tuple[float, float]:
    """a6.sql's bars: the facings at least 1 + half the planted lift, the dip at most 1 − half the planted cut."""
    held = stories.anomaly("A-6").parameters
    return 1 + float(held["competitor_facings_lift"]) / 2, 1 - float(held["dip_pct"]) / 200


def a6(build: Build, stories: StoriesConfig) -> float:
    """a6.sql (G2 S2.3 T7): the sightings, none unless the facings rose and the dip shows (`a6_arc`, `a6_bars`)."""
    seen, lift, dip = a6_arc(build, stories)
    rose, fell = a6_bars(stories)
    return seen if lift >= rose and dip <= fell else 0.0


def a7(build: Build, stories: StoriesConfig) -> float:
    """a7.sql: the reps who made `A7_SKIPPING` or fewer of their planned visits of `A7_YEAR`."""
    plans = build["visit_plan"]
    plans = plans[plans["planned_day"].dt.year == A7_YEAR]
    made = plans["visit_plan_id"].isin(build["visit"]["visit_plan_id"].dropna())
    adherence = made.groupby(plans["employee_id"]).mean()
    return float((adherence <= A7_SKIPPING).sum())


def a8(build: Build, stories: StoriesConfig) -> float:
    """a8.sql: the reps whose visits go unrecorded `A8_SILENT` of the time or more."""
    visits = build["visit"]
    unrecorded = (~visits["recorded"].astype(bool)).groupby(visits["employee_id"]).mean()
    return float((unrecorded >= A8_SILENT).sum())


def a13(build: Build, stories: StoriesConfig) -> float:
    """a13.sql (G2 S2.2): the customers whose `rejections_until_stop`-th listing prompt of A-13's line rejected with
    A-13's reason is their last listing prompt of the line."""
    held = stories.anomaly("A-13").parameters
    prompts = build["rep_prompt"]
    line = prompts["kind"].eq("unsupplied_items") & prompts["payload"].str.contains(f'"{held["line"]}"', regex=False)
    found = prompts[line].sort_values(["customer_id", "generated_on", "rep_prompt_id"])
    refused = found["outcome"].eq("rejected") & found["outcome_reason"].eq(held["reason_cs"])
    so_far = refused.astype(int).groupby(found["customer_id"]).cumsum()
    after = (so_far >= int(held["rejections_until_stop"])).groupby(found["customer_id"]).sum()
    return float((after == 1).sum())


PROBES: Mapping[str, Twin] = {"probes/a6.sql": a6, "probes/a7.sql": a7, "probes/a8.sql": a8, "probes/a13.sql": a13}


def expected(expect: str) -> tuple[str, float]:
    """A probe's `expect` as a direction and a threshold: `>= n`, `= n`, or `rows` (more than none)."""
    if expect == "rows":
        return ">", 0.0
    direction, number = expect.split()
    return {">=": ">=", "=": "between"}[direction], float(number)


TWINS: Mapping[str, Twin] = {
    "detectors/s1.sql": s1,
    "detectors/s1_weather.sql": s1_weather,
    "detectors/s2.sql": s2,
    "detectors/s2_affluence.sql": s2_affluence,
    "detectors/s4.sql": s4,
    "detectors/s5.sql": s5,
    "detectors/s5_best_worst.sql": s5_best_worst,
    "detectors/s6.sql": s6,
    "detectors/s6_targets.sql": s6_targets,
}
NOTES: Mapping[str, Callable[[Build, StoriesConfig], str]] = {"detectors/s1.sql": s1_note, "detectors/s6.sql": s6_note}


@dataclass(frozen=True)
class Verdict:
    story: str
    check: str  # the detector's SQL, or `kmeans` for S4's twin-only check
    metric: str
    value: float | None  # None: not run
    direction: str
    threshold: float | tuple[float, float]
    passed: bool | None  # None: not run
    note: str = ""


def holds(value: float, direction: str, threshold: float | tuple[float, float]) -> bool:
    """`value direction threshold`; a NaN never holds."""
    if math.isnan(value):
        return False
    if direction == "between":
        assert isinstance(threshold, tuple)
        return threshold[0] <= value <= threshold[1]
    assert isinstance(threshold, float | int)
    return {">=": value >= threshold, "<=": value <= threshold, ">": value > threshold, "<": value < threshold}[
        direction
    ]


def judge(story: str, detector: Detector, value: float, null: bool, note: str = "") -> Verdict:
    """The verdict on one metric: the story's direction, or with `null` the null world's (a NaN fails either way)."""
    direction = detector.null_world if null else detector.direction
    return Verdict(
        story,
        detector.sql,
        detector.metric,
        value,
        direction,
        detector.threshold,
        holds(value, direction, detector.threshold),
        note,
    )


def check(run: Path, stories: StoriesConfig, null: bool = False) -> list[Verdict]:
    """Every detector of every enabled story over the CSV build in `run` (module docstring), in the stories' order,
    and S4's k-means, reported; a disabled story's rows say so; then the probes (`probes`)."""
    build = Build(run)
    out = []
    for story in stories.stories:
        if not story.enabled:
            for detector in story.detectors:
                out.append(
                    Verdict(
                        story.story,
                        detector.sql,
                        detector.metric,
                        None,
                        detector.direction,
                        detector.threshold,
                        None,
                        "disabled",
                    )
                )
            if story.story == "S4":
                out.append(Verdict("S4", "kmeans", "adjusted_rand_index", None, ">=", KMEANS_BAR, None, "disabled"))
            continue
        for detector in story.detectors:
            twin = TWINS.get(detector.sql)
            if twin is None:
                out.append(
                    Verdict(
                        story.story,
                        detector.sql,
                        detector.metric,
                        None,
                        detector.direction,
                        detector.threshold,
                        None,
                        "no pandas twin yet",
                    )
                )
                continue
            try:
                value = twin(build, stories)
                note = NOTES[detector.sql](build, stories) if detector.sql in NOTES else ""
            except Missing as missing:
                out.append(
                    Verdict(
                        story.story,
                        detector.sql,
                        detector.metric,
                        None,
                        detector.direction,
                        detector.threshold,
                        None,
                        str(missing),
                    )
                )
                continue
            out.append(judge(story.story, detector, value, null, note))
        if story.story == "S4":
            note = "04's bar, no detector's threshold yet"
            try:
                value = s4_kmeans(build, stories)
                out.append(Verdict("S4", "kmeans", "adjusted_rand_index", value, ">=", KMEANS_BAR, None, note))
            except Missing as missing:
                out.append(Verdict("S4", "kmeans", "adjusted_rand_index", None, ">=", KMEANS_BAR, None, str(missing)))
    return out + probes(build, stories, null)


def probes(build: Build, stories: StoriesConfig, null: bool = False) -> list[Verdict]:
    """Every enabled anomaly's probe with a twin (module docstring), in `anomalies.yaml`'s order."""
    out = []
    for anomaly in stories.anomalies:
        direction, number = expected(anomaly.probe.expect)
        threshold: float | tuple[float, float] = (number, number) if direction == "between" else number
        shown = anomaly.probe.sql
        twin = PROBES.get(shown)
        if not anomaly.enabled or twin is None:
            why = "disabled" if not anomaly.enabled else "no pandas twin yet"
            out.append(Verdict(anomaly.id, shown, "count", None, direction, threshold, None, why))
            continue
        try:
            value = twin(build, stories)
        except Missing as missing:
            out.append(Verdict(anomaly.id, shown, "count", None, direction, threshold, None, str(missing)))
            continue
        found = holds(value, direction, threshold)
        out.append(Verdict(anomaly.id, shown, "count", value, direction, threshold, not found if null else found))
    return out


def report(verdicts: list[Verdict], run: Path, null: bool) -> str:
    """`stories.md`: one row per check."""
    head = f"# stories-check --csv {run.name}{' --null' if null else ''}\n\n"
    rows = ["| story | check | metric | threshold | value | result |", "|---|---|---|---|---|---|"]
    for v in verdicts:
        value = "—" if v.value is None else f"{v.value:.3f}"
        result = (
            ("not run" if v.value is None else "reported") if v.passed is None else ("pass" if v.passed else "FAIL")
        )
        note = f" ({v.note})" if v.note else ""
        rows.append(
            f"| {v.story} | `{v.check}` | {v.metric} | {v.direction} {v.threshold} | {value} | {result}{note} |"
        )
    ran = [v for v in verdicts if v.passed is not None]
    tail = (
        f"\n\n{sum(bool(v.passed) for v in ran)} of {len(ran)} run checks pass; {len(verdicts) - len(ran)} not run.\n"
    )
    return head + "\n".join(rows) + tail
