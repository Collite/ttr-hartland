"""G2 S2.4 T3 under NH-D20 (ttr-hartland#17 §10): the cohort-support diagnosis — beside `s5_score`, never a detector:
neither `stories.py`, the SQL nor the stories check reads it, and it draws nothing.

- **The cohort** (`pairs`): NH-D17's controls of each loan, `s5_score`'s — the stores of its cell (`S5_CELL`) but its
  own and any with a display of `held` out in the 56 days before or during it — one row per loan and control, with
  the control's Customer Margin before and during (sums over the windows) and whether it is `eligible`.
- **The rule** (`eligible`, frozen in §10.0 before any output on a treated loan): a control is a store whose customer
  is active on the loan's `lent_on` (`World.Customer.active_on`: never a potential, `since` ≤ day, `until` none or ≥
  day) — `loop.pos.place`'s candidate on the lending day, read from nothing after it.
- **The score** (`score`): NH-D17's gain of each loan from its cohort, over the controls kept — all of them give
  `s5_score`'s, loan for loan.
- **The strata** (`strata`): each loan's store before it was lent: `inactive` (its customer not active that day),
  `nothing` (active, no margin in the 56 days before), `bought` (active, a margin).
- **The oracle** (`oracle`): what the simulation's display added to a loan — the Customer Margin of its display's
  lines at its store during it × u ÷ (1 + u), u its store's `Grid.uplift`, as `loop.demand` multiplies the lines'
  demand by 1 + u while a loan runs. Simulation truth: a labelled validation oracle, never an estimator's input.
- **The paper over the loans** (`paper`): the gain the mechanism expects of the same loans over the same days — u ×
  the margin of the display's lines a day in the 56 days before × its days — from pre-lending rows and `s5.yaml`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from hartland_sim.checks.stories import (
    S5_CELL,
    S5_CONTROLS,
    WEEK_DAYS,
    Build,
    _sales_pl,
    _sums,
    s5_outlets,
)
from hartland_sim.loop.pos import Grid

Frame = pd.DataFrame
Array = np.ndarray[Any, Any]

STRATA = ("inactive", "nothing", "bought")
KEYS = ["customer_id", "store_id"]


def active_on(build: Build, customers: pd.Series[Any], days: pd.Series[Any]) -> Array:
    """Whether each of `customers` was active on its day of `days` (`Customer.active_on`)."""
    held = build["customer"].drop_duplicates("customer_id").set_index("customer_id")[["since", "until", "status"]]
    found = held.reindex(pd.Index(customers.astype("int64")))
    day = days.to_numpy()
    since, until = found["since"].to_numpy(), found["until"].to_numpy()
    with np.errstate(invalid="ignore"):
        started = ~pd.isna(since) & (since <= day)
        going = pd.isna(until) | (day <= until)
    return np.asarray((found["status"].to_numpy() != "potential") & started & going, dtype=bool)


def _margins(build: Build) -> Frame:
    """A store's Customer Margin a day, by customer (`s5_score`'s `stores`)."""
    sales = _sales_pl(build)
    daily = sales.groupby(["customer_id", "store_id", "ordered_on"], dropna=False)["customer_margin"].sum()
    found = daily.reset_index()
    return found[found["store_id"].notna()]


def pairs(build: Build, loans: Frame, held: Frame) -> Frame:
    """NH-D17's cohort of each of `loans` (`s5_score`'s input), a row per loan and control (module docstring): its
    `pos_loan_id`, the control's `customer_id` and `store_id`, its margin `before` and `during` (sums), `eligible`."""
    week, one = pd.Timedelta(days=WEEK_DAYS), pd.Timedelta(days=1)
    loans = loans.reset_index(drop=True)
    loans["before_from"], loans["before_to"] = loans["lent_on"] - week, loans["lent_on"] - one
    outlets = s5_outlets(build)
    treated = loans[["pos_loan_id", *KEYS, "lent_on", "until", "before_from", "before_to"]]
    found = treated.merge(outlets, on=KEYS).drop(columns="customer_id")
    found = found.merge(outlets, on=S5_CELL, suffixes=("_treated", ""))
    found = found[found["store_id"] != found["store_id_treated"]].reset_index(drop=True)
    held = held.loc[held["store_id"].notna(), ["store_id", "lent_on", "until"]]
    spans = found[["store_id", "before_from", "until"]].reset_index().merge(held, on="store_id", suffixes=("", "_held"))
    busy = spans[(spans["lent_on"] <= spans["until"]) & (spans["until_held"] >= spans["before_from"])]["index"]
    found = found[~found.index.isin(busy)].reset_index(drop=True)
    stores = _margins(build)
    for column, first, last in (("before", "before_from", "before_to"), ("during", "lent_on", "until")):
        found[column] = _sums(stores, KEYS, ["customer_margin"], found, first, last).iloc[:, 0]
    found["eligible"] = active_on(build, found["customer_id"], found["lent_on"])
    return found[["pos_loan_id", *KEYS, "lent_on", "until", "before", "during", "eligible"]]


def score(loans: Frame, cohort: Frame, keep: Array | None = None) -> Frame:
    """`loans` (`s5_score`'s) scored again over the controls of `cohort` (`pairs`) that `keep` keeps — every one when
    None, which is `s5_score`'s own: `controls`, `control_before`, `control_during` and `gain` anew."""
    kept = cohort if keep is None else cohort[keep]
    sums = kept.groupby("pos_loan_id").agg(
        controls=("before", "size"), control_before=("before", "sum"), control_during=("during", "sum")
    )
    out = loans.drop(columns=["controls", "control_before", "control_during", "gain"]).join(sums)
    out["controls"] = out["controls"].fillna(0).astype(int)
    out["control_before"] = out["control_before"].fillna(0.0) / WEEK_DAYS
    out["control_during"] = out["control_during"].fillna(0.0) / out["days"]
    known = (out["controls"] >= S5_CONTROLS) & (out["control_before"] > 0)
    q = out["control_during"] / out["control_before"].where(known)
    out["gain"] = (out["during"] - out["before"] * q) * out["days"]
    return out[loans.columns]


def eligible(cohort: Frame) -> Array:
    """The rule (module docstring): the controls of `cohort` whose customer is active on the loan's `lent_on`."""
    return np.asarray(cohort["eligible"], dtype=bool)


def strata(build: Build, loans: Frame) -> pd.Series[str]:
    """Each of `loans`' (`s5_score`'s) stratum (module docstring)."""
    active = active_on(build, loans["customer_id"], loans["lent_on"])
    bought = np.where(loans["before"].to_numpy() != 0, STRATA[2], STRATA[1])
    return pd.Series(np.where(active, bought, STRATA[0]).tolist(), index=loans.index, name="stratum", dtype=str)


def _lines(build: Build, loans: Frame, displays: Mapping[str, tuple[str, ...]], grid: Grid) -> Frame:
    """A row per loan of `loans` (`s5_score`'s) and line of its display, with its store's u."""
    loan = build["pos_loan"][["pos_loan_id", "pos_asset_id"]].merge(
        build["pos_asset"][["pos_asset_id", "pos_material_id"]], on="pos_asset_id"
    )
    loan = loan.merge(build["pos_material"][["pos_material_id", "code"]], on="pos_material_id")
    material = loan.set_index("pos_loan_id")["code"]
    line_id = build["product_line"].set_index("code")["product_line_id"]
    store = build["store"].set_index("store_id")[["street_type", "size_class"]]
    held = loans.reset_index()[["pos_loan_id", *KEYS, "lent_on", "days"]]
    held["until"] = held["lent_on"] + pd.to_timedelta(held["days"] - 1, unit="D")
    held["code"] = held["pos_loan_id"].map(material)
    held["product_line_id"] = held["code"].map(lambda code: [int(line_id[line]) for line in displays[code]])
    if grid.flat is not None:
        held["u"] = grid.flat
    else:
        kind = store.reindex(held["store_id"].to_numpy())
        street = kind["street_type"].map(grid.street_fit).to_numpy(dtype=float)
        size = kind["size_class"].map(grid.size_fit).to_numpy(dtype=float)
        held["u"] = grid.base * street * size
    return held.explode("product_line_id").astype({"product_line_id": "int64"}).reset_index(drop=True)


def _line_margins(build: Build) -> Frame:
    sales = _sales_pl(build)
    daily = sales.groupby([*KEYS, "product_line_id", "ordered_on"])["customer_margin"].sum()
    return daily.reset_index()


def oracle(build: Build, loans: Frame, displays: Mapping[str, tuple[str, ...]], grid: Grid) -> pd.Series[float]:
    """The simulation's own gain of each of `loans` (`s5_score`'s, at a store), the validation oracle (module
    docstring): Σ over its display's lines of their margin during it × u ÷ (1 + u)."""
    held = _lines(build, loans, displays, grid)
    keys = [*KEYS, "product_line_id"]
    margin = _sums(_line_margins(build), keys, ["customer_margin"], held, "lent_on", "until")
    held["oracle"] = margin.iloc[:, 0] * held["u"] / (1 + held["u"])
    found: pd.Series[float] = held.groupby("pos_loan_id")["oracle"].sum().reindex(loans.index)
    return found


def paper(build: Build, loans: Frame, displays: Mapping[str, tuple[str, ...]], grid: Grid) -> pd.Series[float]:
    """The mechanism's expected gain of each of `loans` (module docstring): u × the margin of its display's lines a
    day in the 56 days before it × its days — pre-lending rows and `s5.yaml` only."""
    held = _lines(build, loans, displays, grid)
    held["before_from"] = held["lent_on"] - pd.Timedelta(days=WEEK_DAYS)
    held["before_to"] = held["lent_on"] - pd.Timedelta(days=1)
    keys = [*KEYS, "product_line_id"]
    margin = _sums(_line_margins(build), keys, ["customer_margin"], held, "before_from", "before_to")
    held["paper"] = margin.iloc[:, 0] / WEEK_DAYS * held["u"] * held["days"]
    found: pd.Series[float] = held.groupby("pos_loan_id")["paper"].sum().reindex(loans.index)
    return found
