"""G2 S2.4 T3's one pooled candidate (NH-D19; ttr-hartland#17 §3–§5, frozen before any of its outputs on a treated
loan was read) — a prototype beside `s5_loans`, never a detector: neither `stories.py`, the SQL nor the stories check
reads it.

- **The placebo loans** (`placebos`): for each loan in scope, up to `DRAWS` distinct stores of its cell (`S5_CELL`)
  other than its own, with no display out in the 56 days before or during it, drawn at random (`SEED`); each is "lent"
  on the same day for the same days and scored by `s5_score` against the real loans' displays only — a placebo is no
  one's display, so no real loan's controls change. Draw 0 estimates the variance; draws 1 on evaluate.
- **The variance** (`Variance.of`): v = σ²_c · X^κ — κ the slope of log g² on log X over draw 0's supported placebos
  whose store bought before (X > 0) and whose gain is not 0; σ²_c the mean of g² ÷ X^κ over those of the loan's exact
  cell when they are `MIN_PLACEBOS` or more, else over its region's. Fewer there, or a σ² of 0: no candidate in the
  region.
- **The regions** (`regions`): the baseline, Σ g of the known loans (s5.sql's region gain), and the candidate,
  θ̂ · Σ X over the known loans whose store bought before plus Σ g of the known new stores (s5.sql's convention,
  reported apart), θ̂ = Σ w X g ÷ Σ w X², w = 1 ÷ v and X = before · q · days, the margin the store would have made
  during the loan at its controls' pace. A known loan whose store's margin before is negative is in neither: counted
  and reported (`other`). Costs are the loans'; the unknown loans stay unknown (`known_cost`).
- **The resampling weights** (`resampling`): A_store · B_month, each Exp(1): one weight a store, its loans share it,
  and one a calendar month of `lent_on`, every region shares it — a calendar shock moves the regions together.
- **The known lifts** (`lifts`): H0 0.10 · X everywhere; H1 ∝ the store's display uplift (`pos.Grid`, the truth's
  shape only, never a weight), 0.10 on average in each region; H2 0.20 · X where the loan's σ² is under its region's
  median, else nothing.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hartland_sim.checks.stories import (
    S5_BEST,
    S5_CELL,
    S5_FROM,
    S5_WORST,
    WEEK_DAYS,
    Build,
    s5_held,
    s5_outlets,
    s5_score,
)
from hartland_sim.loop.pos import Grid

Frame = pd.DataFrame
Array = np.ndarray[Any, Any]

SEED = 20261113  # every draw of T3's protocol
DRAWS = 5  # placebos a loan at most: draw 0 estimates, 1–4 evaluate
MIN_PLACEBOS = 20  # a cell's σ² needs this many, else its region's
RESAMPLES = 200
LIFT = 0.10  # H0's and H1's mean lift on X
ADVERSE = 0.20  # H2's, in the low-variance half


def placebos(build: Build, draws: int = DRAWS, seed: int = SEED) -> Frame:
    """The placebo loans (module docstring), scored: `s5_score`'s columns, then each one's `draw`, the real loan it
    mirrors (`mirrors`) and its cell (`S5_CELL`); ids from 1, none a real loan's."""
    every = s5_held(build)
    scope = every[every["lent_on"] >= pd.Timestamp(S5_FROM)]
    outlets = s5_outlets(build)
    loans = scope[["pos_loan_id", "region", "customer_id", "store_id", "lent_on", "until", "cost"]]
    treated = loans.merge(outlets, on=["customer_id", "store_id"]).drop(columns="customer_id")
    pairs = treated.merge(outlets, on=S5_CELL, suffixes=("_treated", ""))
    pairs = pairs[pairs["store_id"] != pairs["store_id_treated"]].reset_index(drop=True)
    held = every.loc[every["store_id"].notna(), ["store_id", "lent_on", "until"]]
    spans = pairs[["store_id", "lent_on", "until"]].reset_index().merge(held, on="store_id", suffixes=("", "_held"))
    week = pd.Timedelta(days=WEEK_DAYS)
    busy = spans[(spans["lent_on_held"] <= spans["until"]) & (spans["until_held"] >= spans["lent_on"] - week)]
    free = pairs[~pairs.index.isin(busy["index"])].sort_values(["pos_loan_id", "store_id", "customer_id"])
    free = free.drop_duplicates(["pos_loan_id", "store_id"]).reset_index(drop=True)
    free["key"] = np.random.default_rng(seed).random(len(free))
    free = free.sort_values(["pos_loan_id", "key"])
    free["draw"] = free.groupby("pos_loan_id").cumcount()
    free = free[free["draw"] < draws].rename(columns={"pos_loan_id": "mirrors"}).reset_index(drop=True)
    free["pos_loan_id"] = pd.array(np.arange(1, len(free) + 1), dtype="Int64")
    real = set(build["pos_loan"]["pos_loan_id"].astype("int64"))
    if real & set(range(1, len(free) + 1)):
        raise ValueError("a placebo's id is a real loan's")
    columns = ["pos_loan_id", "region", "customer_id", "store_id", "lent_on", "until", "cost"]
    scored = s5_score(build, free[columns], held)
    return scored.join(free.set_index("pos_loan_id")[["draw", "mirrors", *S5_CELL]])


def cells(build: Build, loans: Frame) -> Frame:
    """`loans` (`s5_loans`' or `s5_score`'s) with their cell (`S5_CELL`; NaN for a loan at no store)."""
    outlets = s5_outlets(build).drop_duplicates(["customer_id", "store_id"])
    found = loans[["customer_id", "store_id"]].merge(outlets, on=["customer_id", "store_id"], how="left")
    return loans.assign(**{c: found[c].to_numpy() for c in S5_CELL})


def counterfactual(loans: Frame) -> pd.Series[float]:
    """X = before · q · days of each known loan (NaN where its gain is not known)."""
    q = loans["control_during"] / loans["control_before"].where(loans["gain"].notna())
    found: pd.Series[float] = loans["before"] * q * loans["days"]
    return found


def _fitted(loans: Frame) -> Array:
    return np.asarray(loans["gain"].notna() & (loans["before"] > 0))


@dataclass(frozen=True)
class Variance:
    """v = σ²_c · X^κ (module docstring), from draw 0's placebos."""

    kappa: float
    cells: Mapping[tuple[Hashable, ...], float]
    regions: Mapping[str, float]
    fitted: int  # the placebos κ is fitted on
    zero: int  # those left out of κ's fit, their gain 0

    @classmethod
    def of(cls, estimation: Frame) -> Variance:
        p = estimation[_fitted(estimation)]
        x, g = counterfactual(p).to_numpy(), p["gain"].to_numpy()
        fit = g != 0
        kappa = float(np.polyfit(np.log(x[fit]), np.log(g[fit] ** 2), 1)[0])
        z = p.assign(z=g**2 / x**kappa)
        by_cell = {k: float(v["z"].mean()) for k, v in z.groupby(S5_CELL) if len(v) >= MIN_PLACEBOS}
        by_region = {str(k): float(v["z"].mean()) for k, v in z.groupby("region") if len(v) >= MIN_PLACEBOS}
        return cls(kappa, by_cell, by_region, int(fit.sum()), int((~fit).sum()))

    def scale(self, loans: Frame) -> Array:
        """σ² of each loan's cell, else of its region; NaN where neither has `MIN_PLACEBOS` or it is 0."""
        keys = loans[S5_CELL].itertuples(index=False, name=None)
        found = [self.cells.get(k, self.regions.get(r, math.nan)) for k, r in zip(keys, loans["region"], strict=True)]
        scale = np.asarray(found, dtype=float)
        return np.where(scale > 0, scale, math.nan)


def regions(loans: Frame, variance: Variance, weights: Array | None = None) -> Frame:
    """Per region (module docstring): `baseline`, `candidate`, `theta`, `x` (Σ X fitted), `new_store` (Σ g of the known
    new stores), `other` (Σ g of the known loans in neither), `loan_cost`, `known_cost` and the counts — every sum
    under `weights` (a loan's; 1 without). A region with a fitted loan of no variance has no candidate (NaN)."""
    w = np.ones(len(loans)) if weights is None else np.asarray(weights, dtype=float)
    known = loans["gain"].notna().to_numpy()
    fitted = _fitted(loans)
    new = known & loans["new_store"].to_numpy(dtype=bool)
    x = counterfactual(loans).to_numpy()
    v = variance.scale(loans) * np.where(fitted, x, 1.0) ** variance.kappa
    ok = fitted & np.isfinite(v)
    g = loans["gain"].fillna(0.0).to_numpy()
    cost = loans["cost"].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "region": loans["region"].to_numpy(),
            "baseline": w * g,
            "xg": np.where(ok, w * x * g / np.where(ok, v, 1.0), 0.0),
            "xx": np.where(ok, w * x * x / np.where(ok, v, 1.0), 0.0),
            "x": np.where(fitted, w * x, 0.0),
            "new_store": np.where(new, w * g, 0.0),
            "other": np.where(known & ~fitted & ~new, w * g, 0.0),
            "loan_cost": w * cost,
            "known_cost": np.where(known, w * cost, 0.0),
            "loans": 1,
            "fitted": fitted.astype(int),
            "others": (known & ~fitted & ~new).astype(int),
            "unestimable": (fitted & ~ok).astype(int),
        }
    )
    out = frame.groupby("region").sum()
    out["theta"] = (out["xg"] / out["xx"].where(out["xx"] > 0)).where(out["unestimable"] == 0)
    out["candidate"] = out["theta"] * out["x"] + out["new_store"]
    return out.drop(columns=["xg", "xx"])


def resampling(loans: Frame, n: int = RESAMPLES, seed: int = SEED) -> Array:
    """`n` rows of a weight a loan, A_store · B_month (module docstring); a loan at no store is a store of its own."""
    rng = np.random.default_rng(seed)
    stores, _ = pd.factorize(loans["store_id"])
    lone = stores < 0
    stores[lone] = stores.max() + 1 + np.arange(lone.sum())
    months, _ = pd.factorize(loans["lent_on"].dt.to_period("M"))
    a = rng.exponential(size=(n, int(stores.max()) + 1))
    b = rng.exponential(size=(n, int(months.max()) + 1))
    return np.asarray(a[:, stores] * b[:, months])


def lifts(loans: Frame, variance: Variance, grid: Grid, design: str) -> Array:
    """Each loan's known lift, CZK, under `design` (H0, H1, H2; module docstring); 0 for a loan not fitted."""
    fitted = _fitted(loans)
    x = np.where(fitted, counterfactual(loans).to_numpy(), 0.0)
    regions_ = loans["region"].to_numpy()
    if design == "H0":
        return LIFT * x
    if design == "H1":
        stores = zip(loans["street_type"], loans["size_class"], strict=True)
        u = np.array([grid.base * grid.street_fit[s] * grid.size_fit[z] for s, z in stores])
        mean = pd.Series(u[fitted]).groupby(regions_[fitted]).mean()
        return np.asarray(LIFT * u / pd.Series(regions_).map(mean).to_numpy() * x)
    if design == "H2":
        scale = variance.scale(loans)
        median = pd.Series(scale[fitted]).groupby(regions_[fitted]).median()
        low = scale < pd.Series(regions_).map(median).to_numpy()
        return np.where(low, ADVERSE * x, 0.0)
    raise ValueError(f"no design {design}")


def spread(gain: Mapping[str, float], cost: Mapping[str, float]) -> float:
    """s5_best_worst.sql's PHA ÷ VCH gain ÷ cost (NaN where VCH's is ≤ 0 or a cost is not positive)."""
    if not (cost.get(S5_BEST, 0) > 0 and cost.get(S5_WORST, 0) > 0):
        return math.nan
    worst = gain[S5_WORST] / cost[S5_WORST]
    return gain[S5_BEST] / cost[S5_BEST] / worst if worst > 0 else math.nan


def rho(gain: Mapping[str, float], cost: Mapping[str, float], skill: Mapping[str, float]) -> float:
    """s5.sql's Spearman of (gain − cost) ÷ cost against `skill`, on average ranks."""
    regions_ = [r for r in skill if cost.get(r, 0) > 0]
    returns = pd.Series([(gain[r] - cost[r]) / cost[r] for r in regions_]).rank(method="average")
    skills = pd.Series([float(skill[r]) for r in regions_]).rank(method="average")
    if len(regions_) < 2 or not returns.std() or not skills.std():
        return math.nan
    return float(np.corrcoef(returns, skills)[0, 1])
