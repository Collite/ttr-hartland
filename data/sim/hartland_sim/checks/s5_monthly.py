"""NH-D26/v1's monthly support and sensitivity over design09's audited placement rounds."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, timedelta
from fractions import Fraction
from itertools import combinations
from typing import Any, cast

import pandas as pd

from hartland_sim.checks import s5_decision
from hartland_sim.config import DATA_DIR
from hartland_sim.evaluation import Evaluation

MONTHLY_FILE = DATA_DIR / "s5_evaluation_monthly.yaml"
BASE_REVISION = "NH-D25/v1"
MONTHLY_REVISION = "NH-D26/v1"
OMISSIONS = (1, 2)
ONE_DAY = timedelta(days=1)


def months(held: Evaluation) -> tuple[date, ...]:
    return tuple(day.date() for day in pd.date_range(held.window_from, held.window_until, freq="MS"))


def effective_month(day: date) -> date:
    return (day + ONE_DAY).replace(day=1)


def support(rounds: s5_decision.Frame, held: Evaluation) -> dict[str, bool]:
    """Check the complete calendar and both viable arms independently for each expected region."""
    wanted = Counter(months(held))
    out = {}
    for region in s5_decision.SKILL:
        rows = rounds[rounds.index.get_level_values("region") == region]
        dates = Counter(effective_month(day.date()) for day in rows.index.get_level_values("decision_day"))
        out[region] = (
            dates == wanted
            and bool((rows["chosen"] >= held.support["chosen_per_month"]).all())
            and bool((rows["unchosen"] >= held.support["unchosen_per_month"]).all())
            and int(rows["unchosen"].sum()) >= held.support["unchosen"]
        )
    return out


def judge(base: s5_decision.Judged, held: Evaluation) -> s5_decision.Judged:
    """Reuse the source and economic audit, applying this revision's support to its ordinary rounds."""
    if base.frame is None or base.rounds is None:
        return base
    frame = base.frame.copy()
    covered = support(base.rounds, held)
    frame["integrity"] = [
        ("valid" if covered.get(str(region), False) else "support") if state in {"valid", "support"} else state
        for region, state in frame["integrity"].items()
    ]
    verdict = s5_decision.verdict(frame, held)
    reasons = tuple(f"{region}: {state}" for region, state in frame["integrity"].items() if state != "valid")
    return replace(base, frame=frame, verdict=verdict, status=verdict.status, reasons=reasons)


@dataclass(frozen=True)
class Case:
    omitted: tuple[date, ...]
    regions: dict[str, float]
    verdict: s5_decision.Verdict
    failure: str | None


def _metric_case(q: dict[str, Fraction], omitted: tuple[date, ...], mode: str | None, held: Evaluation) -> Case:
    # The ranks are integers or half-integers. Compare the squared correlation exactly for the positive bar.
    ordered = {value: rank for rank, value in enumerate(sorted(set(q.values())))}
    quality_ranks = pd.Series([ordered[value] for value in q.values()]).rank(method="average")
    skill_ranks = pd.Series([s5_decision.SKILL[region] for region in q]).rank(method="average")
    centre = (len(q) + 1) / 2
    x = [int(2 * (v - centre)) for v in quality_ranks]
    y = [int(2 * (v - centre)) for v in skill_ranks]
    covariance = sum(a * b for a, b in zip(x, y, strict=True))
    variance = sum(a * a for a in x) * sum(b * b for b in y)
    rank = covariance / math.sqrt(variance) if variance else 0.0

    high = sum((q[r] for r in s5_decision.HIGH), Fraction()) / len(s5_decision.HIGH)
    low = sum((q[r] for r in s5_decision.LOW), Fraction()) / len(s5_decision.LOW)
    difference = high - low
    bars = {name: Fraction(str(held.acceptance[name])) for name in ("rank", "q_high", "difference")}
    rank_holds = variance > 0 and covariance >= 0 and covariance**2 >= bars["rank"] ** 2 * variance
    high_holds = high >= bars["q_high"]
    difference_holds = difference >= bars["difference"]
    story = rank_holds and high_holds and difference_holds
    control = not rank_holds and not difference_holds
    verdict = s5_decision.Verdict(
        "PASS" if story else "REJECTED",
        rank,
        float(high),
        float(low),
        float(difference),
        rank_holds,
        high_holds,
        difference_holds,
        control,
    )
    conditions = (
        (("rank", rank_holds), ("q_high", high_holds), ("difference", difference_holds))
        if mode == "story"
        else (("rank", not rank_holds), ("difference", not difference_holds))
    )
    failure = next((name for name, holds in conditions if not holds), None)
    return Case(omitted, {region: float(value) for region, value in q.items()}, verdict, failure)


def cases(judged: s5_decision.Judged, held: Evaluation) -> tuple[Case, ...]:
    """Full window first, then each common calendar omission, once the full window has support."""
    if judged.frame is None or judged.rounds is None or not s5_decision._valid(judged.frame):
        return ()
    monthly: dict[str, dict[date, Fraction]] = {region: {} for region in s5_decision.SKILL}
    fields = ["region", "decision_day", "chosen", "unchosen", "q"]
    for row in judged.rounds.reset_index()[fields].itertuples(index=False, name=None):
        region, day, chosen, unchosen, q = cast(tuple[str, pd.Timestamp, int, int, float], row)
        pairs_twice = 2 * chosen * unchosen
        # Round q comes from a pair count with half credit for ties. Recover that integer before averaging.
        win_credit = round(q * pairs_twice)
        monthly[region][effective_month(day.date())] = Fraction(win_credit, pairs_twice)
    dates = months(held)
    totals = {region: sum(values.values(), Fraction()) for region, values in monthly.items()}
    return tuple(
        _metric_case(
            {
                region: (totals[region] - sum((values[month] for month in omitted), Fraction()))
                / (len(dates) - len(omitted))
                for region, values in monthly.items()
            },
            omitted,
            judged.mode,
            held,
        )
        for count in (0, *OMISSIONS)
        for omitted in combinations(dates, count)
    )


def assessment(found: tuple[Case, ...]) -> dict[str, Any]:
    """Keep the original full-window judgement even when a subset would cross back over its bar."""
    if not found:
        return {"failure_type": "inconclusive", "full_failure": None, "count": 0, "failures": []}
    full, *omitted = found
    failures = [{"omitted": case.omitted, "condition": case.failure} for case in omitted if case.failure]
    categories = ("rank", "q_high", "difference")
    values = {name: [getattr(case.verdict, name) for case in omitted] for name in categories}
    values |= {region: [case.regions[region] for case in omitted] for region in full.regions}
    failure_type = "full_window" if full.failure else "sensitivity" if failures else None
    return {
        "failure_type": failure_type,
        "full_failure": full.failure,
        "count": len(omitted),
        "failures": failures,
        "ranges": {name: (min(scores), max(scores)) for name, scores in values.items()},
    }
