"""NH-D26/v1's monthly support and sensitivity over design09's audited placement rounds."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date, timedelta

import pandas as pd

from hartland_sim.checks import s5_decision
from hartland_sim.evaluation import Evaluation

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
