"""The stories on the build tier (`pytest -m build`, its own CI job): G1 S1.2's DONE line (T7) and S1.4's S6 (T6) on
the story and the null world, `build-world` from the warm-up to 2025-12-31 (`conftest.story_world`, `null_world`; S1's
window is 2024–2025, S4's 2025). The warm-up writes nothing before 2024-01-01 and 2024 starts with a year of history;
the S1, S2 and S4 twins pass on the story world and fail on the null one.

**S6** (S1.4 T6, re-cut in S1.5 T5): per rep, the Customer Margin gained per agreed-discount crown
(`checks.stories.gains`, s6.sql's measure) ranks with its hidden `discount_discipline` (the answer key: the story
world's checkpoint) — Spearman ≥ 0.8 over all reps, the null world's |ρ| < 0.3 — and `04` S6's bounds on the pinned
reps (review-196 ⚑D): the disciplined ones' ratio > 1, the undisciplined ones' < 0.5; s6.sql's twin passes on the
story world and fails on the null one. The story world falls short of all four bars: the lift — a ratio of the few cases
an agreement's windows hold — scatters far wider than the discount moves it, so strict expected failures hold the
numbers and the PR's Found line carries them (the list's stop condition: S1.5 does not solve it)."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hartland_sim.checks.stories import S6_FROM, Build, Verdict, check, gains, s6_reps
from hartland_sim.cli import main
from hartland_sim.config import StoriesConfig
from hartland_sim.loop.build import Built
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
    assert main(["stories-check", "--csv", str(null_world.run), "--null"]) == 0  # each gives the null world's side


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


@pytest.mark.xfail(strict=True, reason="Found (S1.5 T5): Spearman 0.19 on 2024–2025; per-rep residual SD 1.27 vs T2")
def test_s6_the_reps_gain_a_crown_ranks_with_their_discipline(story_world: Built) -> None:
    assert spearman(by_rep(story_world.run, _reps(story_world))) >= 0.8


def test_s6_in_the_null_world_the_ranking_is_gone(story_world: Built, null_world: Built) -> None:
    frame = by_rep(null_world.run, _reps(story_world))
    assert len(frame) == 50 and abs(spearman(frame)) < 0.3


@pytest.mark.xfail(strict=True, reason="Found (S1.5 T5): the pinned disciplined 0.64 a crown, not > 1 (T2: 1.51)")
def test_s6_the_pinned_disciplined_reps_discounts_pay(story_world: Built) -> None:
    assert s6_reps(Build(story_world.run), StoriesConfig.load())[True] > 1


@pytest.mark.xfail(
    strict=True, reason="Found (S1.5 T5): the pinned undisciplined 0.73 a crown, not < 0.5 (T2: 0.08–0.24)"
)
def test_s6_the_pinned_undisciplined_reps_lose_on_their_discounts(story_world: Built) -> None:
    assert s6_reps(Build(story_world.run), StoriesConfig.load())[False] < 0.5


@pytest.mark.xfail(strict=True, reason="Found (S1.5 T5): s6.sql's pinned difference −0.09 on 2024–2025, not ≥ 0.5")
def test_s6_s_twin_passes_on_the_story_world(story_world: Built) -> None:
    (s6,) = [v for v in check(story_world.run, StoriesConfig.load()) if v.check == "detectors/s6.sql"]
    assert s6.passed, s6


def test_s6_s_twin_fails_on_the_null_world(null_world: Built) -> None:
    (s6,) = [v for v in check(null_world.run, StoriesConfig.load()) if v.check == "detectors/s6.sql"]
    assert s6.value is not None and not np.isnan(s6.value) and s6.passed is False
