"""S6 on paper (G1 S1.5 T2, `checks.s6_expected`): a granter's expected incremental Customer Margin per agreed-discount
crown, and `s6.yaml`'s `elasticity.high` derived from it (Bora's S6 rulings, 11-04; the lifted days after the end at
list, review-225 R1) — the smallest of 3.0 … 4.5 at which the pinned disciplined reps pay (≥ 1.15 a crown at the
planner's 0.88 margin share for G3/G4's cost to serve, ≥ 1.4 at 1.0), the pinned undisciplined do not (≤ 0.3 at 1.0),
and the curve rises over the discipline range."""

from __future__ import annotations

from typing import Any

import pytest

from hartland_sim.checks.s6_expected import AFTER_START, LIFTED, TAIL, expected_ratio, small_retail_caps
from hartland_sim.checks.stories import S6_AFTER
from hartland_sim.config import StoriesConfig, WorldConfig

S6 = StoriesConfig.load().story("S6").parameters
WORLD = WorldConfig.load()
CAPS = small_retail_caps(WORLD)
GRID = (3.0, 3.5, 4.0, 4.5)


def _at(high: float) -> dict[str, Any]:
    return {**S6, "elasticity": {**S6["elasticity"], "high": high}}


def test_one_cell_is_the_lifted_cases_margin_less_the_crowns_a_crown() -> None:
    """One line, every customer-line elastic: crowns ∝ d × ((L − T) × (1 + e × d) + U), gain ∝ L × e × d × m −
    crowns."""
    s6 = {**_at(3.5), "elasticity": {**S6["elasticity"], "high": 3.5, "high_share": 1.0}}
    d, e, m = 0.7 * 10 / 100, 3.5, 0.9  # 1.4 sizes at the mean share of the cap, 0.7
    crowns = d * ((LIFTED - TAIL) * (1 + e * d) + AFTER_START)
    assert (LIFTED, AFTER_START, TAIL) == (45, 7, 7)
    assert expected_ratio(1.4, m, s6, {"X": 10}) == pytest.approx((LIFTED * e * d * m - crowns) / crowns)
    flat = min(sum(S6["undisciplined_discount_pct"]) / 2, 4) / 100  # below `disciplined_from`: flat, to the cap
    crowns = flat * ((LIFTED - TAIL) * (1 + e * flat) + AFTER_START)
    assert expected_ratio(0.5, m, s6, {"X": 4}) == pytest.approx((LIFTED * e * flat * m - crowns) / crowns)


def test_the_paper_s_numbers() -> None:
    """review-225 R1's: the planner's T2 (1.21 · 1.51 · 0.24 at 3.5) bought every lifted day at the discount."""
    assert expected_ratio(1.4, 0.88, _at(3.0), CAPS) == pytest.approx(1.28, abs=0.005)
    assert expected_ratio(1.4, 1.0, _at(3.0), CAPS) == pytest.approx(1.59, abs=0.005)
    assert expected_ratio(0.55, 1.0, _at(3.0), CAPS) == pytest.approx(0.27, abs=0.005)
    assert expected_ratio(0.55, 1.0, _at(3.5), CAPS) == pytest.approx(0.44, abs=0.005)  # 3.5 overshoots


def test_the_paper_s_days_and_lines_are_small_retail_s() -> None:
    """The window runs one order past the end (`s6.sql`), and the first order inside and the first after each carry
    half the days between two; a small-retail customer is never granted a line it is not listed."""
    assert S6_AFTER == WORLD.channels.small_retail.order_cycle_days == 2 * AFTER_START == 2 * TAIL
    assert set(WORLD.pricing.max_agreed_discount_pct) - set(CAPS) == {"PROFIKAVA", "PROFIGASTRO"}


def _meets(s6: dict[str, Any]) -> bool:
    pinned, cut = s6["pinned_reps"].values(), s6["disciplined_from"]
    disciplined, undisciplined = [d for d in pinned if d >= cut], [d for d in pinned if d < cut]
    low, top = s6["discipline_range"]
    curve = [expected_ratio(low + k * (top - low) / 18, 1.0, s6, CAPS) for k in range(19)]
    return (
        all(expected_ratio(d, 0.88, s6, CAPS) >= 1.15 and expected_ratio(d, 1.0, s6, CAPS) >= 1.4 for d in disciplined)
        and all(expected_ratio(d, 1.0, s6, CAPS) <= 0.3 for d in undisciplined)
        and all(earlier < later for earlier, later in zip(curve, curve[1:], strict=False))
    )


def test_elasticity_high_is_the_smallest_of_the_grid_that_meets_04_s_bounds() -> None:
    met = [high for high in GRID if _meets(_at(high))]
    assert met[0] == 3.0 and S6["elasticity"]["high"] == met[0]
    assert sorted(S6["pinned_reps"].values()) == [0.5, 0.55, 1.35, 1.4]
