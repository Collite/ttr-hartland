"""G2 S2.3 T6 (review-199 R199-B/C): S5 on paper (`checks.s5_expected`), from the mechanics over the opening market —
the derivation `s5_best_worst`'s threshold comes from, before any build — and the numbers the YAML cites."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hartland_sim.checks.s5_expected import Region, chances, expected, rho, spread
from hartland_sim.config import StoriesConfig
from hartland_sim.loop.pos import Terms
from hartland_sim.setup.run import Setup

STORIES = StoriesConfig.load()
CZK_A_CALL = 45  # world.yaml field.pos_budget: a region's budget a month, to the thousand


@pytest.fixture(scope="module")
def paper(opening: Setup) -> tuple[dict[str, Region], dict[str, Region]]:
    """The story's and the null world's regions on paper."""
    world = opening.world
    return expected(world, opening.config, STORIES), expected(world, opening.config, STORIES.null())


def test_the_spread_s_threshold_lies_between_the_null_world_and_the_story(
    paper: tuple[dict[str, Region], dict[str, Region]],
) -> None:
    """Strictly between, at their geometric mean to the 0.05 below — and the values s5.yaml cites are the paper's."""
    story, null = paper
    (check,) = STORIES.story("S5").also
    assert isinstance(check.threshold, float)
    assert spread(null) < check.threshold < spread(story)
    assert check.threshold == math.floor(math.sqrt(spread(null) * spread(story)) * 20) / 20
    assert (round(spread(story), 2), round(spread(null), 2)) == (1.54, 1.21)
    assert (round(story["PHA"].per_cost, 2), round(story["VCH"].per_cost, 2)) == (7.10, 4.61)
    assert (round(null["PHA"].per_cost, 2), round(null["VCH"].per_cost, 2)) == (3.82, 3.15)


def test_the_story_ranks_the_regions_by_skill_and_the_null_world_does_not(
    paper: tuple[dict[str, Region], dict[str, Region]],
) -> None:
    """s5.sql's Spearman on paper: at its threshold or above in the story, below it in the null world, where the
    regions differ by their markets only."""
    story, null = paper
    threshold = STORIES.story("S5").detector.threshold
    assert isinstance(threshold, float)
    assert rho(story, STORIES) >= threshold > rho(null, STORIES)
    skilled = {r for r, skill in STORIES.story("S5").parameters["pos_skill"].items() if skill > 1}
    assert min(story[r].pos_return for r in skilled) > max(v.pos_return for r, v in story.items() if r not in skilled)


def test_a_region_s_budget_is_its_planned_calls_at_the_opening(opening: Setup) -> None:
    world = opening.world
    assert world.demand is not None
    terms = Terms.of(opening.config, STORIES, world.demand.lines)
    calls: dict[str, float] = {}
    for tourplan_id in world.tourplan_of.values():
        held = world.tourplans[tourplan_id]
        region = terms.region_of[held.area]
        calls[region] = calls.get(region, 0.0) + 52 / 12 / held.cycle_weeks
    budget = opening.config.field.pos_budget
    assert {r: round(CZK_A_CALL * calls[r], -3) for r in terms.regions} == {r: budget[r] for r in terms.regions}


def test_the_chances_hold_the_displays_out_and_follow_the_weights() -> None:
    weights = np.array([1.0, 2.0, 4.0, 8.0, 0.5])
    found = chances(weights, 2.0)
    assert found.sum() == pytest.approx(2.0, rel=1e-9)
    assert (np.diff(found[np.argsort(weights)]) > 0).all() and (found < 1).all()
    assert chances(weights, 10.0).sum() == pytest.approx(len(weights), abs=1e-5)  # more displays than stores
