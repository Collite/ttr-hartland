"""NH-D28 held-out certification of forecast-based placement quality."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hartland_sim.checks import s5_decision, s5_monthly
from hartland_sim.checks import s5_development as dev
from hartland_sim.config import Detector, StoriesConfig, WorldConfig
from hartland_sim.evaluation import Evaluation
from hartland_sim.protocol import Protocol
from hartland_sim.world import Meta

WorldKey = tuple[int, str]
GROUP_SQL = "detectors/s5_best_worst.sql"
GAP_METRIC = "placement_quality_group_gap"


def manifest(held: Evaluation, protocol: Protocol) -> tuple[WorldKey, ...]:
    """Validate the frozen protocol before admitting just its thirty held-out identities."""
    if held.revision != s5_monthly.GROUP_REVISION:
        raise dev.Refused("certification requires NH-D28/v1")
    dev.manifest(held, protocol)
    return tuple((seed, mode) for seed in held.certification for mode in s5_decision.SCORED)


def check_detector(detector: Detector, held: Evaluation) -> None:
    """Cross-check the canonical answer key against the frozen evaluator."""
    if (detector.sql, detector.metric, detector.direction, detector.threshold, detector.null_world) != (
        GROUP_SQL,
        GAP_METRIC,
        ">=",
        held.acceptance["difference"],
        "<",
    ):
        raise dev.Refused("S5 requires the NH-D28 group gap and its 0.10 difference threshold")


@dataclass(frozen=True)
class Assessment:
    judged: s5_decision.Judged
    cases: tuple[s5_monthly.Case, ...]

    @property
    def failures(self) -> tuple[str, ...]:
        expected = {"story": "PASS", "skill-neutral": "REJECTED", "full-null": "REJECTED"}
        reasons = list(self.judged.reasons)
        if self.judged.status != expected.get(self.judged.mode or ""):
            reasons.append(f"full window: {self.judged.status}")
        if len(self.cases) != 301:
            reasons.append(f"{len(self.cases)} of 301 required date sets")
        for case in self.cases:
            if case.failure:
                label = " and ".join(day.isoformat() for day in case.omitted) or "full window"
                conditions = (case.failure,) if isinstance(case.failure, str) else case.failure
                reasons.append(f"{label}: {', '.join(conditions)}")
        return tuple(reasons)


def assess(
    run: Path,
    key: WorldKey,
    head: str,
    held: Evaluation,
    protocol: Protocol,
    world: WorldConfig,
    stories: StoriesConfig,
) -> Assessment:
    """Authenticate the source first, then use the same audited group judge as the ordinary acceptance path."""
    seed, mode = key
    wanted = dev.expected(seed, mode, world, stories, protocol.window_until)
    reasons = dev.source(run, seed, mode, head, wanted)
    if not reasons and Meta.load(run).last_day_done != protocol.window_until:
        reasons.append(f"build must end on {protocol.window_until}")
    if reasons:
        return Assessment(s5_decision.Judged(run, mode, "NOT_CERTIFIED", tuple(reasons), None, None, None, {}), ())
    base = s5_decision.judge(run, protocol, source_evaluation=held)
    return Assessment(*s5_monthly.evaluate(base, held))


def suite_status(results: Sequence[tuple[WorldKey, Sequence[str]]], held: Evaluation, protocol: Protocol) -> str:
    """Only a clean result for every registered seed and mode can pass the finite suite."""
    if tuple(key for key, _ in results) != manifest(held, protocol) or any(failures for _, failures in results):
        return "NOT_CERTIFIED"
    return "CERTIFIED"
