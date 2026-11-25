"""The frozen NH-D37 rate direction, exact ties, threshold edges and paired identity controls."""

from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

from hartland_sim.checks import s3, s3_skills
from hartland_sim.checks.stories import Build
from hartland_sim.config import StoriesConfig
from hartland_sim.world import Meta


def test_signed_rates_use_equal_fractions_and_mark_zero_exposure_undefined() -> None:
    counts = pd.DataFrame({
        "converting_visits": [1, 2, 0], "sales_visits": [3, 6, 0],
        "wins": [0, 1, 0], "potential_days": [5, 5, 0],
        "losses": [1, 2, 0], "active_days": [10, 10, 0], "first_paid_lines": [0, 3, 0],
    })
    rates = s3.exact_rates(counts)
    assert rates["conversion"].tolist() == [Fraction(1, 3), Fraction(1, 3), None]
    assert rates["retention"].tolist() == [Fraction(-1, 10), Fraction(-1, 5), None]
    rank = rates.rank(ascending=False, method="min")
    assert rank["conversion"].iloc[:2].tolist() == [1, 1]
    assert rank["retention"].iloc[:2].tolist() == [1, 2]
    assert rank.iloc[2].isna().all()


def test_correlations_align_labels_by_employee_and_use_average_ranks() -> None:
    labels = {str(i): {measure: float(i) for measure in s3.MEASURES} for i in range(50)}
    keys = list(reversed(labels))
    rates = pd.DataFrame({measure: [int(key) // 2 for key in keys] for measure in s3.MEASURES}, index=keys)
    actual = s3_skills.signed_correlations(rates, labels, False)
    assert actual["conversion"][0] == pytest.approx(0.9993995796516864)
    assert actual["conversion"][1] is True
    assert s3_skills.signed_correlations(rates, labels, True)["conversion"][1] is False
    rates["conversion"] = 1
    assert s3_skills.signed_correlations(rates, labels, True)["conversion"] == [None, None]
    rates.loc["0", "retention"] = None
    assert s3_skills.signed_correlations(rates, labels, False)["retention"] == [None, None]


@pytest.mark.parametrize(("rho", "null", "expected"), [
    (0.70, False, True), (0.699999999, False, False), (0.30, True, False),
    (-0.30, True, False), (-0.299999999, True, True), (None, False, False),
])
def test_thresholds_apply_before_rounding(rho: float | None, null: bool, expected: bool) -> None:
    assert s3_skills.correlation_passes(rho, null) == expected


def test_pair_rejects_mismatched_null_ids_before_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story, null = tmp_path / "story", tmp_path / "null"
    Meta(42, "story", "world", s3_skills.LAST_DAY, "story", (("pilot_mode", "story"),)).save(story)
    Meta(42, "null", "world", s3_skills.LAST_DAY, "null", (("pilot_mode", "full-null"),)).save(null)

    def labels(path: Path) -> dict[str, dict[str, float]]:
        employees = range(50) if path == story else range(1, 51)
        return {str(employee): {measure: 1.0 for measure in s3.MEASURES} for employee in employees}

    monkeypatch.setattr(s3_skills, "read_labels", labels)
    with pytest.raises(ValueError, match="full-null employee mapping"):
        s3_skills.read_pair(story, null, "", tmp_path / "receipt.json")


def test_business_table_keeps_hidden_labels_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({
        "converting_visits": [3], "sales_visits": [4], "wins": [1], "potential_days": [100],
        "losses": [2], "active_days": [200], "first_paid_lines": [4],
    }, index=[11])
    monkeypatch.setattr(s3, "counts", lambda _: frame)
    monkeypatch.setattr(s3, "slots", lambda _: pd.DataFrame({"employee_id": [11], "code": ["PHA-03"]}))
    result = s3.business_report(Build(tmp_path), StoriesConfig.load())
    assert "| 11 | PHA-03 | 3 / 4 · 0.750000 · 1" in result
    assert "2 / 200 · -0.010000 · 1" in result
    assert "skill" not in result
