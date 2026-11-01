"""G0 S0.4 T3 — the answer key: `data/stories/` (contracts §3, design 04), loaded by `config.StoriesConfig`."""

from __future__ import annotations

import re
import shutil
from collections import Counter
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.cli import main
from hartland_sim.config import OPPOSITE, STORIES_DIR, ConfigError, InventoryConfig, StoriesConfig, WorldConfig
from hartland_sim.io.reference import read
from hartland_sim.setup.classifier.chains import BY_CODE

# contracts §2: the committed stream catalogue a story's draws name.
STREAMS = set(
    "demand basket acquire churn visit note observe sighting prompt_outcome pos pricing_agree payment return complaint "
    "supply route event shop weather letter scan".split()
)

WRITES = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy)\b", re.I)


@pytest.fixture(scope="module")
def stories() -> StoriesConfig:
    return StoriesConfig.load()


def _sql(path: str) -> str:
    return (STORIES_DIR / path).read_text(encoding="utf-8")


def _values(sql: str, cte: str) -> list[tuple[str, ...]]:
    """The rows of a `<cte> (…) AS (VALUES …)` list in `sql`, as tuples of their literals' text."""
    block = re.search(rf"{cte} \([^)]*\) AS \(\s*VALUES(.*?)\n\)", sql, re.S)
    assert block, cte
    return [tuple(part.strip(" '") for part in row.split(",")) for row in re.findall(r"\(([^()]*)\)", block.group(1))]


def test_six_stories_and_fifteen_cases(stories: StoriesConfig) -> None:
    assert [story.story for story in stories.stories] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert [anomaly.id for anomaly in stories.anomalies] == [f"A-{n}" for n in range(1, 16)]
    assert all(story.enabled for story in stories.stories)
    assert all(anomaly.enabled for anomaly in stories.anomalies)


def test_every_story_has_a_detector_and_a_null_world_direction(stories: StoriesConfig) -> None:
    for story in stories.stories:
        for detector in story.detectors:
            assert detector.null_world == OPPOSITE[detector.direction], detector.sql
    assert sum(len(story.detectors) for story in stories.stories) == 10


def test_every_case_has_a_probe(stories: StoriesConfig) -> None:
    for anomaly in stories.anomalies:
        assert anomaly.probe.sql == f"probes/a{anomaly.id[2:]}.sql"
        assert re.fullmatch(r">= [1-9][0-9]*|= [0-9]+|rows", anomaly.probe.expect)


def test_every_sql_file_is_one_read_only_select(stories: StoriesConfig) -> None:
    files = [detector.sql for story in stories.stories for detector in story.detectors]
    files += [anomaly.probe.sql for anomaly in stories.anomalies]
    on_disk = sorted(f"{path.parent.name}/{path.name}" for path in STORIES_DIR.glob("*/*.sql"))
    assert sorted(files) == on_disk
    for name in files:
        code = "\n".join(line for line in _sql(name).splitlines() if not line.startswith("--"))
        assert re.match(r"\s*(WITH|SELECT)\b", code), name
        assert ";" not in code, name
        assert not WRITES.search(code), name


def test_the_sql_reads_only_the_inventory(stories: StoriesConfig) -> None:
    """Every relation after FROM or JOIN is a table or view of the inventory, or a CTE of the same file (a function
    after FROM, such as generate_series, is not a relation)."""
    inventory = InventoryConfig.load()
    known = {table.name for table in inventory.tables} | {view.name for view in inventory.views}
    for path in sorted(STORIES_DIR.glob("*/*.sql")):
        code = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("--"))
        ctes = set(re.findall(r"(?:WITH|,)\s*(\w+)(?: \([^)]*\))? AS \(", code))
        named = set(re.findall(r"\b(?:FROM|JOIN)\s+(\w+)\b(?!\.|\()", code, re.I)) - {"LATERAL"}
        assert named - ctes <= known, f"{path.name}: {sorted(named - ctes - known)}"


def test_each_story_draws_from_a_named_stream(stories: StoriesConfig) -> None:
    assert {story.seed_stream for story in stories.stories} <= STREAMS


def test_titles_read_as_czech(stories: StoriesConfig) -> None:
    titles = [story.title_cs for story in stories.stories] + [anomaly.title_cs for anomaly in stories.anomalies]
    assert_czech_all(titles, what="the answer key's titles")


# ── the SQL repeats codes of the YAML: kept equal ───────────────────────────


def _weeks(ranges: Any) -> list[int]:
    weeks: list[int] = []
    for low, high in ranges:
        weeks += list(range(low, high + 1)) if low <= high else [*range(low, 54), *range(1, high + 1)]
    return weeks


def test_s1_weeks_are_the_curves(stories: StoriesConfig) -> None:
    curves = stories.story("S1").parameters["curves"]
    sql = _sql("detectors/s1.sql")
    for line, curve in curves.items():
        row = f"('{line}', ARRAY{_weeks(curve['peak_weeks'])}, ARRAY{_weeks(curve['trough_weeks'])})".replace(" ", "")
        assert (row.replace(",ARRAY", ", ARRAY") in sql) is bool(curve["peak_weeks"]), line
    assert set(curves) == {line for line in WorldConfig.load().pricing.max_agreed_discount_pct} - {
        "PROFIKAVA",
        "PROFIGASTRO",
    }


def test_s1_and_s2_locate_an_order_where_it_is_sold() -> None:
    """S1-weather and S2-affluence read an order's place by one rule: a store, an own shop or a web consumer's town.
    Any other delivery point without a store (a chain's DC, a vending depot, a wholesaler's warehouse) drops out."""
    sold = re.compile(r"dp\.store_id IS NOT NULL OR o\.channel IN \([^)]*\)")
    found = {name: sold.findall(_sql(f"detectors/{name}.sql")) for name in ("s1_weather", "s2_affluence")}
    assert found == {name: ["dp.store_id IS NOT NULL OR o.channel IN ('web', 'own_shops')"] for name in found}


def test_s3_names_the_pinned_stars_and_weak_reps(stories: StoriesConfig) -> None:
    expected = [(pin.slot, pin.profile) for pin in stories.pins.reps if pin.profile in {"star", "weak"}]
    assert _values(_sql("detectors/s3.sql"), "pinned") == expected


def test_s4_names_the_leading_lines(stories: StoriesConfig) -> None:
    leading = stories.story("S4").parameters["leading_line"]
    assert _values(_sql("detectors/s4.sql"), "expected") == list(leading.items())


def test_s5_names_the_managers_skills(stories: StoriesConfig) -> None:
    skill = stories.story("S5").parameters["pos_skill"]
    assert [(region, float(value)) for region, value in _values(_sql("detectors/s5.sql"), "skill")] == list(
        skill.items()
    )
    generous = set(stories.story("S5").parameters["budget_factor"])
    best = max(skill, key=skill.__getitem__)
    worst = min((region for region in skill if region not in generous), key=skill.__getitem__)
    assert re.search(
        rf"FILTER \(WHERE region = '{best}'\)\n    / NULLIF\(GREATEST\(MAX\(gain / NULLIF\(cost, 0\)\)"
        rf" FILTER \(WHERE region = '{worst}'\)",
        _sql("detectors/s5_best_worst.sql"),
    )


def test_s5s_spread_waits_for_its_threshold(stories: StoriesConfig) -> None:
    """The spread is gain ÷ cost, not 04's return, so 04's 3 is provisional until G2 S2.3 T1 derives it from the
    mechanics: the YAML, the SQL's header and the stories README say so."""
    (spread,) = stories.story("S5").also
    assert (spread.sql, spread.metric, spread.threshold, spread.direction) == (
        "detectors/s5_best_worst.sql",
        "best_over_worst_gain_per_cost",
        3,
        ">=",
    )
    assert re.search(r"The threshold,\s+(-- )?3, is provisional", _sql(spread.sql))
    assert "# Provisional." in _sql("s5.yaml")
    assert "One is still to be set that way: S5's spread" in " ".join(_sql("README.md").split())


def test_s5s_two_files_judge_one_return() -> None:
    """The spread and the rank read the same region return: one block, byte for byte in both files."""

    def block(sql: str) -> str:
        found = re.search(r"^-- region return: begin\n.*^-- region return: end\n", sql, re.S | re.M)
        assert found
        return found.group(0)

    assert block(_sql("detectors/s5.sql")) == block(_sql("detectors/s5_best_worst.sql"))
    assert "giveaway AS (" in block(_sql("detectors/s5.sql"))


def test_s6_names_the_pinned_reps_and_managers(stories: StoriesConfig) -> None:
    parameters = stories.story("S6").parameters
    low, high = parameters["discipline_range"]
    reps = [(slot, "true" if discipline >= 1 else "false") for slot, discipline in parameters["pinned_reps"].items()]
    assert _values(_sql("detectors/s6.sql"), "pinned") == reps
    managers = parameters["pinned_product_managers"]
    assert _values(_sql("detectors/s6_targets.sql"), "pinned") == [
        (line, "true" if discipline >= 1 else "false") for line, discipline in managers.items()
    ]
    assert all(low <= value <= high for value in [*parameters["pinned_reps"].values(), *managers.values()])
    # The metric is a difference, not a ratio (the undisciplined gain is negative by design): the YAML, the SQL's
    # header and its final SELECT say the same.
    detector = stories.story("S6").detector
    assert (detector.metric, detector.threshold, detector.direction) == (
        "disciplined_minus_undisciplined_margin_per_discount",
        0.5,
        ">=",
    )
    sql = _sql("detectors/s6.sql")
    assert "gain per crown minus the undisciplined reps'" in " ".join(sql.split())
    assert "gives at least 0.5" in " ".join(sql.split())
    assert re.search(r"FILTER \(WHERE disciplined\), 0\)\n    - SUM\(margin_gain\)", sql)


def test_no_window_ends_on_a_fixed_last_day(stories: StoriesConfig) -> None:
    """A prolonged world is judged to its new end: no detector or probe stops at the original last day."""
    files = {detector.sql for story in stories.stories for detector in story.detectors}
    files |= {anomaly.probe.sql for anomaly in stories.anomalies}
    assert [name for name in sorted(files) if "2026-10-31" in _sql(name)] == []


# ── the pins (design 04 §S3, §7) ────────────────────────────────────────────


def test_the_eight_pinned_reps_of_04(stories: StoriesConfig, ref: Path) -> None:
    pins = {pin.profile: pin for pin in stories.pins.reps}
    profiles = Counter(pin.profile for pin in stories.pins.reps)
    assert profiles == {"star": 2, "weak": 2, "hunter": 1, "farmer": 1, "cross_seller": 1, "converter": 1}
    for pin in stories.pins.reps:
        skills = (pin.conversion, pin.acquisition, pin.retention, pin.cross_sell)
        if pin.profile == "star":
            assert min(skills) >= 1.3
        if pin.profile == "weak":
            assert max(skills) <= 0.7
    assert (pins["hunter"].acquisition, pins["hunter"].retention) == (1.4, 0.65)
    assert (pins["farmer"].retention, pins["farmer"].acquisition) == (1.35, 0.7)
    assert pins["cross_seller"].cross_sell == 1.4
    assert pins["converter"].conversion == 1.35
    areas = {row["code"] for row in read("area", ref)}
    slots = [pin.slot for pin in stories.pins.reps]
    assert len(set(slots)) == 8 and set(slots) <= areas
    assert not set(slots) & set(stories.story("S6").parameters["pinned_reps"])


def test_a7_and_a8_fall_on_pinned_reps(stories: StoriesConfig) -> None:
    """04 §7: three reps skip visits, one of them with the best conversion; one rep records nothing."""
    skippers = [pin for pin in stories.pins.reps if pin.skips_visits]
    assert len(skippers) == 3
    assert max(pin.conversion for pin in stories.pins.reps) in {pin.conversion for pin in skippers}
    assert len([pin for pin in stories.pins.reps if pin.records_nothing]) == 1


def test_the_pinned_customers_resolve_by_rule(stories: StoriesConfig, ref: Path) -> None:
    areas = {row["code"] for row in read("area", ref)}
    a1 = [pin for pin in stories.pins.customers if pin.anomaly == "A-1"]
    assert Counter(pin.channel for pin in a1) == {"horeca": 2, "small_retail": 1}
    (franchise,) = [pin.chain for pin in a1 if pin.chain is not None]
    assert BY_CODE[franchise].kind == "convenience_franchise"
    assert {pin.anomaly for pin in stories.pins.customers} == {"A-1", "A-13", "A-14"}
    assert all(pin.area in areas for pin in stories.pins.customers)


def test_the_cases_name_real_codes(stories: StoriesConfig, ref: Path) -> None:
    skus = {row["sku_code"] for row in read("catalogue", ref)}
    sites = {row["code"] for row in read("site", ref)}
    for anomaly in stories.anomalies:
        parameters = anomaly.parameters
        for key in ("sku", "sibling"):
            if key in parameters:
                assert parameters[key] in skus, anomaly.id
        for key in ("dc", "shop"):
            if key in parameters:
                assert parameters[key] in sites, anomaly.id
        for code in parameters.get("chains", ()) + ((parameters["chain"],) if "chain" in parameters else ()):
            assert code in BY_CODE, anomaly.id
        named = [parameters[key] for key in ("promotion", "sku") if key in parameters]
        assert not named or named[0] in _sql(anomaly.probe.sql), anomaly.id
    code = stories.anomaly("A-10").parameters["promotion"]
    assert re.fullmatch(r"PR-2025-10-CLASSIC-\w+", code)


# ── the hash and a broken key ───────────────────────────────────────────────


@pytest.fixture
def stories_copy(tmp_path: Path) -> Path:
    target = tmp_path / "stories"
    shutil.copytree(STORIES_DIR, target)
    return target


def test_the_hash_follows_the_yaml(stories: StoriesConfig, stories_copy: Path) -> None:
    assert StoriesConfig.load(stories_copy).sha256 == stories.sha256
    path = stories_copy / "s2.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("tv_factor: { tv: 1.4", "tv_factor: { tv: 1.5"))
    assert StoriesConfig.load(stories_copy).sha256 != stories.sha256


def _edit(path: Path, change: Any) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_a_null_world_in_the_same_direction_fails(stories_copy: Path) -> None:
    _edit(stories_copy / "s4.yaml", lambda data: data["detector"].update(null_world=">="))
    with pytest.raises(ConfigError, match=r"s4\.yaml: detectors/s4\.sql: the null world must be the opposite"):
        StoriesConfig.load(stories_copy)


def test_a_missing_detector_and_a_missing_probe_fail_together(stories_copy: Path) -> None:
    (stories_copy / "detectors" / "s2_affluence.sql").unlink()
    (stories_copy / "probes" / "a9.sql").unlink()
    with pytest.raises(ConfigError) as error:
        StoriesConfig.load(stories_copy)
    assert str(error.value).splitlines() == [
        "s2.yaml: no detectors/s2_affluence.sql",
        "anomalies.yaml: A-9: no probes/a9.sql",
    ]


def test_the_parameters_dates_are_dates(stories: StoriesConfig) -> None:
    assert stories.anomaly("A-2").parameters["launched_on"] == date(2025, 9, 1)
    assert stories.anomaly("A-9").parameters["month"] == date(2026, 3, 1)

    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else []
        if isinstance(value, Mapping):
            return [text for item in value.values() for text in strings(item)]
        if isinstance(value, tuple):
            return [text for item in value for text in strings(item)]
        return []

    parameters = [story.parameters for story in stories.stories] + [anomaly.parameters for anomaly in stories.anomalies]
    assert strings(tuple(parameters)) == []


def test_two_missing_keys_of_a_detector_are_both_named(stories_copy: Path) -> None:
    def two(data: dict[str, Any]) -> None:
        data["detector"].pop("metric")
        data["detector"].pop("threshold")

    _edit(stories_copy / "s4.yaml", two)
    with pytest.raises(ConfigError) as error:
        StoriesConfig.load(stories_copy)
    assert str(error.value).splitlines() == [
        "s4.yaml: detector.metric: missing",
        "s4.yaml: detector.threshold: missing",
    ]


def test_a_story_without_its_detector_fails_with_the_path(stories_copy: Path) -> None:
    _edit(stories_copy / "s5.yaml", lambda data: data.pop("detector"))
    with pytest.raises(ConfigError, match=r"^s5\.yaml: detector: missing$"):
        StoriesConfig.load(stories_copy)


def test_dry_lists_the_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stories-check", "--dry"]) == 0
    out = capsys.readouterr().out
    assert "stories-check --dry: 6 stories, 10 detectors, 15 probes — nothing run" in out
    assert "      detectors/s1.sql             min_peak_over_trough >= 1.25 · null world <" in out
    assert "      detectors/s5_best_worst.sql  best_over_worst_gain_per_cost >= 3.0 · null world <" in out
