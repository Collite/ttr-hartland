"""G0 S0.4 T6 — the template seeds: the promotion calendar's shapes, the visit-note sentences, the letter and
marketing seeds (DOC.1 fills those)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from hartland_sim.checks.czech import assert_czech, assert_czech_all
from hartland_sim.config import ConfigError, InventoryConfig, StoriesConfig, WorldConfig, read_yaml, validate
from hartland_sim.io.reference import read
from hartland_sim.setup.classifier.chains import BY_CODE
from hartland_sim.setup.promotions import TEMPLATES_FILE, PlannedPromotion, Templates, calendar

HISTORY_YEARS = (2024, 2025, 2026)


SLOT = re.compile(r"\{\{ (\w+) \}\}")


@pytest.fixture(scope="module")
def templates() -> Templates:
    return Templates.load()


@pytest.fixture(scope="module")
def world() -> WorldConfig:
    return WorldConfig.load()


def _year(templates: Templates, world: WorldConfig, year: int) -> list[PlannedPromotion]:
    return calendar(templates, year, world.promotions.code)


@pytest.mark.parametrize("year", HISTORY_YEARS)
def test_a_years_calendar_is_25_promotions_a_third_with_tv(templates: Templates, world: WorldConfig, year: int) -> None:
    """⚑NH-8: ~25 a year, a third TV-supported, each 4–8 weeks, all four mechanics, ~4 key-account-only."""
    promotions = _year(templates, world, year)
    assert 23 <= len(promotions) <= 27
    tv = sum(promotion.template.tv for promotion in promotions)
    assert abs(tv - len(promotions) / 3) <= 2
    low, high = world.promotions.weeks
    assert all(low * 7 <= (p.valid_to - p.valid_from).days + 1 <= high * 7 for p in promotions)
    assert {p.template.mechanic for p in promotions} == set(world.promotions.mechanics)
    key_account_only = [p for p in promotions if dict(p.template.eligibility) in ({"channel": ("big_retail",)},)]
    key_account_only += [p for p in promotions if "chain" in p.template.eligibility]
    assert abs(len(key_account_only) - world.promotions.key_account_exclusive_per_year) <= 1
    pairs = {tuple(sorted((p.code, other))) for p in promotions for other in p.excludes}
    low_pairs, high_pairs = world.promotions.exclusion_pairs_per_year
    assert low_pairs <= len(pairs) <= high_pairs


@pytest.mark.parametrize("year", HISTORY_YEARS)
def test_the_codes_are_stable_and_unique(templates: Templates, world: WorldConfig, year: int) -> None:
    promotions = _year(templates, world, year)
    codes = [p.code for p in promotions]
    assert len(set(codes)) == len(codes)
    for promotion in promotions:
        assert re.fullmatch(rf"PR-{year}-\d\d-[A-Z0-9]+-[A-Z0-9]+", promotion.code)
        assert promotion.code[8:10] == f"{promotion.valid_from.month:02d}"
    assert _year(templates, world, year) == promotions


def test_every_start_falls_in_its_month_for_years_to_come(templates: Templates, world: WorldConfig) -> None:
    """The warm-up year and the prolongs ahead (G5 S5.3) get the same months, so the codes keep their month."""
    for year in range(2023, 2036):
        _year(templates, world, year)


def test_the_tv_eight_are_s2s(templates: Templates, ref: Path) -> None:
    tv = Counter(template.line for template in templates.templates if template.tv)
    assert tv == StoriesConfig.load().story("S2").parameters["tv_promotions_per_year"]
    # The comparison is fair: each TV mechanic runs on a premium line and on a basic one.
    premium = {row["code"] for row in read("product_line", ref) if row["is_premium"]}
    by_mechanic: dict[tuple[str, float | None], set[bool]] = {}
    for template in templates.templates:
        if template.tv:
            by_mechanic.setdefault((template.mechanic, template.discount_pct), set()).add(template.line in premium)
    assert all(sides == {True, False} for sides in by_mechanic.values())


def test_a10s_promotion_is_on_the_calendar(templates: Templates, world: WorldConfig) -> None:
    code = StoriesConfig.load().anomaly("A-10").parameters["promotion"]
    (promotion,) = [p for p in _year(templates, world, 2025) if p.code == code]
    assert promotion.template.mechanic == "price_off"
    assert promotion.template.target_sku == StoriesConfig.load().anomaly("A-10").parameters["sku"]


def test_the_templates_name_real_lines_skus_and_scopes(templates: Templates, ref: Path) -> None:
    inventory = InventoryConfig.load()
    skus = {row["sku_code"]: row for row in read("catalogue", ref)}
    product_line = {row["code"]: row["product_line"] for row in read("product", ref)}
    lines = {row["code"]: row for row in read("product_line", ref)}
    for template in templates.templates:
        assert template.line in lines, template.key
        for sku in (template.condition_sku, template.reward_sku, template.target_sku):
            if sku is not None:
                assert product_line[skus[sku]["product"]] in {template.line, "SMESI"}, (template.key, sku)
                assert skus[sku]["delisted_on"] is None, (template.key, sku)
        scopes = template.eligibility
        assert set(scopes.get("channel", ())) <= set(inventory.codes["channel"]), template.key
        assert set(scopes.get("store_class", ())) <= set(inventory.codes["store_class"]), template.key
        assert set(scopes.get("chain", ())) <= set(BY_CODE), template.key
        if lines[template.line]["is_professional"]:
            assert set(scopes["channel"]) <= {"horeca", "vending"}, template.key


def test_the_rewards_exist_when_the_promotion_runs(templates: Templates, world: WorldConfig, ref: Path) -> None:
    launched = {row["sku_code"]: row["launched_on"] for row in read("catalogue", ref)}
    for year in HISTORY_YEARS:
        for promotion in _year(templates, world, year):
            for sku in (promotion.template.condition_sku, promotion.template.reward_sku, promotion.template.target_sku):
                if sku is not None:
                    assert launched[sku] <= promotion.valid_from, (promotion.code, sku)


def test_names_and_goals_read_as_czech(templates: Templates) -> None:
    assert_czech_all([t.name_cs for t in templates.templates], what="promotion names", allow=["Harcafé"])
    assert_czech_all([t.goal_cs for t in templates.templates], what="promotion goals")


def _broken(tmp_path: Path, change: Any) -> Path:
    data = yaml.safe_load(TEMPLATES_FILE.read_text(encoding="utf-8"))
    change(data)
    path = tmp_path / "promotion_templates.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_a_mechanic_without_its_fields_fails(tmp_path: Path) -> None:
    path = _broken(tmp_path, lambda data: data["templates"][0].pop("discount_pct"))
    with pytest.raises(ConfigError, match=r"templates\.0: missing|discount_pct"):
        Templates.load(path)


def test_an_exclusion_of_nothing_fails(tmp_path: Path) -> None:
    path = _broken(tmp_path, lambda data: data["exclusions"].append(["GOLD-TV", "GOLD-NIC"]))
    with pytest.raises(ConfigError, match=r"exclusion \('GOLD-TV', 'GOLD-NIC'\) names no template"):
        Templates.load(path)


# ── the visit notes ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def notes(ref: Path) -> Any:
    data = read_yaml(ref / "visit_note_templates.yaml")
    validate(data, "visit_note_templates", "visit_note_templates.yaml")
    return data


def test_every_visit_outcome_has_its_sentences(notes: Any) -> None:
    assert set(notes["outcomes"]) == set(InventoryConfig.load().codes["visit_outcome"])


def test_the_notes_use_the_declared_slots_and_read_as_czech(notes: Any) -> None:
    sentences = [s for group in (*notes["outcomes"].values(), *notes["situations"].values()) for s in group]
    used = {slot for sentence in sentences for slot in SLOT.findall(sentence)}
    assert used == set(notes["slots"])
    plain = [SLOT.sub("", sentence) for sentence in sentences]
    assert_czech_all(plain + notes["objections"], what="visit notes")


# ── the document seeds (DOC.1 fills the libraries) ──────────────────────────


@pytest.mark.parametrize(
    "seed",
    [
        "templates/letters/return_request/formalni-uplny-email.yaml",
        "templates/marketing/promotion_sheet/promocni-list.yaml",
    ],
)
def test_a_seed_in_doc1s_shape(ref: Path, seed: str) -> None:
    template = yaml.safe_load((ref / seed).read_text(encoding="utf-8"))
    assert set(template) == {"family", "variant", "channel", "slots", "injectors", "body"}
    assert template["family"] in InventoryConfig.load().codes["document_family"]
    assert set(SLOT.findall(template["body"])) == set(template["slots"])
    assert_czech(SLOT.sub("", template["body"]), what=seed)
