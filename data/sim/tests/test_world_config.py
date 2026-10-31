"""G0 S0.4 T2 — `data/world.yaml`: every number of 01 under its key, validated, typed (`config.WorldConfig`)."""

from __future__ import annotations

import dataclasses
from functools import reduce
from pathlib import Path
from typing import Any

import pytest
import yaml

from hartland_sim.config import WORLD_FILE, ConfigError, WorldConfig
from hartland_sim.io.reference import read
from hartland_sim.setup.classifier.chains import CHAINS


@pytest.fixture(scope="module")
def world() -> WorldConfig:
    return WorldConfig.load()


def _get(world: WorldConfig, dotted: str) -> Any:
    def step(node: Any, name: str) -> Any:
        return node[name] if hasattr(node, "keys") else getattr(node, name)

    return reduce(step, dotted.split("."), world)


# 01-company.md's numbers, where 01 states them (⚑NH-5…8 as ruled, NH-D5).
DESIGN_NUMBERS = {
    "sites.own_shops": 12,
    "channels.small_retail.customers": 5000,
    "channels.small_retail.active_share_at_start": 0.40,
    "channels.web.consumers": 30000,
    "channels.vending.operators": 40,
    "channels.horeca.direct_customers": 600,
    "channels.horeca.wholesalers": 2,
    "salesforce.reps": 50,
    "salesforce.area_managers": 6,
    "salesforce.kams.big_retail": 8,
    "salesforce.kams.gas_ka": 1,
    "salesforce.kams.vending": 3,
    "salesforce.kams.horeca": 4,
    "salesforce.product_managers": 11,
    "pricing.channel_factor.big_retail": 0.86,
    "pricing.channel_factor.gas_ka": 0.90,
    "pricing.channel_factor.vending": 0.80,
    "pricing.channel_factor.horeca": 0.88,
    "pricing.channel_factor.own_shops": 0.75,
    "pricing.channel_factor.web": 1.12,
    "pricing.tiers.cross_sell_both_brands_pct": 1,
    "pricing.tiers.cross_sell_max_pct": 3,
    "pricing.budgets.rep_discount_pct_of_planned_gross": 2,
    "promotions.per_year": 25,
    "promotions.weeks": (4, 8),
    "promotions.key_account_exclusive_per_year": 4,
    "policies.stock.default.min_days": 10,
    "policies.stock.default.max_days": 35,
    "policies.stock.professional.min_days": 15,
    "policies.stock.professional.max_days": 45,
    "policies.prompt_caps.potential": 3,
    "policies.prompt_caps.occasional": 4,
    "policies.prompt_caps.regular": 5,
    "policies.prompt_caps.vip": 6,
    "finance.payment_terms_days.big_retail": 60,
    "finance.payment_terms_days.gas_ka": 45,
    "finance.payment_terms_days.vending": 30,
    "finance.payment_terms_days.horeca": 30,
    "finance.payment_terms_days.small_retail": 14,
    "finance.payment_behaviour.on_time": 0.70,
    "finance.payment_behaviour.late_14": 0.22,
    "finance.payment_behaviour.late_30": 0.06,
    "finance.payment_behaviour.late_more": 0.02,
    "finance.hold.overdue_days_over": 21,
    "finance.hold.exposure_over_limit": True,
    "finance.vat_pct.goods": 12,
    "finance.vat_pct.served": 21,
    "calendar.coffee_jump.pct": 9,
    "delivery.carriers_per_dc": 3,
    "delivery.events_per_route_day": 0.03,
    "dials.receipts_per_shop_day": 150,
}


@pytest.mark.parametrize(("dotted", "expected"), DESIGN_NUMBERS.items())
def test_the_design_number_is_under_its_key(world: WorldConfig, dotted: str, expected: Any) -> None:
    assert _get(world, dotted) == expected


def test_the_tiers_of_01_5_2(world: WorldConfig) -> None:
    volume = [(tier.code, tier.min_net_revenue, tier.pct) for tier in world.pricing.tiers.volume]
    assert volume == [("VOL-T1", 20000, 2), ("VOL-T2", 50000, 4), ("VOL-T3", 120000, 6)]
    cross = [(tier.code, tier.min_lines, tier.pct) for tier in world.pricing.tiers.cross_sell]
    assert cross == [("XS-T1", 3, 1), ("XS-T2", 5, 2)]


def test_every_line_has_a_max_agreed_discount(world: WorldConfig, ref: Path) -> None:
    lines = {row["code"] for row in read("product_line", ref)}
    assert set(world.pricing.max_agreed_discount_pct) == lines
    assert world.pricing.max_agreed_discount_pct["GOLD"] == 5
    assert world.pricing.max_agreed_discount_pct["PROFIKAVA"] == 12


def test_the_lines_named_anywhere_exist(world: WorldConfig, ref: Path) -> None:
    lines = {row["code"] for row in read("product_line", ref)}
    assert set(world.policies.seasonal_exclusion) <= lines
    for listing in world.channels.listings.values():
        for named in (listing.retail, listing.professional):
            assert named == "all" or set(named) <= lines


def test_every_channel_has_a_factor_a_term_and_a_listing(world: WorldConfig) -> None:
    channels = {field.name for field in dataclasses.fields(world.channels)} - {"listings"}
    assert set(world.pricing.channel_factor) == channels
    assert set(world.finance.payment_terms_days) == channels
    assert set(world.channels.listings) == channels


def test_the_hero_overrides_name_catalogue_skus(world: WorldConfig, ref: Path) -> None:
    skus = {row["sku_code"]: row for row in read("catalogue", ref)}
    for override in world.pricing.hero_overrides:
        assert skus[override.sku]["importance"] == 10


def test_the_key_accounts_are_the_classifiers(world: WorldConfig) -> None:
    accounts = world.channels.big_retail.accounts + world.channels.gas_ka.accounts
    assert sorted(accounts) == sorted(chain.code for chain in CHAINS if chain.is_key_account)
    assert set(world.channels.big_retail.dc_level_orders) <= set(world.channels.big_retail.accounts)
    assert set(world.pricing.retro_bonus_pct) - {"default"} <= set(accounts)


def test_the_dcs_serve_every_region_once(world: WorldConfig, ref: Path) -> None:
    regions = [region for served in world.sites.dcs.values() for region in served]
    assert sorted(regions) == sorted(row["code"] for row in read("region", ref))
    sites = {row["code"]: row["kind"] for row in read("site", ref)}
    assert {sites[dc] for dc in world.sites.dcs} == {"dc"}
    assert sites[world.sites.hq] == "hq"


def test_the_codes_name_reference_rows(world: WorldConfig, ref: Path) -> None:
    entities = {row["code"]: row["role"] for row in read("legal_entity", ref)}
    assert entities[world.company.center] == "center"
    assert {entities[code] for code in world.company.factories} == {"factory"}
    assert set(world.suppliers.coffee_split) | {world.suppliers.fallback} <= set(world.company.factories)
    assert world.salesforce.reps == len(read("area", ref))


def test_the_shares_add_up(world: WorldConfig) -> None:
    assert sum(dataclasses.astuple(world.finance.payment_behaviour)) == pytest.approx(1)
    assert sum(world.suppliers.coffee_split.values()) == pytest.approx(1)


def test_the_history_runs_in_order(world: WorldConfig) -> None:
    history = world.history
    assert history.warm_up_from < history.first_day < history.until
    assert history.first_day.isoformat() == "2024-01-01"


def test_the_view_is_read_only(world: WorldConfig) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        world.master_seed = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        world.pricing.channel_factor["web"] = 2.0  # type: ignore[index]
    assert isinstance(world.pricing.tiers.volume, tuple)


# ── a broken file fails with the path of each problem ───────────────────────


def _broken(tmp_path: Path, change: Any) -> Path:
    data = yaml.safe_load(WORLD_FILE.read_text(encoding="utf-8"))
    change(data)
    path = tmp_path / "world.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_a_missing_key_fails_with_its_path(tmp_path: Path) -> None:
    path = _broken(tmp_path, lambda data: data["pricing"]["tiers"].pop("volume"))
    with pytest.raises(ConfigError, match=r"^world\.yaml: pricing\.tiers\.volume: missing$"):
        WorldConfig.load(path)


def test_a_mistyped_value_fails_with_its_path(tmp_path: Path) -> None:
    path = _broken(tmp_path, lambda data: data["finance"]["hold"].update(overdue_days_over="21"))
    with pytest.raises(ConfigError, match=r"world\.yaml: finance\.hold\.overdue_days_over: '21' is not of type"):
        WorldConfig.load(path)


def test_every_problem_is_listed_at_once(tmp_path: Path) -> None:
    def two(data: dict[str, Any]) -> None:
        data.pop("dials")
        data["channels"]["web"]["order_grain"] = "pallet"
        data["surprise"] = 1

    with pytest.raises(ConfigError) as error:
        WorldConfig.load(_broken(tmp_path, two))
    lines = str(error.value).splitlines()
    assert len(lines) == 3
    assert "world.yaml: dials: missing" in lines
    assert any(line.startswith("world.yaml: channels.web.order_grain: 'pallet'") for line in lines)
    assert any("'surprise' was unexpected" in line for line in lines)


def test_a_bad_date_fails_with_its_path(tmp_path: Path) -> None:
    path = _broken(tmp_path, lambda data: data["history"].update(until="2026-02-30"))
    with pytest.raises(ConfigError, match=r"history\.until: '2026-02-30' is not a 'date'"):
        WorldConfig.load(path)


def test_not_yaml_fails_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "world.yaml"
    path.write_text("master_seed: [1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"^world\.yaml: not YAML"):
        WorldConfig.load(path)
