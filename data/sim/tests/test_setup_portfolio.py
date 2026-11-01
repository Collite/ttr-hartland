"""Setup steps 4 and 6 (G1 S1.1 T5): the portfolio, the listings, the supplier terms, the stock policies, the rules,
the discount budgets, PoS materials and assets, carriers and vehicles."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Any

import pytest

from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.config import Channel, StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import read
from hartland_sim.setup.policies import budgets
from hartland_sim.setup.portfolio import CHANNELS, listed
from hartland_sim.setup.rows import tables
from hartland_sim.setup.run import Setup
from hartland_sim.world import OPEN, Budget

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()


@pytest.fixture(scope="module")
def setup(opening: Setup) -> Setup:
    return opening


def _rows(setup: Setup, table: str) -> list[dict[str, Any]]:
    return setup.tables[table]


def test_the_hierarchy_is_the_catalogue(setup: Setup) -> None:
    counts = {table: len(_rows(setup, table)) for table in ("business", "brand", "product_line", "product", "sku")}
    assert counts == {"business": 2, "brand": 2, "product_line": 11, "product": 48, "sku": 123}
    on_sale = [
        sku for sku in _rows(setup, "sku") if sku["delisted_on"] is None or sku["delisted_on"] > date(2026, 10, 31)
    ]
    assert len(on_sale) == 117
    launched = [sku for sku in _rows(setup, "sku") if sku["launched_on"] >= date(2024, 1, 1)]
    assert launched and all(sku["launched_on"] <= date(2026, 10, 31) for sku in launched)
    lines = {line["code"]: line["max_agreed_discount_pct"] for line in _rows(setup, "product_line")}
    assert lines == dict(WORLD.pricing.max_agreed_discount_pct)


def test_names_read_as_czech(setup: Setup) -> None:
    brands = ["Harcafé", "Harsoup"]
    assert_czech_all([sku["name"] for sku in _rows(setup, "sku")], what="sku.name", allow=brands)
    assert_czech_all([p["name"] for p in _rows(setup, "product")], what="product.name", allow=brands)
    assert_czech_all([m["name"] for m in _rows(setup, "pos_material")], what="pos_material.name", allow=brands)
    assert_czech_all([c["name"] for c in _rows(setup, "carrier")], what="carrier.name")
    assert_czech_all([u["name"] for u in _rows(setup, "org_unit")], what="org_unit.name", allow=brands)


def _listing(setup: Setup) -> dict[tuple[Channel, int | None], list[dict[str, Any]]]:
    out: dict[tuple[Channel, int | None], list[dict[str, Any]]] = defaultdict(list)
    for item in _rows(setup, "channel_listing"):
        out[item["channel"], item["chain_id"]].append(item)
    return out


def test_each_channel_lists_its_world_yaml_lines(setup: Setup) -> None:
    by_group = _listing(setup)
    chains = [row_id("chain", code=code) for code in WORLD.channels.big_retail.accounts]
    assert sorted(key for key in by_group if key[0] == "big_retail") == sorted(("big_retail", c) for c in chains)
    assert {channel for channel, _ in by_group} == set(CHANNELS)  # no `special_retail`: a sub-channel
    for (channel, _), items in by_group.items():
        expected = {sku.sku_id for sku in setup.portfolio.skus if listed(WORLD.channels.listings[channel], sku)}
        assert {item["sku_id"] for item in items} == expected, channel
    professional = {sku.sku_id for sku in setup.portfolio.skus if sku.is_professional}
    assert not professional & {item["sku_id"] for item in by_group["web", None]}
    assert {item["sku_id"] for item in by_group["horeca", None]} == professional


def test_a_listing_runs_from_launch_to_delisting_in_importance_order(setup: Setup) -> None:
    for items in _listing(setup).values():
        assert [item["listing_sequence"] for item in items] == list(range(1, len(items) + 1))
        importance = [setup.portfolio.sku(_code(setup, item["sku_id"])).importance for item in items]
        assert importance == sorted(importance, reverse=True)
    for item in _rows(setup, "channel_listing"):
        sku = setup.portfolio.sku(_code(setup, item["sku_id"]))
        assert item["valid_from"] == sku.launched_on
        if sku.delisted_on is not None:
            assert item["valid_to"] == sku.delisted_on - timedelta(days=1)


def _code(setup: Setup, sku_id: int) -> str:
    (code,) = [sku.code for sku in setup.portfolio.skus if sku.sku_id == sku_id]
    return code


def test_a9_two_chains_delist_espresso_decaf_in_march_2026(setup: Setup) -> None:
    a9 = STORIES.anomaly("A-9").parameters
    sku = row_id("sku", sku_code=a9["sku"])
    ending = {
        item["chain_id"]: item["valid_to"]
        for item in _rows(setup, "channel_listing")
        if item["sku_id"] == sku and item["chain_id"] is not None
    }
    delisting = {row_id("chain", code=chain) for chain in a9["chains"]}
    assert {chain for chain, end in ending.items() if date(2026, 3, 1) <= end <= date(2026, 3, 31)} == delisting
    assert all(end == OPEN for chain, end in ending.items() if chain not in delisting)


def test_seasonal_exclusions_and_supplier_terms(setup: Setup) -> None:
    exclusions = Counter(item["product_line_id"] for item in _rows(setup, "seasonal_exclusion"))
    assert exclusions == {
        row_id("product_line", code=line): len(m) for line, m in WORLD.policies.seasonal_exclusion.items()
    }
    shares: dict[int, float] = defaultdict(float)
    for terms in _rows(setup, "supplier_terms"):
        shares[terms["sku_id"]] += terms["sourcing_share"]
        supplier = {row_id("legal_entity", code=c): c for c in ("HKO", "HHO", "HCH")}[terms["supplier_entity_id"]]
        assert terms["lead_time_days"] == WORLD.suppliers.lead_time_days[supplier]
        assert terms["min_lot_cases"] == WORLD.suppliers.min_lot_cases[supplier]
    assert set(shares) == {sku.sku_id for sku in setup.portfolio.skus}
    assert all(abs(share - 1.0) < 1e-9 for share in shares.values())
    split = [t for t in _rows(setup, "supplier_terms") if t["sourcing_share"] < 1]
    assert split and {t["sourcing_share"] for t in split} == set(WORLD.suppliers.coffee_split.values())


def test_stock_policies_per_sku_and_dc(setup: Setup) -> None:
    policies = _rows(setup, "stock_policy")
    assert len(policies) == 2 * 123
    for policy in policies:
        sku = setup.portfolio.sku(_code(setup, policy["sku_id"]))
        cover = WORLD.policies.stock["professional" if sku.is_professional else "default"]
        assert (policy["min_days_cover"], policy["max_days_cover"]) == (cover.min_days, cover.max_days)


def test_the_rules_are_world_yaml_s_owned_by_the_sales_director(setup: Setup) -> None:
    prompts = _rows(setup, "prompt_rule")
    assert [rule["code"] for rule in prompts] == [rule.code for rule in WORLD.policies.prompt_rules]
    director = setup.org.actors["management:OU-OBC"].employee_id
    assert {rule["owner_employee_id"] for rule in prompts} == {director}
    assert all(json.loads(rule["parameters"]) for rule in prompts)
    lifecycle = _rows(setup, "lifecycle_rule")
    assert Counter((rule["phase"], rule["rule_kind"]) for rule in lifecycle) == {
        (phase, kind): 1
        for phase in ("potential", "occasional", "regular", "vip")
        for kind in ("contact_frequency", "concessions", "listing_sequence")
    }
    vip = {rule["rule_kind"]: json.loads(rule["parameters"]) for rule in lifecycle if rule["phase"] == "vip"}
    assert vip["contact_frequency"]["prompts_per_visit"] == WORLD.policies.prompt_caps["vip"]
    assert vip["contact_frequency"]["entry"]["min_net_revenue"] == WORLD.policies.lifecycle.vip_min_net_revenue
    assert all(rule["valid_from"] == WORLD.history.warm_up_from for rule in prompts + lifecycle)


def test_budgets_are_two_percent_of_the_opening_quarter_s_planned_gross() -> None:
    out = tables()
    state = budgets(out, WORLD, [(1, "small_retail", 80), (2, "big_retail", 2), (2, "gas_ka", 1)])
    gross = WORLD.pricing.budgets.opening_quarter_gross
    quarter = date(2023, 1, 1)
    assert state == {
        (1, quarter): Budget(round(0.02 * 80 * gross["small_retail"], 2)),
        (2, quarter): Budget(round(0.02 * (2 * gross["big_retail"] + gross["gas_ka"]), 2)),
    }
    assert [(b["employee_id"], b["quarter"], b["used_amount"]) for b in out["discount_budget"]] == [
        (1, quarter, 0.0),
        (2, quarter, 0.0),
    ]


def test_pos_materials_and_each_region_s_asset_pool(setup: Setup) -> None:
    materials = read("pos_material")
    assert len(_rows(setup, "pos_material")) == len(materials) and 20 <= len(materials) <= 30
    per_region = sum(material["assets_per_region"] for material in materials)
    assets = _rows(setup, "pos_asset")
    assert Counter(asset["region_id"] for asset in assets) == {
        row_id("region", code=region): per_region for region in ("PHA", "SJC", "ZSC", "VCH", "JMV", "SMS")
    }
    assert 2500 <= len(assets) <= 3500
    assert all(date(2019, 1, 1) <= asset["acquired_on"] < WORLD.history.warm_up_from for asset in assets)
    conditions = Counter(asset["condition"] for asset in assets)
    assert set(conditions) == {"new", "good", "worn"}
    assert len({asset["asset_no"] for asset in assets}) == len(assets)


def test_carriers_and_their_fleets(setup: Setup) -> None:
    carriers = _rows(setup, "carrier")
    per_dc = Counter(carrier["dc_site_id"] for carrier in carriers)
    assert per_dc == {row_id("site", code="DC-JIRNY"): 3, row_id("site", code="DC-MODRICE"): 3, None: 2}
    vehicles = _rows(setup, "vehicle")
    assert len(vehicles) == 60
    assert len({vehicle["plate"] for vehicle in vehicles}) == 60
    assert all(re.fullmatch(r"[1-9][SBA][A-Z0-9] [0-9]{4}", vehicle["plate"]) for vehicle in vehicles)
    for vehicle in vehicles:
        capacity = WORLD.delivery.capacity[vehicle["vehicle_class"]]
        assert (vehicle["capacity_cases"], vehicle["capacity_kg"]) == (capacity.cases, capacity.kg)
