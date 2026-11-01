"""G0 S0.4 T4 — `data/inventory.yaml`: every table of 03 + 06 §3 (+ the return-request family), per contracts §1."""

from __future__ import annotations

import dataclasses
import json
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any

import pytest
import yaml

from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.config import INVENTORY_FILE, ConfigError, InventoryConfig, WorldConfig
from hartland_sim.setup.classifier.chains import CHAINS
from hartland_sim.setup.classifier.place import STORE_CLASSES, STREET_TYPES

# 03 §1–§12 and §13b, 06 §3, and the automation extension's return-request family — by the model's db files.
TABLES = {
    "reference": "legal_entity kraj okres site calendar region area area_okres weather_daily",
    "org": "cost_centre org_unit employee assignment",
    "customers": "chain store customer delivery_point lifecycle_phase competitor competitor_product",
    "products": "business brand product_line product sku channel_listing",
    "pricing": "standard_price_list standard_price channel_price_list channel_price customer_condition "
    "price_agreement discount_budget",
    "plans": "sales_plan supply_plan stock_policy prompt_rule lifecycle_rule",
    "orders": "sales_order sales_order_line sales_order_line_discount return",
    "finance": "invoice invoice_line payment payment_allocation dunning supplier ap_invoice gl_account gl_journal "
    "asset_register",
    "supply": "purchase_order purchase_order_line goods_receipt goods_receipt_line inbound_transport supplier_terms "
    "stock_day stock_movement stockout",
    "delivery": "carrier vehicle shipment route route_stop delivery_event dispatch_case dc_invoice",
    "field": "tourplan visit_plan visit visit_observation competitor_sighting pos_material pos_asset pos_loan "
    "pos_handout rep_prompt",
    "marketing": "campaign campaign_cost promotion promotion_eligibility promotion_exclusion promotion_result "
    "seasonal_exclusion",
    "shops": "shop_receipt shop_receipt_line menu_item",
    "documents": "document document_golden complaint return_request_hdr return_request_line return_receipt "
    "credit_note refund_payment replacement_shipment",
}

# contracts §5: the provenance columns, and where `source` says generated or pipeline.
PROVENANCE = {
    ("promotion", "document_id"),
    ("standard_price_list", "document_id"),
    ("channel_price_list", "document_id"),
    ("sku", "launch_document_id"),
    ("sku", "delist_document_id"),
    ("campaign", "document_id"),
    ("prompt_rule", "document_id"),
    ("lifecycle_rule", "document_id"),
    ("complaint", "document_id"),
    ("return_request_hdr", "document_id"),
}

AUTOMATION_FAMILY = {
    "return_request_hdr",
    "return_request_line",
    "return_receipt",
    "credit_note",
    "refund_payment",
    "replacement_shipment",
}


@pytest.fixture(scope="module")
def inventory() -> InventoryConfig:
    return InventoryConfig.load()


def test_every_table_of_the_design_in_its_group(inventory: InventoryConfig) -> None:
    by_group: dict[str, set[str]] = {}
    for table in inventory.tables:
        by_group.setdefault(table.group, set()).add(table.name)
    assert by_group == {group: set(names.split()) for group, names in TABLES.items()}
    assert len(inventory.tables) == 98


def test_the_generators_own_tables_are_not_here(inventory: InventoryConfig) -> None:
    names = {table.name for table in inventory.tables} | {view.name for view in inventory.views}
    assert not names & {"_sim_meta", "_sim_checkpoint"}


def test_the_four_views_and_the_familys_own(inventory: InventoryConfig) -> None:
    materialized = {view.name: view.sql_file for view in inventory.views if view.kind == "materialized_view"}
    assert materialized == {
        name: f"views/{name}.sql" for name in ("sales_pl", "trade_spend", "route_current", "receivable")
    }
    assert [(view.name, view.kind) for view in inventory.views if view.kind == "view"] == [("return_request", "view")]


def test_the_graph_is_acyclic_and_the_load_order_respects_it(inventory: InventoryConfig) -> None:
    graph = {table.name: {fk.table for fk in table.fks if fk.table != table.name} for table in inventory.tables}
    order = list(TopologicalSorter(graph).static_order())  # raises CycleError on a cycle
    assert len(order) == len(inventory.tables)
    load_order = {table.name: table.load_order for table in inventory.tables}
    for table, targets in graph.items():
        assert all(load_order[target] < load_order[table] for target in targets), table
    assert [table.load_order for table in inventory.tables] == sorted(load_order.values())


def test_every_ref_resolves_to_a_primary_key(inventory: InventoryConfig) -> None:
    by_name = {table.name: table for table in inventory.tables}
    for table in inventory.tables:
        for fk in table.fks:
            assert by_name[fk.table].pk == (fk.target,), f"{table.name}.{fk.column}"


def test_the_types_of_contracts_1(inventory: InventoryConfig) -> None:
    jsonb = set()
    for table in inventory.tables:
        for column in table.columns:
            is_id = column.name == f"{table.name}_id" and table.pk == (column.name,)
            if is_id or (column.ref is not None and column.ref != "calendar.day"):
                assert column.type == "bigint", f"{table.name}.{column.name}"
            if column.type == "jsonb":
                jsonb.add(f"{table.name}.{column.name}")
            if table.name in AUTOMATION_FAMILY:
                continue  # its money and quantities as the family has them
            if column.name in {"amount", "cost"} or column.name.endswith(("_amount", "_cost", "_price_case")):
                assert column.type == "numeric(14,2)", f"{table.name}.{column.name}"
            if column.name in {"cases", "qty_cases", "qty_pieces"} or column.name.endswith("_cases"):
                assert column.type in {"numeric(12,3)", "integer"}, f"{table.name}.{column.name}"
    assert jsonb == {"document_golden.payload", "document_golden.mentions", "visit_observation.extra"}


def test_every_table_with_an_id_has_its_business_key(inventory: InventoryConfig) -> None:
    """contracts §2: an id is a hash of the business key; a composite primary key is the business key itself."""
    for table in inventory.tables:
        surrogate = len(table.pk) == 1 and table.name not in {"calendar", "document_golden"}
        assert (table.id_key is not None) is surrogate, table.name
    # a transfer's two legs (one per DC), and two write-offs without a ref, hash apart
    assert {"dc", "day"} <= set(inventory.table("stock_movement").id_key or ())


def test_the_provenance_columns_of_contracts_5(inventory: InventoryConfig) -> None:
    for name, column in PROVENANCE:
        found = inventory.table(name).column(column)
        assert (found.ref, found.nullable) == ("document.document_id", True), f"{name}.{column}"
    for name in ("complaint", "return_request_hdr"):
        assert inventory.table(name).column("source").coded == "source"
    assert set(inventory.codes["source"]) == {"generated", "pipeline"}
    sender = inventory.table("document").column("sender_customer_id")
    assert sender.ref == "customer.customer_id"


def test_the_discount_list_of_contracts_4(inventory: InventoryConfig) -> None:
    discount = inventory.table("sales_order_line_discount")
    assert discount.pk == ("sales_order_line_id", "seq")
    assert set(inventory.codes[discount.column("kind").coded or ""]) == {
        "volume",
        "cross_sell",
        "agreed",
        "individual",
        "promotion",
        "free_goods",
        "retro",
    }
    assert set(inventory.codes["budget"]) == {"sales", "marketing"}
    for column in ("code", "promotion_id", "amount_case", "budget"):
        discount.column(column)
    inventory.table("sales_order_line").column("discount_total")


def test_the_registry_of_contracts_5(inventory: InventoryConfig) -> None:
    assert set(inventory.codes["document_family"]) == {
        "return_request",
        "complaint",
        "promotion_sheet",
        "price_memo",
        "product_announcement",
        "policy",
        "campaign_brief",
        "competitor_note",
    }
    assert set(inventory.codes["document_channel"]) == {"email", "web_form", "letter_scan", "internal"}
    assert set(inventory.codes["document_format"]) == {"eml", "md", "docx", "pdf", "json", "xlsx"}


def test_the_codes_match_the_world_and_the_classifier(inventory: InventoryConfig, ref: Path) -> None:
    world = WorldConfig.load()
    channels = {field.name for field in dataclasses.fields(world.channels)} - {"listings"}
    assert set(inventory.codes["channel"]) == channels - {"special_retail"}
    assert set(inventory.codes["sub_channel"]) == {"special_retail"}
    assert set(inventory.codes["store_class"]) == {*STORE_CLASSES, "horeca"}
    assert tuple(inventory.codes["street_type"]) == STREET_TYPES
    assert set(inventory.codes["chain_kind"]) == {chain.kind for chain in CHAINS}
    assert set(inventory.codes["phase"]) == {*world.policies.lifecycle.phases, "lost"}
    catalogue = json.loads((ref / "schemas" / "catalogue.schema.json").read_text(encoding="utf-8"))
    assert list(inventory.codes["pack_format"]) == catalogue["properties"]["pack_format"]["enum"]
    assert set(inventory.codes["mechanic"]) == set(world.promotions.mechanics)


def test_every_label_reads_as_czech(inventory: InventoryConfig) -> None:
    labels = [label for values in inventory.codes.values() for label in values.values()]
    assert_czech_all(labels, what="inventory codes")
    columns = [column.label_cs for table in inventory.tables for column in table.columns if column.label_cs]
    assert_czech_all(columns, what="inventory label_cs")


def test_every_code_list_is_used(inventory: InventoryConfig) -> None:
    used = {column.coded for table in inventory.tables for column in table.columns}
    assert set(inventory.codes) <= used


# ── a broken inventory fails naming the place ───────────────────────────────


def _broken(tmp_path: Path, change: Any) -> Path:
    data = yaml.safe_load(INVENTORY_FILE.read_text(encoding="utf-8"))
    change({table["name"]: table for table in data["tables"]}, data)
    path = tmp_path / "inventory.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_a_ref_loading_later_fails(tmp_path: Path) -> None:
    def later(tables: dict[str, Any], data: Any) -> None:
        tables["kraj"]["load_order"] = 35

    with pytest.raises(ConfigError, match=r"inventory\.yaml: okres\.kraj_id: ref kraj loads at 35, not before 30"):
        InventoryConfig.load(_broken(tmp_path, later))


def test_a_ref_to_nothing_fails(tmp_path: Path) -> None:
    def nowhere(tables: dict[str, Any], data: Any) -> None:
        tables["okres"]["columns"][3]["ref"] = "county.county_id"

    with pytest.raises(ConfigError, match=r"okres\.kraj_id: ref county\.county_id names no table"):
        InventoryConfig.load(_broken(tmp_path, nowhere))


def test_an_unknown_code_list_and_a_missing_key_column_fail_together(tmp_path: Path) -> None:
    def two(tables: dict[str, Any], data: Any) -> None:
        tables["site"]["columns"][2]["coded"] = "site_kinds"
        tables["store"]["indexes"].append(["chain"])

    with pytest.raises(ConfigError) as error:
        InventoryConfig.load(_broken(tmp_path, two))
    assert str(error.value).splitlines() == [
        "inventory.yaml: site.kind: coded 'site_kinds' is no code list",
        "inventory.yaml: store: index names no column 'chain'",
    ]


def test_two_missing_keys_of_a_column_are_both_named(tmp_path: Path) -> None:
    def bare(tables: dict[str, Any], data: Any) -> None:
        tables["kraj"]["columns"][1] = {"nullable": True}

    with pytest.raises(ConfigError) as error:
        InventoryConfig.load(_broken(tmp_path, bare))
    assert str(error.value).splitlines() == [
        "inventory.yaml: tables.kraj.columns.1.name: missing",
        "inventory.yaml: tables.kraj.columns.1.type: missing",
    ]


def test_a_missing_view_file_and_a_long_name_fail_together(tmp_path: Path) -> None:
    """Both would otherwise crash `just ddl` with a traceback: they arrive as problem lines instead."""

    def two(tables: dict[str, Any], data: Any) -> None:
        tables["promotion"]["indexes"] = [["target_product_line_id", "target_uplift_pct", "target_period_weeks"]]
        data["views"][0]["sql_file"] = "views/nowhere.sql"

    with pytest.raises(ConfigError) as error:
        InventoryConfig.load(_broken(tmp_path, two))
    assert str(error.value).splitlines() == [
        "inventory.yaml: promotion: promotion_target_product_line_id_target_uplift_pct_target_period_weeks_idx is over"
        " 63 bytes",
        "inventory.yaml: sales_pl: no data/ddl/views/nowhere.sql",
    ]


def test_a_schema_problem_names_the_table(tmp_path: Path) -> None:
    def typo(tables: dict[str, Any], data: Any) -> None:
        tables["customer"]["columns"][1]["type"] = "varchar"

    with pytest.raises(ConfigError, match=r"inventory\.yaml: tables\.customer\.columns\.code\.type: 'varchar'"):
        InventoryConfig.load(_broken(tmp_path, typo))
