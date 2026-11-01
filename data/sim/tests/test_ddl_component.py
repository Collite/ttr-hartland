"""G0 S0.4 T5, the component tier: schema.sql applies on a fresh Postgres 16 and is what the inventory says."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date

import psycopg
import pytest

from hartland_sim.config import InventoryConfig
from hartland_sim.io.ddl import DOCWH, LICENCE_NOTICE, READONLY

pytestmark = pytest.mark.component

Put = Callable[..., None]  # conftest's `put`

# information_schema's spelling of the inventory's types.
_TYPES = {"timestamptz": "timestamp with time zone", "time": "time without time zone"}


@pytest.fixture(scope="module")
def inventory() -> InventoryConfig:
    return InventoryConfig.load()


@pytest.fixture(scope="module")
def db(hartland_db: psycopg.Connection) -> psycopg.Connection:
    return hartland_db


def _type(row: tuple[str, int | None, int | None]) -> str:
    data_type, precision, scale = row
    return f"numeric({precision},{scale})" if data_type == "numeric" else data_type


def test_every_inventory_table_with_its_columns(db: psycopg.Connection, inventory: InventoryConfig) -> None:
    rows = db.execute(
        "SELECT table_name, column_name, data_type, numeric_precision, numeric_scale, is_nullable"
        " FROM information_schema.columns c JOIN information_schema.tables t USING (table_schema, table_name)"
        " WHERE table_schema = 'hartland' AND table_type = 'BASE TABLE'"
        " ORDER BY table_name, ordinal_position"
    ).fetchall()
    actual: dict[str, list[tuple[str, str, bool]]] = {}
    for table, column, data_type, precision, scale, nullable in rows:
        actual.setdefault(table, []).append((column, _type((data_type, precision, scale)), nullable == "YES"))
    expected = {
        table.name: [(c.name, _TYPES.get(c.type, c.type), c.nullable) for c in table.columns]
        for table in inventory.tables
    }
    assert actual == expected


def test_the_views_refresh_on_empty_tables(db: psycopg.Connection, inventory: InventoryConfig) -> None:
    for view in inventory.views:
        if view.kind == "materialized_view":
            db.execute(f"REFRESH MATERIALIZED VIEW hartland.{view.name}")
        count = db.execute(f"SELECT count(*) FROM hartland.{view.name}").fetchone()
        assert count == (0,), view.name


def test_sales_pl_has_the_columns_of_contracts_4(db: psycopg.Connection) -> None:
    rows = db.execute(
        "SELECT a.attname FROM pg_attribute a WHERE a.attrelid = 'hartland.sales_pl'::regclass AND a.attnum > 0"
        " ORDER BY a.attnum"
    ).fetchall()
    assert [name for (name,) in rows] == (
        "line_id order_id ordered_on delivered_on channel sub_channel customer_id delivery_point_id store_id sku_id"
        " product_line_id brand_id rep_employee_id area_id okres_code dc_site_id cases pieces list_price_case"
        " gross_revenue discount_volume discount_cross_sell discount_agreed discount_individual discount_promotion"
        " free_goods_value net_revenue delivery_cost warehouse_cost customer_margin cogs marketing_cost"
        " product_margin vat_rate vat"
    ).split()


def test_the_licence_comment_survives_the_apply(db: psycopg.Connection) -> None:
    assert db.execute("SELECT obj_description('hartland'::regnamespace)").fetchone() == (LICENCE_NOTICE,)


def test_every_index_is_there(db: psycopg.Connection, inventory: InventoryConfig) -> None:
    names = {name for (name,) in db.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'hartland'")}
    expected = {"_".join([table.name, *key, "idx"]) for table in inventory.tables for key in table.indexes}
    assert expected <= names


def test_the_docwh_loader_reads_no_answer(db: psycopg.Connection) -> None:
    """contracts §5: the sender, the answer tables and views, and the provenance `document_id`s are refused."""
    refused = [
        "SELECT sender_customer_id FROM hartland.document",
        "SELECT * FROM hartland.document",
        "SELECT * FROM hartland.document_golden",
        "SELECT * FROM hartland.complaint",
        "SELECT * FROM hartland.return_request_hdr",
        "SELECT * FROM hartland.return_request",
        "SELECT document_id FROM hartland.promotion",
        "SELECT launch_document_id FROM hartland.sku",
    ]
    db.execute(f"SET ROLE {DOCWH}")
    try:
        db.execute("SELECT document_id, code, family, path FROM hartland.document")
        db.execute("SELECT count(*) FROM hartland.customer")
        db.execute("SELECT code, target_sku_id FROM hartland.promotion")
        db.execute("SELECT count(*) FROM hartland.sales_pl")
        for statement in refused:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                db.execute(statement)
    finally:
        db.execute("RESET ROLE")
    db.execute(f"SET ROLE {READONLY}")
    try:
        db.execute("SELECT sender_customer_id FROM hartland.document")
    finally:
        db.execute("RESET ROLE")


def test_sales_pl_credits_a_kam_channel_sale_to_the_kam(db: psycopg.Connection, put: Put) -> None:
    """At one store with an area rep: a big-retail sale is the customer's KAM's, a small-retail sale the rep's, and an
    own-shop sale (no KAM) the taker's."""
    year = {"valid_from": date(2024, 1, 1), "valid_to": date(2026, 12, 31)}
    with db.transaction(force_rollback=True):
        put(db, "okres", okres_id=1)
        put(db, "store", store_id=2, okres_id=1, area_id=3)
        put(db, "assignment", assignment_id=4, employee_id=41, kind="rep_area", area_id=3, **year)
        put(db, "assignment", assignment_id=5, employee_id=51, kind="kam_customer", customer_id=10, **year)
        put(db, "product_line", product_line_id=6)
        put(db, "product", product_id=7, product_line_id=6)
        put(db, "sku", sku_id=8, product_id=7)
        for n, channel in ((10, "big_retail"), (11, "small_retail"), (12, "own_shops")):
            put(db, "customer", customer_id=n, channel=channel)
            put(db, "delivery_point", delivery_point_id=n, customer_id=n, store_id=2, okres_id=1)
            put(
                db,
                "sales_order",
                sales_order_id=n,
                customer_id=n,
                delivery_point_id=n,
                channel=channel,
                ordered_on=date(2025, 3, 3),
                taken_by_employee_id=99,
            )
            put(db, "sales_order_line", sales_order_line_id=n, sales_order_id=n, sku_id=8)
        db.execute("REFRESH MATERIALIZED VIEW hartland.sales_pl")
        rows = db.execute("SELECT order_id, rep_employee_id FROM hartland.sales_pl ORDER BY order_id").fetchall()
    assert rows == [(10, 51), (11, 41), (12, 99)]


def test_a_deferred_key_lets_a_load_copy_children_first(db: psycopg.Connection) -> None:
    """The foreign keys are checked at commit, so `load-world` may copy in any order inside one transaction."""
    with db.transaction(force_rollback=True):
        db.execute("INSERT INTO hartland.kraj VALUES (2, 'CZ020', 'Středočeský kraj')")
        db.execute("INSERT INTO hartland.okres VALUES (1, 'CZ0201', 'Benešov', 2, NULL, 1.0, 100000)")
    with pytest.raises(psycopg.errors.ForeignKeyViolation), db.transaction():
        db.execute("INSERT INTO hartland.okres VALUES (1, 'CZ0201', 'Benešov', 99, NULL, 1.0, 100000)")


def test_a_coded_column_refuses_another_value(db: psycopg.Connection) -> None:
    with pytest.raises(psycopg.errors.CheckViolation), db.transaction():
        db.execute("INSERT INTO hartland.chain VALUES (1, 'X', 'X', 'hypermarket', false)")


def test_no_name_is_a_reserved_word(db: psycopg.Connection, inventory: InventoryConfig) -> None:
    """The render quotes nothing, so no table or column may need quoting."""
    reserved = {word for (word,) in db.execute("SELECT word FROM pg_get_keywords() WHERE catcode IN ('R', 'T')")}
    names = {table.name for table in inventory.tables} | {view.name for view in inventory.views}
    names |= {column.name for table in inventory.tables for column in table.columns}
    assert not names & reserved
    assert all(re.fullmatch(r"[a-z_][a-z0-9_]*", name) for name in names)
