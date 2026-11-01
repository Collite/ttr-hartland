"""G0 S0.4 T5 — `just ddl`: the inventory rendered to schema.sql, indexes.sql and views/grants.sql (contracts §1)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from hartland_sim.checks.czech import assert_czech
from hartland_sim.cli import main
from hartland_sim.config import DDL_DIR, InventoryConfig
from hartland_sim.io.ddl import GENERATED, LICENCE_NOTICE, _wrapped, drift, render, write


@pytest.fixture(scope="module")
def inventory() -> InventoryConfig:
    return InventoryConfig.load()


@pytest.fixture(scope="module")
def schema_sql(inventory: InventoryConfig) -> str:
    return render(inventory)["schema.sql"]


def test_the_committed_files_are_the_render(inventory: InventoryConfig) -> None:
    assert drift(inventory) == []


def test_the_render_is_stable(inventory: InventoryConfig) -> None:
    assert render(inventory) == render(InventoryConfig.load())


def test_the_tables_in_load_order(inventory: InventoryConfig, schema_sql: str) -> None:
    created = re.findall(r"^CREATE TABLE hartland\.(\w+) \($", schema_sql, re.M)
    assert created == [table.name for table in inventory.tables]


def test_every_foreign_key_is_deferred(inventory: InventoryConfig, schema_sql: str) -> None:
    fks = re.findall(r"FOREIGN KEY .*$", schema_sql, re.M)
    assert len(fks) == sum(len(table.fks) for table in inventory.tables) == 205
    assert all(fk.rstrip(",").endswith("DEFERRABLE INITIALLY DEFERRED") for fk in fks)


def test_a_coded_column_checks_its_values(schema_sql: str) -> None:
    assert (
        "    channel text NOT NULL CHECK (channel IN ('big_retail', 'small_retail', 'gas_ka', 'own_shops', "
        in schema_sql
    )
    assert "    sub_channel text CHECK (sub_channel IN ('special_retail'))," in schema_sql


def test_the_views_follow_the_tables(schema_sql: str) -> None:
    for name in ("sales_pl", "trade_spend", "route_current", "receivable"):
        assert re.search(rf"^CREATE MATERIALIZED VIEW hartland\.{name} AS\n(.*\n)*?WITH NO DATA;$", schema_sql, re.M)
    assert "CREATE VIEW hartland.return_request AS" in schema_sql
    assert schema_sql.index("CREATE MATERIALIZED VIEW") > schema_sql.rindex("CREATE TABLE")


def test_the_schema_carries_the_licence_in_czech(schema_sql: str) -> None:
    """NH-D13, LICENSE-DATA.md §2: every dump carries the notice."""
    assert f"COMMENT ON SCHEMA hartland IS '{LICENCE_NOTICE}';" in schema_sql
    assert "https://opendatacommons.org/licenses/odbl/1-0/" in LICENCE_NOTICE
    assert "https://opendatacommons.org/licenses/dbcl/1-0/" in LICENCE_NOTICE
    assert "Obsahuje údaje z OpenStreetMap, © přispěvatelé OpenStreetMap" in LICENCE_NOTICE
    names = ["Open Database License", "Database Contents License", "New Hartland", "OpenStreetMap", "ODbL"]
    assert_czech(LICENCE_NOTICE, what="the schema comment", allow=names)


# The answers the DocWH is scored against (contracts §5): the golden extractions, the complaints, the return family.
ANSWERS = {
    "document_golden",
    "complaint",
    "return_request_hdr",
    "return_request_line",
    "return_receipt",
    "credit_note",
    "refund_payment",
    "replacement_shipment",
    "return_request",
}
# The tables the loader reads column by column: the sender, and the provenance `document_id`s.
BY_COLUMN = {
    "document",
    "sku",
    "standard_price_list",
    "channel_price_list",
    "prompt_rule",
    "lifecycle_rule",
    "campaign",
    "promotion",
}


def _docwh_grants(grants: str) -> tuple[set[str], dict[str, list[str]]]:
    """What hartland_docwh is granted: the tables and views whole, and the column lists by table."""
    whole = set(re.findall(r"^GRANT SELECT ON hartland\.(\w+) TO hartland_docwh;$", grants, re.M))
    pattern = r"^GRANT SELECT \((.*)\) ON hartland\.(\w+) TO hartland_docwh;$"
    by_column = {table: columns.split(", ") for columns, table in re.findall(pattern, grants, re.M)}
    return whole, by_column


def test_the_docwh_loader_reads_no_answer(inventory: InventoryConfig) -> None:
    grants = render(inventory)["views/grants.sql"]
    whole, by_column = _docwh_grants(grants)
    hidden = {table.name for table in inventory.tables if not table.docwh}
    assert hidden | {view.name for view in inventory.views if not view.docwh} == ANSWERS
    assert not (whole | set(by_column)) & ANSWERS
    assert set(by_column) == BY_COLUMN
    everything = {table.name for table in inventory.tables} | {view.name for view in inventory.views}
    assert whole == everything - ANSWERS - BY_COLUMN
    for table, columns in by_column.items():
        refs = {column.name: column.ref for column in inventory.table(table).columns}
        assert [name for name in columns if refs[name] == "document.document_id"] == [], table
        assert set(columns) == set(refs) - {name for name, ref in refs.items() if ref == "document.document_id"} - {
            "sender_customer_id"
        }, table
    assert "sender_customer_id" not in by_column["document"]
    assert "GRANT SELECT ON ALL TABLES IN SCHEMA hartland TO hartland_readonly;" in grants


def test_a_view_the_docwh_loader_reads_reads_no_answer(inventory: InventoryConfig) -> None:
    """A view runs with its owner's rights, so a view that read an answer would hand it on."""
    hidden = {table.name for table in inventory.tables if not table.docwh}
    hidden |= {column.name for table in inventory.tables for column in table.columns if column.hidden_from_docwh}
    for view in inventory.views:
        if view.docwh:
            lines = (DDL_DIR / view.sql_file).read_text(encoding="utf-8").splitlines()
            code = [re.sub(r"'[^']*'", "", line.split("--")[0]) for line in lines]  # no comments, no literals
            words = {word for line in code for word in re.findall(r"\w+", line)}
            assert not words & hidden, view.name


def test_the_grant_header_wraps_the_names_it_is_given() -> None:
    """The list ends at its last place, not at the first name equal to the last; an empty list is still a list."""
    assert _wrapped(["a", "b", "a"], "--   ", width=12) == ["--   a, b,", "--   a;"]
    assert _wrapped([], "--   ") == ["--   ;"]


def test_every_index_of_the_inventory(inventory: InventoryConfig) -> None:
    indexes = render(inventory)["indexes.sql"]
    assert len(re.findall(r"^CREATE INDEX ", indexes, re.M)) == sum(len(table.indexes) for table in inventory.tables)


# ── the verb ─────────────────────────────────────────────────────────────────


@pytest.fixture
def ddl_copy(tmp_path: Path) -> Path:
    target = tmp_path / "ddl"
    shutil.copytree(DDL_DIR, target)
    return target


def test_check_passes_on_the_committed_files(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ddl", "--check"]) == 0
    assert "data/ddl/ matches data/inventory.yaml (98 tables)" in capsys.readouterr().out


def test_check_fails_naming_each_file_that_drifted(ddl_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    schema = ddl_copy / "schema.sql"
    schema.write_text(schema.read_text(encoding="utf-8").replace("code text NOT NULL", "code text", 1))
    (ddl_copy / "indexes.sql").unlink()
    assert main(["ddl", "--check", "--ddl-dir", str(ddl_copy)]) == 1
    assert capsys.readouterr().err.splitlines() == [
        "data/ddl/schema.sql: differs from the render of data/inventory.yaml",
        "data/ddl/indexes.sql: missing",
        "hartland-sim ddl --check: run `just ddl` and commit data/ddl/",
    ]


def test_a_view_file_missing_from_the_ddl_dir_is_named(ddl_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The inventory checks the repo's own view files; `--ddl-dir` names another folder, checked when rendered."""
    (ddl_copy / "views" / "trade_spend.sql").unlink()
    (ddl_copy / "views" / "receivable.sql").unlink()
    schema = (ddl_copy / "schema.sql").read_text(encoding="utf-8")
    for check in (["--check"], []):
        assert main(["ddl", *check, "--ddl-dir", str(ddl_copy)]) == 1
        assert capsys.readouterr().err.splitlines() == [
            f"trade_spend: no {ddl_copy}/views/trade_spend.sql",
            f"receivable: no {ddl_copy}/views/receivable.sql",
        ]
    assert (ddl_copy / "schema.sql").read_text(encoding="utf-8") == schema  # nothing written


def test_ddl_writes_what_changed(ddl_copy: Path, inventory: InventoryConfig) -> None:
    for name in GENERATED:
        (ddl_copy / name).unlink()
    assert write(inventory, ddl_copy) == list(GENERATED)
    assert write(inventory, ddl_copy) == []
    assert drift(inventory, ddl_copy) == []


def test_a_view_edit_is_drift_too(ddl_copy: Path, inventory: InventoryConfig) -> None:
    """The view files are sources the render reads: editing one without re-rendering is caught."""
    view = ddl_copy / "views" / "trade_spend.sql"
    view.write_text(view.read_text(encoding="utf-8") + "-- edited\n", encoding="utf-8")
    assert drift(inventory, ddl_copy) == ["data/ddl/schema.sql: differs from the render of data/inventory.yaml"]
