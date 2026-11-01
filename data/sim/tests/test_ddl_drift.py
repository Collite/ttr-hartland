"""`just ddl-drift` (scripts/ddl_drift.py; contracts §1, M0 S0.1 T4): schema.sql ⇄ the db layer.

The comparison on small hand-written sides; the schema side on the real schema.sql; and, where a built ttr-core is at
hand (TATRMAN_PACKAGES, or ../tatrman), the list's own case end to end: a column added to the inventory and rendered
by `just ddl` fails with the table and the column named, and the `.ttrm` edit makes it clean.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from hartland_sim.config import InventoryConfig
from hartland_sim.io.ddl import render

REPO = Path(__file__).resolve().parents[3]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ddl_drift", REPO / "scripts/ddl_drift.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ddl_drift"] = module
    spec.loader.exec_module(module)
    return module


dd = _load()

SQL = """
CREATE SCHEMA hartland;
CREATE TABLE hartland.kraj (
    kraj_id bigint NOT NULL,
    name text NOT NULL,
    CONSTRAINT kraj_pkey PRIMARY KEY (kraj_id)
);
CREATE TABLE hartland.okres (
    okres_id bigint NOT NULL,
    kraj_id bigint NOT NULL,
    affluence numeric(4,3),
    at timestamptz NOT NULL,
    extra jsonb,
    CONSTRAINT okres_pkey PRIMARY KEY (okres_id),
    CONSTRAINT okres_kraj_id_fkey FOREIGN KEY (kraj_id) REFERENCES hartland.kraj (kraj_id) DEFERRABLE INITIALLY DEFERRED
);
CREATE MATERIALIZED VIEW hartland.big AS
SELECT o.okres_id, k.name AS kraj_name FROM okres o JOIN kraj k ON k.kraj_id = o.kraj_id
WITH NO DATA;
CREATE VIEW hartland.small AS
SELECT okres_id FROM okres;
"""


def _layer() -> dict[str, object]:
    """The db layer that matches SQL exactly (scripts/db-layer.mjs's shape)."""

    def col(name: str, type_: str, optional: bool = False, length: int | None = None, precision: int | None = None):  # type: ignore[no-untyped-def]
        return {"name": name, "type": type_, "length": length, "precision": precision, "optional": optional}

    return {
        "tables": {
            "kraj": {"file": "a.ttrm", "kind": "table", "primaryKey": ["kraj_id"],
                     "columns": [col("kraj_id", "bigint"), col("name", "text")]},
            "okres": {"file": "a.ttrm", "kind": "table", "primaryKey": ["okres_id"],
                      "columns": [col("okres_id", "bigint"), col("kraj_id", "bigint"),
                                  col("affluence", "decimal", True, 4, 3), col("at", "datetime"),
                                  col("extra", "text", True)]},
            "big": {"file": "views.ttrm", "kind": "table", "primaryKey": [],
                    "columns": [col("okres_id", "bigint"), col("kraj_name", "text")]},
            "small": {"file": "views.ttrm", "kind": "view", "primaryKey": [], "columns": [col("okres_id", "bigint")]},
        },
        "fks": {"okres_kraj_id_fkey": {"file": "fks.ttrm", "from": ["okres.kraj_id"], "to": ["kraj.kraj_id"]}},
        "parseErrors": [],
    }  # fmt: skip


def _diffs(layer: dict[str, object], sql: str = SQL, gaps: dict[tuple[str, str], str] | None = None) -> list[str]:
    diffs, _ = dd.compare(dd.schema_from_sql(sql), dd.layer_from_json(layer), {} if gaps is None else gaps)
    return list(diffs)


def _tables(layer: dict[str, object]) -> dict[str, dict[str, object]]:
    return layer["tables"]  # type: ignore[return-value]


def test_a_matching_layer_is_clean() -> None:
    assert _diffs(_layer()) == []


def test_a_column_only_in_schema_sql_is_named() -> None:
    sql = SQL.replace("    extra jsonb,\n", "    extra jsonb,\n    note text,\n")
    assert _diffs(_layer(), sql) == ["okres.note: in schema.sql (text), not in model/db/a.ttrm"]


def test_type_precision_and_nullability_are_compared() -> None:
    layer = _layer()
    okres = _tables(layer)["okres"]["columns"]
    okres[2].update(length=5, precision=2)  # type: ignore[index]
    okres[3].update(type="date")  # type: ignore[index]
    okres[0].update(optional=True)  # type: ignore[index]
    assert _diffs(layer) == [
        "okres.okres_id: NOT NULL in schema.sql, nullable in model/db",
        "okres.affluence: type decimal(4,3) in schema.sql, decimal(5,2) in model/db",
        "okres.at: type datetime in schema.sql, date in model/db",
    ]


def test_missing_and_extra_relations_and_the_primary_key() -> None:
    layer = _layer()
    del _tables(layer)["kraj"]
    _tables(layer)["ghost"] = {"file": "b.ttrm", "kind": "table", "primaryKey": [], "columns": []}
    _tables(layer)["okres"]["primaryKey"] = ["okres_id", "kraj_id"]
    assert _diffs(layer) == [
        "kraj: in schema.sql (table), not in model/db",
        "ghost: in model/db/b.ttrm, not in schema.sql",
        "okres: primary key ['okres_id'] in schema.sql, ['okres_id', 'kraj_id'] in model/db",
    ]


def test_foreign_keys_by_name_and_columns() -> None:
    layer = _layer()
    layer["fks"] = {"okres_kraj_fk": {"file": "fks.ttrm", "from": ["okres.kraj_id"], "to": ["kraj.name"]}}
    assert _diffs(layer) == [
        "okres: foreign key okres_kraj_id_fkey (okres.kraj_id) not in fks.ttrm",
        "foreign key okres_kraj_fk: in fks.ttrm, not in schema.sql",
    ]
    layer["fks"] = {"okres_kraj_id_fkey": {"file": "fks.ttrm", "from": ["okres.kraj_id"], "to": ["kraj.name"]}}
    assert _diffs(layer) == [
        "foreign key okres_kraj_id_fkey: schema.sql okres.kraj_id → kraj.kraj_id; fks.ttrm okres.kraj_id → kraj.name"
    ]


def test_views_by_kind_and_column_names() -> None:
    layer = _layer()
    _tables(layer)["big"]["kind"] = "view"
    _tables(layer)["small"]["columns"] = [{"name": "okres", "type": "bigint", "optional": False}]
    assert _diffs(layer) == [
        "big: a materialized_view in schema.sql, so a `def table` in model/db — it is a `def view`",
        "small.okres_id: a column of the view in schema.sql, not in model/db/views.ttrm",
        "small.okres: in model/db/views.ttrm, not a column of the view in schema.sql",
    ]


def test_column_order_is_compared() -> None:
    layer = _layer()
    cols = _tables(layer)["kraj"]["columns"]
    _tables(layer)["kraj"]["columns"] = list(reversed(cols))  # type: ignore[call-overload]
    assert _diffs(layer) == [
        "kraj: column order differs — schema.sql ['kraj_id', 'name'], model/db ['name', 'kraj_id']"
    ]


def test_a_known_gap_is_a_note_until_it_closes_or_changes() -> None:
    layer = _layer()
    _tables(layer)["okres"]["columns"] = _tables(layer)["okres"]["columns"][:4]  # type: ignore[index]
    gaps = {("okres", "extra"): "a keyword"}
    diffs, notes = dd.compare(dd.schema_from_sql(SQL), dd.layer_from_json(layer), gaps)
    assert (diffs, notes) == ([], ["okres.extra: not in model/db — known gap: a keyword"])
    assert _diffs(_layer(), gaps=gaps) == [
        "okres.extra: a known gap, but model/db declares it — remove it from KNOWN_GAPS"
    ]
    assert _diffs(_layer(), gaps={("okres", "gone"): "x"}) == [
        "okres.gone: a known gap, but schema.sql has no such column — remove it from KNOWN_GAPS"
    ]


def test_parse_errors_and_select_star_fail() -> None:
    layer = _layer()
    layer["parseErrors"] = ["a.ttrm:3: mismatched input"]
    sql = SQL.replace("SELECT okres_id FROM okres;", "SELECT * FROM okres;")
    assert _diffs(layer, sql)[:2] == [
        "view small: `SELECT *` — a view's columns must be named",
        "model/db/a.ttrm:3: mismatched input",
    ]


def test_the_real_schema_sql_reads_whole() -> None:
    schema = dd.schema_from_sql((REPO / "data/ddl/schema.sql").read_text(encoding="utf-8"))
    kinds = [r.kind for r in schema.relations.values()]
    assert (kinds.count("table"), kinds.count("materialized_view"), kinds.count("view")) == (98, 4, 1)
    assert len(schema.fks) == 205
    assert schema.errors == []
    okres = schema.relations["okres"]
    assert [c.type_text() for c in okres.columns] == [
        "bigint",
        "text",
        "text",
        "bigint",
        "bigint",
        "decimal(4,3)",
        "int",
    ]
    assert [c.optional for c in okres.columns] == [False, False, False, False, True, False, False]


def test_every_known_gap_is_a_real_column() -> None:
    schema = dd.schema_from_sql((REPO / "data/ddl/schema.sql").read_text(encoding="utf-8"))
    for table, column in dd.KNOWN_GAPS:
        assert column in {c.name for c in schema.relations[table].columns}


def _toolchain() -> Path | None:
    packages = Path(os.environ.get("TATRMAN_PACKAGES", REPO.parent / "tatrman/packages")).resolve()
    if shutil.which("node") and (packages / "parser/dist/index.js").exists():
        return packages
    return None


@pytest.mark.skipif(_toolchain() is None, reason="no built ttr-core (TATRMAN_PACKAGES or ../tatrman) and node")
def test_a_column_added_to_the_inventory_fails_until_the_ttrm_has_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TATRMAN_PACKAGES", str(_toolchain()))
    raw = yaml.safe_load((REPO / "data/inventory.yaml").read_text(encoding="utf-8"))
    okres = next(t for t in raw["tables"] if t["name"] == "okres")
    okres["columns"].append({"name": "district_note", "type": "text", "nullable": True})
    (tmp_path / "inventory.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (tmp_path / "schema.sql").write_text(render(InventoryConfig.load(tmp_path / "inventory.yaml"))["schema.sql"])
    db = tmp_path / "db"
    shutil.copytree(REPO / "model/db", db)
    args = ["--schema", str(tmp_path / "schema.sql"), "--model-db", str(db)]

    assert dd.main(args) == 1
    assert (
        "DRIFT      okres.district_note: in schema.sql (text), not in model/db/reference.ttrm"
        in capsys.readouterr().out
    )

    reference = (db / "reference.ttrm").read_text(encoding="utf-8")
    population = "        def column population      { type: int }"
    note = "        def column district_note   { type: text, optional: true }"
    edited = reference.replace(f"{population}\n", f"{population},\n{note}\n")
    assert edited != reference
    (db / "reference.ttrm").write_text(edited, encoding="utf-8")
    assert dd.main(args) == 0
    assert "— clean" in capsys.readouterr().out
