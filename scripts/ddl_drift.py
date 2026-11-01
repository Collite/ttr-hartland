"""`just ddl-drift` (contracts §1, M0 S0.1 T4): data/ddl/schema.sql ⇄ the db layer model/db/*.ttrm.

    uv run python scripts/ddl_drift.py [--schema PATH] [--model-db DIR | --layer-json PATH]

The inventory is the contract: `just ddl` renders schema.sql from it, and the db layer declares the same relations.
This compares the two and prints every difference, exit 1 on any:

- tables by name; each table's columns by name, type and nullability, in order; its primary key; its foreign keys
  (name, from-columns, to-columns) — schema.sql's FKs against `fks.ttrm`'s;
- views by name, kind (a materialized view is a `def table` in views.ttrm, ⚑NH-P-10; a plain view a `def view`) and
  column names, in order — the SELECT's output names. A view's types are not compared: the SQL text does not say them.

Types compare in the importer's vocabulary (ttr-core's SqlTypeMapper, which mirrors the metadata service's): a
Postgres type maps to the TTR-M type `ttr import-schema` writes, and `numeric(p,s)` to `decimal` with length p and
precision s, which the db layer carries by hand. So `jsonb` and `text` both read `text` — the one loss.

The schema side is sqlglot over schema.sql (the materialized views' SELECTs parsed one by one: sqlglot does not take
`CREATE MATERIALIZED VIEW … WITH NO DATA`); the layer side is ttr-core's own parser (scripts/db-layer.mjs, through
TATRMAN_PACKAGES — the justfile passes TTR_CORE's), or a JSON dump of it (`--layer-json`, the tests').

KNOWN_GAPS lists the columns the layer cannot declare (M0 S0.1 Found while doing). They are printed, never failed —
until the gap closes or changes: a known gap the layer now declares, or schema.sql no longer has, fails, so the entry
is removed in the change that closes it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

logging.getLogger("sqlglot").setLevel(logging.ERROR)  # `RESET search_path` is a Command to it, and harmless here

REPO = Path(__file__).resolve().parent.parent
SCHEMA_SQL = REPO / "data/ddl/schema.sql"
SCHEMA = "hartland"

# (table, column) → why the db layer cannot declare it.
KNOWN_GAPS: dict[tuple[str, str], str] = {
    ("document", "language"): "`language` is a TTR-M keyword, not an identifier",
    ("competitor_sighting", "description"): "`description` is a TTR-M keyword, not an identifier",
    ("delivery_event", "description"): "`description` is a TTR-M keyword, not an identifier",
}

# Postgres type (lower case, without its modifier) → the TTR-M type `ttr import-schema` writes for it.
PG_TO_TTR = {
    "bigint": "bigint",
    "int": "int",
    "integer": "int",
    "smallint": "int",
    "text": "text",
    "varchar": "text",
    "char": "text",
    "uuid": "text",
    "json": "text",
    "jsonb": "text",
    "boolean": "bool",
    "date": "date",
    "timestamptz": "datetime",
    "timestamp": "datetime",
    "time": "time",
    "numeric": "decimal",
    "decimal": "decimal",
    "double": "float",
    "real": "float",
}


@dataclass(frozen=True)
class Column:
    name: str
    type: str  # the TTR-M type
    length: int | None = None
    precision: int | None = None
    optional: bool = False

    def type_text(self) -> str:
        return self.type if self.length is None else f"{self.type}({self.length},{self.precision})"


@dataclass
class Relation:
    name: str
    kind: str  # table · materialized_view · view
    columns: list[Column] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    file: str = ""


@dataclass(frozen=True)
class ForeignKey:
    name: str
    columns: tuple[str, ...]  # "<table>.<column>"
    references: tuple[str, ...]


@dataclass
class Side:
    relations: dict[str, Relation] = field(default_factory=dict)
    fks: dict[str, ForeignKey] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# ── schema.sql ──────────────────────────────────────────────────────────────────────────────────────────────────────

_MATVIEW = re.compile(rf"^CREATE MATERIALIZED VIEW {SCHEMA}\.(\w+) AS\n(.*?)\nWITH NO DATA;", re.S | re.M)


def pg_type(kind: exp.DataType) -> tuple[str, int | None, int | None]:
    """A column's Postgres type → (TTR-M type, length, precision); length/precision only for numeric(p,s)."""
    text = kind.sql("postgres").lower()
    base = re.split(r"[\s(]", text, maxsplit=1)[0]
    if text.startswith("timestamp") and "time zone" in text or base == "timestamptz":
        base = "timestamptz"
    elif text.startswith("timestamp"):
        base = "timestamp"
    elif text.startswith("double"):
        base = "double"
    ttr = PG_TO_TTR.get(base)
    if ttr is None:
        raise ValueError(f"no TTR-M type for Postgres `{text}` — extend PG_TO_TTR (and the importer's mapping)")
    modifier = re.search(r"\((\d+)\s*,\s*(\d+)\)", text)
    if ttr == "decimal" and modifier:
        return ttr, int(modifier.group(1)), int(modifier.group(2))
    return ttr, None, None


def _names(node: exp.Expression | None) -> list[str]:
    return [e.name for e in (node.expressions if node is not None else [])]


def _select_columns(name: str, query: exp.Expr | exp.Expression, errors: list[str]) -> list[Column]:
    if not isinstance(query, exp.Query):
        errors.append(f"view {name}: its body is not a query")
        return []
    columns = []
    for projection in query.selects:
        if isinstance(projection, exp.Star) or projection.alias_or_name == "*":
            errors.append(f"view {name}: `SELECT *` — a view's columns must be named")
            continue
        columns.append(Column(projection.alias_or_name, ""))
    return columns


def schema_from_sql(text: str) -> Side:
    """schema.sql → its tables (columns, PK, FKs) and views (column names)."""
    side = Side()
    for match in _MATVIEW.finditer(text):
        name = match.group(1)
        query = sqlglot.parse_one(match.group(2), read="postgres")
        side.relations[name] = Relation(name, "materialized_view", _select_columns(name, query, side.errors))
    rest = _MATVIEW.sub("", text)
    for statement in sqlglot.parse(rest, read="postgres"):
        if not isinstance(statement, exp.Create):
            continue
        kind = statement.args.get("kind")
        if kind == "TABLE":
            _table(statement, side)
        elif kind == "VIEW":
            name = statement.this.name
            side.relations[name] = Relation(name, "view", _select_columns(name, statement.expression, side.errors))
    return side


def _table(statement: exp.Create, side: Side) -> None:
    schema = statement.this
    name = schema.this.name
    relation = Relation(name, "table")
    for element in schema.expressions:
        if isinstance(element, exp.ColumnDef):
            ttr, length, precision = pg_type(element.args["kind"])
            not_null = any(isinstance(c.args.get("kind"), exp.NotNullColumnConstraint) for c in element.constraints)
            relation.columns.append(Column(element.name, ttr, length, precision, optional=not not_null))
        elif isinstance(element, exp.Constraint):
            for constraint in element.expressions:
                if isinstance(constraint, exp.PrimaryKey):
                    relation.primary_key = _names(constraint)
                elif isinstance(constraint, exp.ForeignKey):
                    reference = constraint.args["reference"].this  # Schema(Table, [columns])
                    target = reference.this.name
                    side.fks[element.name] = ForeignKey(
                        element.name,
                        tuple(f"{name}.{c}" for c in _names(constraint)),
                        tuple(f"{target}.{c}" for c in _names(reference)),
                    )
    side.relations[name] = relation


# ── the db layer ────────────────────────────────────────────────────────────────────────────────────────────────────


def layer_from_json(dump: dict[str, Any]) -> Side:
    """scripts/db-layer.mjs's JSON → the layer's relations and FKs."""
    side = Side(errors=[f"model/db/{e}" for e in dump.get("parseErrors", [])])
    tables: dict[str, dict[str, Any]] = dump.get("tables", {})
    for name, t in tables.items():
        columns = [
            Column(c["name"], c["type"] or "", c.get("length"), c.get("precision"), bool(c.get("optional")))
            for c in t["columns"]
        ]
        kind = "table" if t["kind"] == "table" else "view"
        side.relations[name] = Relation(name, kind, columns, list(t["primaryKey"]), str(t["file"]))
    fks: dict[str, dict[str, list[str]]] = dump.get("fks", {})
    for name, fk in fks.items():
        side.fks[name] = ForeignKey(name, tuple(fk["from"]), tuple(fk["to"]))
    return side


def layer_from_model(db_dir: Path = REPO / "model/db") -> Side:
    """Run scripts/db-layer.mjs over DB_DIR with the ttr-core TATRMAN_PACKAGES names (default ../tatrman)."""
    result = subprocess.run(
        ["node", str(REPO / "scripts/db-layer.mjs"), str(db_dir)],
        capture_output=True,
        text=True,
        env=os.environ,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"ddl-drift: scripts/db-layer.mjs failed (is TATRMAN_PACKAGES a built ttr-core?):\n{result.stderr}"
        )
    return layer_from_json(json.loads(result.stdout))


# ── the comparison ──────────────────────────────────────────────────────────────────────────────────────────────────


def compare(
    schema: Side, layer: Side, known_gaps: dict[tuple[str, str], str] = KNOWN_GAPS
) -> tuple[list[str], list[str]]:
    """(differences, known-gap notes). Every difference names the relation and, where it is one, the column."""
    diffs = [*schema.errors, *layer.errors]
    notes: list[str] = []
    for name in sorted(set(schema.relations) - set(layer.relations)):
        diffs.append(f"{name}: in schema.sql ({schema.relations[name].kind}), not in model/db")
    for name in sorted(set(layer.relations) - set(schema.relations)):
        diffs.append(f"{name}: in model/db/{layer.relations[name].file}, not in schema.sql")
    for name in sorted(set(schema.relations) & set(layer.relations)):
        sql, ttr = schema.relations[name], layer.relations[name]
        if sql.kind == "table":
            diffs += _table_diffs(sql, ttr, known_gaps, notes)
        else:
            diffs += _view_diffs(sql, ttr)
    for table, column in sorted(known_gaps):
        if table in schema.relations and column not in {c.name for c in schema.relations[table].columns}:
            diffs.append(
                f"{table}.{column}: a known gap, but schema.sql has no such column — remove it from KNOWN_GAPS"
            )
    for name in sorted(set(schema.fks) - set(layer.fks)):
        fk = schema.fks[name]
        diffs.append(f"{fk.columns[0].split('.')[0]}: foreign key {name} ({', '.join(fk.columns)}) not in fks.ttrm")
    for name in sorted(set(layer.fks) - set(schema.fks)):
        diffs.append(f"foreign key {name}: in fks.ttrm, not in schema.sql")
    for name in sorted(set(schema.fks) & set(layer.fks)):
        sql_fk, ttr_fk = schema.fks[name], layer.fks[name]
        if (sql_fk.columns, sql_fk.references) != (ttr_fk.columns, ttr_fk.references):
            diffs.append(
                f"foreign key {name}: schema.sql {', '.join(sql_fk.columns)} → {', '.join(sql_fk.references)}; "
                f"fks.ttrm {', '.join(ttr_fk.columns)} → {', '.join(ttr_fk.references)}"
            )
    return diffs, notes


def _table_diffs(sql: Relation, ttr: Relation, known_gaps: dict[tuple[str, str], str], notes: list[str]) -> list[str]:
    diffs = []
    if ttr.kind != "table":
        diffs.append(f"{sql.name}: a table in schema.sql, a `def view` in model/db/{ttr.file}")
    declared = {c.name: c for c in ttr.columns}
    expected = []
    for column in sql.columns:
        gap = known_gaps.get((sql.name, column.name))
        if gap is not None:
            if column.name in declared:
                diffs.append(
                    f"{sql.name}.{column.name}: a known gap, but model/db declares it — remove it from KNOWN_GAPS"
                )
            else:
                notes.append(f"{sql.name}.{column.name}: not in model/db — known gap: {gap}")
            continue
        expected.append(column)
        mine = declared.get(column.name)
        if mine is None:
            diffs.append(f"{sql.name}.{column.name}: in schema.sql ({column.type_text()}), not in model/db/{ttr.file}")
            continue
        if (mine.type, mine.length, mine.precision) != (column.type, column.length, column.precision):
            diffs.append(
                f"{sql.name}.{column.name}: type {column.type_text()} in schema.sql, {mine.type_text()} in model/db"
            )
        if mine.optional != column.optional:
            say = {True: "nullable", False: "NOT NULL"}
            diffs.append(
                f"{sql.name}.{column.name}: {say[column.optional]} in schema.sql, {say[mine.optional]} in model/db"
            )
    for column in ttr.columns:
        if column.name not in {c.name for c in sql.columns}:
            diffs.append(f"{sql.name}.{column.name}: in model/db/{ttr.file}, not in schema.sql")
    order_sql = [c.name for c in expected if c.name in declared]
    order_ttr = [c.name for c in ttr.columns if c.name in {e.name for e in expected}]
    if order_sql != order_ttr:
        diffs.append(f"{sql.name}: column order differs — schema.sql {order_sql}, model/db {order_ttr}")
    if sql.primary_key != ttr.primary_key:
        diffs.append(f"{sql.name}: primary key {sql.primary_key} in schema.sql, {ttr.primary_key} in model/db")
    return diffs


def _view_diffs(sql: Relation, ttr: Relation) -> list[str]:
    diffs = []
    want = "table" if sql.kind == "materialized_view" else "view"
    if ttr.kind != want:
        diffs.append(
            f"{sql.name}: a {sql.kind} in schema.sql, so a `def {want}` in model/db — it is a `def {ttr.kind}`"
        )
    names_sql, names_ttr = [c.name for c in sql.columns], [c.name for c in ttr.columns]
    for name in names_sql:
        if name not in names_ttr:
            diffs.append(f"{sql.name}.{name}: a column of the view in schema.sql, not in model/db/{ttr.file}")
    for name in names_ttr:
        if name not in names_sql:
            diffs.append(f"{sql.name}.{name}: in model/db/{ttr.file}, not a column of the view in schema.sql")
    if set(names_sql) == set(names_ttr) and names_sql != names_ttr:
        diffs.append(f"{sql.name}: column order differs — schema.sql {names_sql}, model/db {names_ttr}")
    return diffs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ddl-drift", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--schema", type=Path, default=SCHEMA_SQL, help="the rendered DDL (default data/ddl/schema.sql)"
    )
    parser.add_argument("--model-db", type=Path, default=REPO / "model/db", help="the db layer (default model/db)")
    parser.add_argument("--layer-json", type=Path, help="a scripts/db-layer.mjs dump instead of running it")
    args = parser.parse_args(argv)
    schema = schema_from_sql(args.schema.read_text(encoding="utf-8"))
    if args.layer_json:
        layer = layer_from_json(json.loads(args.layer_json.read_text(encoding="utf-8")))
    else:
        layer = layer_from_model(args.model_db)
    diffs, notes = compare(schema, layer)
    for note in notes:
        print(f"known gap  {note}")
    for diff in diffs:
        print(f"DRIFT      {diff}")
    tables = sum(r.kind == "table" for r in schema.relations.values())
    views = len(schema.relations) - tables
    verdict = f"{len(diffs)} difference(s)" if diffs else "clean"
    print(f"ddl-drift: {tables} tables, {views} views, {len(schema.fks)} foreign keys — {verdict}")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
