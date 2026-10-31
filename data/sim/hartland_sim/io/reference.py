"""The committed reference files: `data/reference/<name>.csv`, each described by `schemas/<name>.schema.json`.

The convention (G0 S0.2 T1). A schema is JSON Schema (draft 2020-12) for **one row**: `properties` in the CSV's
column order, each with its JSON type — `string`, `integer`, `number` or `boolean`, plus `null` when the cell may be
empty — and its constraints (`enum`, `pattern`, `format: date`, `minimum`, …). The keywords a row cannot carry:

- `x-rows: [min, max]` — the expected row count (required);
- `x-key: [column, …]` — those columns are unique together;
- `x-ref: "<file>.<column>"` on a column — every non-empty value exists in that column of that file;
- `x-czech: true` on a string column — its non-empty values read as Czech (`checks.czech`, NH-D8): each value alone,
  then the column together.

The CSV is UTF-8 without a BOM, comma-separated, `"`-quoted where needed, `\\n` line ends, one header row; an empty
cell is NULL, a boolean is `true` or `false`, every cell is NFC with no surrounding whitespace.

`records()` reads a file against its schema; `checks.reference.check()` (`just reference-check`) lists every problem
it finds, plus the keywords above; `read()`, the generator's loader, stops at the first.
"""

from __future__ import annotations

import codecs
import csv
import io
import json
import re
import unicodedata
from collections.abc import Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

# This file is `data/sim/hartland_sim/io/reference.py`: four levels below the repo root. A wheel install has no
# `data/reference` beside it, and `check()` then reports the folder empty.
REFERENCE_DIR = Path(__file__).resolve().parents[4] / "data" / "reference"

Row = dict[str, Any]

_INTEGER = re.compile(r"-?(0|[1-9]\d*)")
_NUMBER = re.compile(r"-?(0|[1-9]\d*)(\.\d+)?")


class ReferenceFileError(ValueError):
    """A reference file that breaks the convention or its schema; the message names the file, the line and the
    column."""


def schema_path(root: Path, name: str) -> Path:
    return root / "schemas" / f"{name}.schema.json"


def load_schema(path: Path) -> dict[str, Any]:
    """The schema at `path`; `ReferenceFileError` when it is not a JSON object."""
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ReferenceFileError(f"schemas/{path.name}: not JSON ({error})") from None
    if not isinstance(schema, dict):
        raise ReferenceFileError(f"schemas/{path.name}: not a JSON object")
    return schema


def column_types(column: Mapping[str, Any]) -> list[str]:
    declared = column.get("type", "string")
    return [declared] if isinstance(declared, str) else list(declared)


def parse_cell(value: str, column: Mapping[str, Any]) -> Any:
    """The CSV text of a cell → its JSON value, by the column's declared type; `ValueError` when it is not one."""
    types = column_types(column)
    if value == "":
        if "null" in types:
            return None
        if "string" in types:
            return ""
        raise ValueError("empty, and the column is not nullable")
    if "integer" in types and _INTEGER.fullmatch(value):
        return int(value)
    if "number" in types and _NUMBER.fullmatch(value):
        return int(value) if _INTEGER.fullmatch(value) else float(value)
    if "boolean" in types and value in ("true", "false"):
        return value == "true"
    if "string" in types:
        return value
    raise ValueError(f"{value!r} is not {' or '.join(t for t in types if t != 'null')}")


_UNCONSTRAINING = frozenset({"type", "description", "title"})


def validator(schema: Mapping[str, Any]) -> Draft202012Validator:
    """The row validator: the schema's constraints, without what the reader already guarantees — each cell's type
    (`parse_cell`), every column present and no other (the header). Validating only the constrained columns gives
    the same verdicts several times faster on the large OSM files."""
    columns = {
        column: spec
        for column, spec in schema["properties"].items()
        if any(not keyword.startswith("x-") and keyword not in _UNCONSTRAINING for keyword in spec)
    }
    return Draft202012Validator(
        {"$schema": schema.get("$schema", ""), "type": "object", "properties": columns},
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def format_problems(name: str, raw: bytes) -> list[str]:
    """What the file's bytes break of the convention's format: a BOM, CR line ends, no newline at the end."""
    problems = []
    if raw.startswith(codecs.BOM_UTF8):
        problems.append(f"{name}.csv: starts with a byte-order mark")
    if b"\r" in raw:
        problems.append(f"{name}.csv: CR line ends (the convention is \\n)")
    if raw and not raw.endswith(b"\n"):
        problems.append(f"{name}.csv: no newline at the end")
    return problems


def decode(name: str, raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReferenceFileError(f"{name}.csv: not UTF-8 ({error})") from None


def records(name: str, text: str, schema: Mapping[str, Any]) -> Iterator[tuple[int, Row | None, list[str]]]:
    """Each record of a reference CSV's text against its schema, in order: the line it ends on, its JSON-typed row
    (`None` when it has the wrong number of cells or a cell does not parse) and its problems, each naming the file,
    the line and the column. `ReferenceFileError` where reading cannot go on: a header that is not the schema's
    columns, broken CSV syntax."""
    columns: Mapping[str, Mapping[str, Any]] = schema["properties"]
    check = validator(schema)
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, [])
        if header != list(columns):
            raise ReferenceFileError(f"{name}.csv line 1: header {header} ≠ the schema's columns {list(columns)}")
        for record in reader:
            yield reader.line_num, *_row(f"{name}.csv line {reader.line_num}", record, columns, check)
    except csv.Error as error:
        raise ReferenceFileError(f"{name}.csv line {reader.line_num}: {error}") from None


def _row(
    where: str, record: list[str], columns: Mapping[str, Mapping[str, Any]], check: Draft202012Validator
) -> tuple[Row | None, list[str]]:
    if len(record) != len(columns):
        return None, [f"{where}: {len(record)} cells, the header has {len(columns)}"]
    row: Row = {}
    problems: list[str] = []
    for (column, spec), value in zip(columns.items(), record, strict=True):
        cell = f"{where}, column `{column}`"
        if value != value.strip():
            problems.append(f"{cell}: whitespace around {value!r}")
        if not unicodedata.is_normalized("NFC", value):
            problems.append(f"{cell}: not NFC")
        try:
            row[column] = parse_cell(value, spec)
        except ValueError as error:
            problems.append(f"{cell}: {error}")
            return None, problems
    for violation in check.iter_errors(row):
        cell = f"{where}, column `{violation.absolute_path[0]}`" if violation.absolute_path else where
        problems.append(f"{cell}: {violation.message}")
    return row, problems


def read(name: str, root: Path = REFERENCE_DIR) -> list[Row]:
    """The typed rows of `<root>/<name>.csv` (dates as `datetime.date`); `ReferenceFileError` at the first problem
    `records()` finds. `just reference-check` lists every problem at once — this is the generator's loader."""
    schema = load_schema(schema_path(root, name))
    raw = (root / f"{name}.csv").read_bytes()
    problems = format_problems(name, raw)
    if problems:
        raise ReferenceFileError(problems[0])
    dates = [column for column, spec in schema["properties"].items() if spec.get("format") == "date"]
    rows: list[Row] = []
    for _, row, problems in records(name, decode(name, raw), schema):
        if row is None or problems:
            raise ReferenceFileError(problems[0])
        rows.append(row | {column: date.fromisoformat(row[column]) for column in dates if row[column] is not None})
    return rows
