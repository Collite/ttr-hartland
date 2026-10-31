"""The committed reference files: `data/reference/<name>.csv`, each described by `schemas/<name>.schema.json`.

The convention (G0 S0.2 T1). A schema is JSON Schema (draft 2020-12) for **one row**: `properties` in the CSV's
column order, each with its JSON type — `string`, `integer`, `number` or `boolean`, plus `null` when the cell may be
empty — and its constraints (`enum`, `pattern`, `format: date`, `minimum`, …). The keywords a row cannot carry:

- `x-rows: [min, max]` — the expected row count (required);
- `x-key: [column, …]` — those columns are unique together;
- `x-ref: "<file>.<column>"` on a column — every non-empty value exists in that column of that file;
- `x-czech: true` on a column — its non-empty values read as Czech (`checks.czech.assert_czech_all`, NH-D8).

The CSV is UTF-8 without a BOM, comma-separated, `"`-quoted where needed, `\\n` line ends, one header row; an empty
cell is NULL, a boolean is `true` or `false`, every cell is NFC with no surrounding whitespace.
`checks.reference.check()` enforces all of it; `read()` is how the generator loads a file.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REFERENCE_DIR = Path("data/reference")

Row = dict[str, Any]

_INTEGER = re.compile(r"-?(0|[1-9]\d*)")
_NUMBER = re.compile(r"-?(0|[1-9]\d*)(\.\d+)?")


class ReferenceFileError(ValueError):
    """A reference file that breaks its schema; the message names the file, the line and the column."""


def schema_path(root: Path, name: str) -> Path:
    return root / "schemas" / f"{name}.schema.json"


def load_schema(path: Path) -> dict[str, Any]:
    schema: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
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


def split(text: str) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """The header and the records of a CSV text, each record with the line it ends on."""
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader, [])
    records = [(reader.line_num, record) for record in reader]
    return header, records


def validator(schema: Mapping[str, Any]) -> Draft202012Validator:
    return Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)


def read(name: str, root: Path = REFERENCE_DIR) -> list[Row]:
    """The typed rows of `<root>/<name>.csv` (dates as `datetime.date`); `ReferenceFileError` on the first bad cell.
    `just reference-check` lists every problem at once — this is the generator's loader, not the check."""
    schema = load_schema(schema_path(root, name))
    columns: dict[str, Mapping[str, Any]] = schema["properties"]
    header, records = split((root / f"{name}.csv").read_text(encoding="utf-8"))
    if header != list(columns):
        raise ReferenceFileError(f"{name}.csv: header {header} ≠ the schema's columns {list(columns)}")
    check = validator(schema)
    rows: list[Row] = []
    for line, record in records:
        if len(record) != len(header):
            raise ReferenceFileError(f"{name}.csv line {line}: {len(record)} cells, the header has {len(header)}")
        row: Row = {}
        for column, value in zip(header, record, strict=True):
            try:
                row[column] = parse_cell(value, columns[column])
            except ValueError as error:
                raise ReferenceFileError(f"{name}.csv line {line}, column `{column}`: {error}") from None
        for violation in check.iter_errors(row):
            where = f", column `{violation.absolute_path[0]}`" if violation.absolute_path else ""
            raise ReferenceFileError(f"{name}.csv line {line}{where}: {violation.message}")
        for column, spec in columns.items():
            if spec.get("format") == "date" and row[column] is not None:
                row[column] = date.fromisoformat(row[column])
        rows.append(row)
    return rows
