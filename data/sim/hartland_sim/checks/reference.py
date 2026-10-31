"""`just reference-check` (contracts §7): every `data/reference/*.csv` against its schema, its row count, its keys,
its references to the other files and the NH-D8 guard. The convention is `hartland_sim.io.reference`'s docstring.

`check()` lists every problem at once (up to `PER_FILE` per file), each naming the file, the line and the column.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, SchemaError

from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.io.reference import REFERENCE_DIR, Row, load_schema, parse_cell, split, validator

PER_FILE = 20

SCHEMA_SUFFIX = ".schema.json"


@dataclass
class Report:
    files: int = 0
    rows: int = 0
    problems: list[str] = field(default_factory=list)


@dataclass
class _File:
    name: str
    schema: dict[str, Any]
    rows: list[Row] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def problem(self, message: str) -> None:
        self.problems.append(f"{self.name}.csv{message}")


def check(root: Path = REFERENCE_DIR) -> Report:
    report = Report()
    csvs = {path.stem for path in root.glob("*.csv")}
    schemas = {path.name.removesuffix(SCHEMA_SUFFIX) for path in (root / "schemas").glob(f"*{SCHEMA_SUFFIX}")}
    report.problems += [f"{name}.csv: no schema (schemas/{name}{SCHEMA_SUFFIX})" for name in sorted(csvs - schemas)]
    report.problems += [f"schemas/{name}{SCHEMA_SUFFIX}: no {name}.csv" for name in sorted(schemas - csvs)]
    files = [_read(root, name) for name in sorted(csvs & schemas)]
    by_name = {file.name: file for file in files}
    for file in files:
        _references(file, by_name)
        report.files += 1
        report.rows += len(file.rows)
        report.problems += file.problems[:PER_FILE]
        if len(file.problems) > PER_FILE:
            report.problems.append(f"{file.name}.csv: … and {len(file.problems) - PER_FILE} more")
    return report


def _read(root: Path, name: str) -> _File:
    schema = load_schema(root / "schemas" / f"{name}{SCHEMA_SUFFIX}")
    file = _File(name, schema)
    if not _schema_holds(file):
        return file
    raw = (root / f"{name}.csv").read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        file.problem(": starts with a byte-order mark")
    if b"\r" in raw:
        file.problem(": CR line ends (the convention is \\n)")
    if raw and not raw.endswith(b"\n"):
        file.problem(": no newline at the end")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        file.problem(f": not UTF-8 ({error})")
        return file
    columns: dict[str, Mapping[str, Any]] = schema["properties"]
    header, records = split(text)
    if header != list(columns):
        file.problem(f" line 1: header {header} ≠ the schema's columns {list(columns)}")
        return file
    check = validator(schema)
    for line, record in records:
        row = _row(file, line, header, record, columns)
        if row is None:
            continue
        for violation in check.iter_errors(row):
            where = f", column `{violation.absolute_path[0]}`" if violation.absolute_path else ""
            file.problem(f" line {line}{where}: {violation.message}")
        file.rows.append(row)
        file.lines.append(line)
    _count(file)
    _keys(file)
    _czech(file)
    return file


def _schema_holds(file: _File) -> bool:
    schema = file.schema
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        file.problems.append(f"schemas/{file.name}{SCHEMA_SUFFIX}: {error.message}")
        return False
    columns = schema.get("properties") or {}
    wants = []
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        wants.append("`type: object` and `additionalProperties: false`")
    if schema.get("required") != list(columns):
        wants.append("`required` lists every column, in order")
    if not _is_range(schema.get("x-rows")):
        wants.append("`x-rows: [min, max]`")
    if not set(schema.get("x-key", [])) <= set(columns):
        wants.append("`x-key` names columns")
    for want in wants:
        file.problems.append(f"schemas/{file.name}{SCHEMA_SUFFIX}: the convention wants {want}")
    return not wants


def _is_range(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(bound, int) for bound in value)
        and 0 <= value[0] <= value[1]
    )


def _row(file: _File, line: int, header: list[str], record: list[str], columns: Mapping[str, Any]) -> Row | None:
    if len(record) != len(header):
        file.problem(f" line {line}: {len(record)} cells, the header has {len(header)}")
        return None
    row: Row = {}
    for column, value in zip(header, record, strict=True):
        where = f" line {line}, column `{column}`"
        if value != value.strip():
            file.problem(f"{where}: whitespace around {value!r}")
        if not unicodedata.is_normalized("NFC", value):
            file.problem(f"{where}: not NFC")
        try:
            row[column] = parse_cell(value, columns[column])
        except ValueError as error:
            file.problem(f"{where}: {error}")
            return None
    return row


def _count(file: _File) -> None:
    low, high = file.schema["x-rows"]
    if not low <= len(file.rows) <= high:
        expected = str(low) if low == high else f"{low}–{high}"
        file.problem(f": {len(file.rows)} rows, expected {expected}")


def _keys(file: _File) -> None:
    key = file.schema.get("x-key")
    if not key:
        return
    seen: dict[tuple[Any, ...], int] = {}
    for line, row in zip(file.lines, file.rows, strict=True):
        value = tuple(row[column] for column in key)
        if value in seen:
            file.problem(f" line {line}: key ({', '.join(key)}) = {value} again (first on line {seen[value]})")
        seen.setdefault(value, line)


def _czech(file: _File) -> None:
    for column, spec in file.schema["properties"].items():
        if spec.get("x-czech"):
            texts = [row[column] for row in file.rows if row[column] not in (None, "")]
            try:
                assert_czech_all(texts, what=f"column `{column}`")
            except AssertionError as error:
                file.problem(f", {error}")


def _references(file: _File, files: Mapping[str, _File]) -> None:
    for column, spec in file.schema["properties"].items():
        target = spec.get("x-ref")
        if not target:
            continue
        target_name, _, target_column = target.partition(".")
        other = files.get(target_name)
        if other is None or target_column not in other.schema["properties"]:
            file.problem(f", column `{column}`: x-ref {target!r} names no file column")
            continue
        known = {row[target_column] for row in other.rows}
        for line, row in zip(file.lines, file.rows, strict=True):
            if row[column] is not None and row[column] not in known:
                file.problem(f" line {line}, column `{column}`: {row[column]!r} is not in {target}")
