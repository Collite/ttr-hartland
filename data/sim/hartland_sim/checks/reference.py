"""`just reference-check` (contracts §7): every `data/reference/*.csv` against its schema, its row count, its keys,
its references to the other files and the NH-D8 guard. The convention is `hartland_sim.io.reference`'s docstring.

`check()` lists every problem at once (up to `PER_FILE` per file), each naming the file, the line and the column. A
file that cannot be read through (a broken schema, header or CSV syntax) is one problem, and the files that refer to
it say they were not checked against it. A folder with no reference files at all is a problem too.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, SchemaError

from hartland_sim.checks.czech import assert_czech, assert_czech_column
from hartland_sim.io.reference import (
    REFERENCE_DIR,
    ReferenceFileError,
    Row,
    column_types,
    decode,
    format_problems,
    load_schema,
    records,
    schema_path,
)

PER_FILE = 20

SCHEMA_SUFFIX = ".schema.json"

# A schema's top level holds these and nothing else: `io.reference.validator` checks the rows against the columns
# alone, so an `allOf`, an `if`, a `$defs` there would be ignored.
CONVENTION_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "title",
        "description",
        "x-rows",
        "x-key",
        "type",
        "additionalProperties",
        "required",
        "properties",
    }
)

_REF = re.compile(r"\w+\.\w+")


@dataclass
class Report:
    files: int = 0
    rows: int = 0
    count: int = 0  # every problem found; `problems` lists up to PER_FILE of each file's
    problems: list[str] = field(default_factory=list)


@dataclass
class _File:
    name: str
    schema: dict[str, Any] = field(default_factory=dict)
    records: int = 0
    rows: list[Row] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)
    loaded: bool = False  # the schema held, and the CSV was read through to its end
    problems: list[str] = field(default_factory=list)

    def problem(self, message: str) -> None:
        self.problems.append(f"{self.name}.csv{message}")


def check(root: Path = REFERENCE_DIR) -> Report:
    report = Report()
    csvs = {path.stem for path in root.glob("*.csv")}
    schemas = {path.name.removesuffix(SCHEMA_SUFFIX) for path in (root / "schemas").glob(f"*{SCHEMA_SUFFIX}")}
    alone = [f"{name}.csv: no schema (schemas/{name}{SCHEMA_SUFFIX})" for name in sorted(csvs - schemas)]
    alone += [f"schemas/{name}{SCHEMA_SUFFIX}: no {name}.csv" for name in sorted(schemas - csvs)]
    if not csvs and not schemas:
        alone.append(f"{root}: no reference files (is --dir right?)")
    report.count = len(alone)
    report.problems += alone
    files = [_read(root, name) for name in sorted(csvs & schemas)]
    by_name = {file.name: file for file in files}
    for file in files:
        _references(file, by_name)
        report.files += 1
        report.rows += len(file.rows)
        report.count += len(file.problems)
        report.problems += file.problems[:PER_FILE]
        if len(file.problems) > PER_FILE:
            report.problems.append(f"{file.name}.csv: … and {len(file.problems) - PER_FILE} more")
    return report


def _read(root: Path, name: str) -> _File:
    file = _File(name)
    try:
        file.schema = load_schema(schema_path(root, name))
    except ReferenceFileError as error:
        file.problems.append(str(error))
        return file
    if not _schema_holds(file):
        return file
    raw = (root / f"{name}.csv").read_bytes()
    file.problems += format_problems(name, raw)
    try:
        for line, row, problems in records(name, decode(name, raw), file.schema):
            file.records += 1
            file.problems += problems
            if row is not None:
                file.rows.append(row)
                file.lines.append(line)
    except ReferenceFileError as error:
        file.problems.append(str(error))
        return file
    file.loaded = True
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
    columns = schema.get("properties")
    if not columns or not all(isinstance(spec, dict) for spec in columns.values()):
        wants = ["`properties`: an object per column"]
    else:
        wants = _wants(schema, columns)
    for want in wants:
        file.problems.append(f"schemas/{file.name}{SCHEMA_SUFFIX}: the convention wants {want}")
    return not wants


def _wants(schema: Mapping[str, Any], columns: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """What a schema with an object per column lacks of the convention."""
    wants = []
    other = sorted(set(schema) - CONVENTION_KEYWORDS)
    if other:
        wants.append(f"no other top-level keyword (the row validator reads the columns only): {other}")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        wants.append("`type: object` and `additionalProperties: false`")
    if schema.get("required") != list(columns):
        wants.append("`required` lists every column, in order")
    if not _is_range(schema.get("x-rows")):
        wants.append("`x-rows: [min, max]`")
    key = schema.get("x-key", [])
    if not isinstance(key, list) or not all(isinstance(column, str) and column in columns for column in key):
        wants.append("`x-key` names columns")
    for column, spec in columns.items():
        if spec.get("x-czech") and "string" not in column_types(spec):
            wants.append(f"`x-czech` on a string column only (`{column}`)")
        if "x-ref" in spec and not (isinstance(spec["x-ref"], str) and _REF.fullmatch(spec["x-ref"])):
            wants.append(f'`x-ref: "<file>.<column>"` (`{column}`)')
    return wants


def _is_range(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(type(bound) is int for bound in value)
        and 0 <= value[0] <= value[1]
    )


def _count(file: _File) -> None:
    low, high = file.schema["x-rows"]
    if not low <= file.records <= high:
        expected = str(low) if low == high else f"{low}–{high}"
        file.problem(f": {file.records} rows, expected {expected}")


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
    """Each value alone, on its line — a value that is not NFC `records()` has already reported — then the column's
    values together."""
    for column, spec in file.schema["properties"].items():
        if not spec.get("x-czech"):
            continue
        passed = []
        for line, row in zip(file.lines, file.rows, strict=True):
            value = row[column]
            if not isinstance(value, str) or not value or not unicodedata.is_normalized("NFC", value):
                continue
            try:
                assert_czech(value, what=f"{file.name}.csv line {line}, column `{column}`")
            except AssertionError as error:
                file.problems.append(str(error))
            else:
                passed.append(value)
        try:
            assert_czech_column(passed, what=f"column `{column}`")
        except AssertionError as error:
            file.problem(f", {error}")


def _references(file: _File, files: Mapping[str, _File]) -> None:
    if not file.loaded:
        return
    for column, spec in file.schema["properties"].items():
        target = spec.get("x-ref")
        if not target:
            continue
        target_name, _, target_column = target.partition(".")
        other = files.get(target_name)
        if other is not None and not other.loaded:
            file.problem(f", column `{column}`: not checked — {target_name}.csv did not load")
            continue
        if other is None or target_column not in other.schema["properties"]:
            file.problem(f", column `{column}`: x-ref {target!r} names no file column")
            continue
        known = {row[target_column] for row in other.rows}
        for line, row in zip(file.lines, file.rows, strict=True):
            if row[column] is not None and row[column] not in known:
                file.problem(f" line {line}, column `{column}`: {row[column]!r} is not in {target}")
