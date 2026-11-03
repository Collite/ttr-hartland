"""The streaming CSV writer (architecture §2): one file per table, `<run>/<table>.csv`, its header the table's columns
in `data/inventory.yaml`, its cells the text `COPY … (FORMAT csv, HEADER)` reads back as the same values.

Each cell is formatted by its column's type, so the same value always gives the same bytes: an integer as itself, a
`numeric(p,s)` with exactly `s` decimals (a value that is not finite, or too large for `p` digits, is refused here
rather than by COPY after the whole build), a date as ISO, a `time` as `HH:MM`, a `timestamptz` as ISO with its offset,
a boolean as `true`/`false`, `jsonb` as compact JSON with sorted keys, text as is. `None` is an empty cell — COPY's
NULL — so an empty text, which COPY would read as NULL too, is refused rather than written.

`CsvWriter` keeps each table's file open and appends: setup writes its tables once, the day loop appends each day's
rows (`append`); a table's file is created with its header on its first rows, and `write_all` writes a set of tables
in the inventory's `load_order`.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, time
from functools import cache
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from hartland_sim.config import Table, inventory

_NUMERIC = re.compile(r"numeric\((\d+),(\d+)\)")


class CellError(ValueError):
    """A value its column's type cannot hold; the message names the table, the column and the value."""


@cache
def _tables() -> dict[str, Table]:
    return {table.name: table for table in inventory().tables}


Format = Callable[[Any], str | None]


def cell(value: Any, kind: str) -> str | None:
    """The CSV text of `value` in a column of type `kind`; `None` for NULL."""
    return _format(kind)(value)


@cache
def _format(kind: str) -> Format:
    """The function `cell(·, kind)` is, made once per type: the type's checks resolved here, not per cell."""
    if kind in ("bigint", "integer"):
        return _integer
    numeric = _NUMERIC.fullmatch(kind)
    if numeric:
        precision, scale = int(numeric.group(1)), int(numeric.group(2))
        return _numeric(kind, f".{scale}f", 10 ** (precision - scale))
    found = _FORMATS.get(kind)
    if found is None:
        raise CellError(f"no format for type {kind}")
    return found


def _integer(value: Any) -> str | None:
    if type(value) is int:  # the common case: an id, a count
        return str(value)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CellError(f"{value!r} is not an integer")
    return str(value)


def _numeric(kind: str, spec: str, limit: int) -> Format:
    safe = limit - 1  # |value| below it formats to fewer digits than p allows, whatever the rounding to s decimals

    def number(value: Any) -> str | None:
        if type(value) is float and -safe < value < safe:  # the common case, NaN excluded by the comparison
            return format(value, spec)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CellError(f"{value!r} is not a number")
        if not math.isfinite(value):
            raise CellError(f"{value!r} is not a finite number")
        text = format(value, spec)
        if int(text.partition(".")[0].lstrip("-")) >= limit:  # |the text| ≥ 10^(p − s): too many digits for p
            raise CellError(f"{value!r} does not fit {kind}")
        return text

    return number


def _boolean(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise CellError(f"{value!r} is not a boolean")
    return "true" if value else "false"


def _date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime) or not isinstance(value, date):
        raise CellError(f"{value!r} is not a date")
    return value.isoformat()


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CellError(f"{value!r} is not an aware datetime")
    return value.isoformat()


def _time(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, time):
        return value.strftime("%H:%M")
    if isinstance(value, str) and _TIME.fullmatch(value):
        return value
    raise CellError(f"{value!r} is not a time")


def _json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CellError(f"{value!r} is not JSON: {error}") from None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CellError(f"{value!r} is not text")
    if value == "":
        raise CellError("an empty text, which COPY reads as NULL")
    return value


_TIME = re.compile(r"([01]\d|2[0-3]):[0-5]\d")
_FORMATS: dict[str, Format] = {
    "boolean": _boolean,
    "date": _date,
    "timestamptz": _timestamp,
    "time": _time,
    "jsonb": _json,
    "text": _text,
}


@cache
def _columns(name: str) -> tuple[tuple[tuple[str, Format], ...], tuple[int, ...]]:
    """`name`'s columns as (name, its format), in the inventory's order, and the indices of the NOT NULL ones."""
    columns = _tables()[name].columns
    required = tuple(i for i, column in enumerate(columns) if not column.nullable)
    return tuple((column.name, _format(column.type)) for column in columns), required


def cells(table: Table, row: Mapping[str, Any]) -> list[str | None]:
    columns, required = _columns(table.name)
    try:
        out = [form(row[name]) for name, form in columns]
    except CellError:
        for name, form in columns:  # again, to name the column
            try:
                form(row[name])
            except CellError as error:
                raise CellError(f"{table.name}.{name}: {error}") from None
        raise
    for i in required:
        if out[i] is None:
            raise CellError(f"{table.name}.{columns[i][0]}: NULL in a NOT NULL column")
    return out


class CsvWriter:
    """Append rows to `<run>/<table>.csv`, one open file per table."""

    def __init__(self, run: Path) -> None:
        self.run = run
        self.files: dict[str, tuple[IO[str], Any]] = {}

    def __enter__(self) -> CsvWriter:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()

    def _writer(self, name: str) -> Any:
        if name not in self.files:
            table = _tables()[name]
            path = self.run / f"{name}.csv"
            new = not path.exists()
            self.run.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8", newline="")
            writer = csv.writer(handle, lineterminator="\n")
            if new:
                writer.writerow([column.name for column in table.columns])
            self.files[name] = (handle, writer)
        return self.files[name][1]

    def append(self, name: str, rows: Iterable[Mapping[str, Any]]) -> int:
        """Append `rows` to `name`'s file; the number written."""
        table = _tables()[name]
        writer = self._writer(name)
        n = 0
        for row in rows:
            writer.writerow(cells(table, row))
            n += 1
        return n

    def write_all(self, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
        """Each of `tables` in the inventory's `load_order`; the names written. A name no table has is refused."""
        stray = sorted(set(tables) - set(_tables()))
        if stray:
            raise KeyError(f"not in the inventory: {stray}")
        written = []
        for name in _tables():
            if name in tables:
                self.append(name, tables[name])
                written.append(name)
        return written

    def close(self) -> None:
        for handle, _ in self.files.values():
            handle.close()
        self.files.clear()
