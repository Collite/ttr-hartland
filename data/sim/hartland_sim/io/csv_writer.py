"""The streaming CSV writer (architecture §2): one file per table, `<run>/<table>.csv`, its header the table's columns
in `data/inventory.yaml`, its cells the text `COPY … (FORMAT csv, HEADER)` reads back as the same values.

Each cell is formatted by its column's type, so the same value always gives the same bytes: an integer as itself, a
`numeric(p,s)` with exactly `s` decimals, a date as ISO, a `time` as `HH:MM`, a `timestamptz` as ISO with its offset,
a boolean as `true`/`false`, `jsonb` as compact JSON with sorted keys, text as is. `None` is an empty cell — COPY's
NULL — so an empty text, which COPY would read as NULL too, is refused rather than written.

`CsvWriter` keeps each table's file open and appends: setup writes its tables once, the day loop appends each day's
rows (`append`); a table's file is created with its header on its first rows, and `write_all` writes a set of tables
in the inventory's `load_order`.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, time
from functools import cache
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from hartland_sim.config import InventoryConfig, Table

_NUMERIC = re.compile(r"numeric\((\d+),(\d+)\)")


class CellError(ValueError):
    """A value its column's type cannot hold; the message names the table, the column and the value."""


@cache
def _tables() -> dict[str, Table]:
    return {table.name: table for table in InventoryConfig.load().tables}


def cell(value: Any, kind: str) -> str | None:
    """The CSV text of `value` in a column of type `kind`; `None` for NULL."""
    if value is None:
        return None
    if kind in ("bigint", "integer"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CellError(f"{value!r} is not an integer")
        return str(value)
    numeric = _NUMERIC.fullmatch(kind)
    if numeric:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CellError(f"{value!r} is not a number")
        return f"{value:.{int(numeric.group(2))}f}"
    if kind == "boolean":
        if not isinstance(value, bool):
            raise CellError(f"{value!r} is not a boolean")
        return "true" if value else "false"
    if kind == "date":
        if isinstance(value, datetime) or not isinstance(value, date):
            raise CellError(f"{value!r} is not a date")
        return value.isoformat()
    if kind == "timestamptz":
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise CellError(f"{value!r} is not an aware datetime")
        return value.isoformat()
    if kind == "time":
        if isinstance(value, time):
            return value.strftime("%H:%M")
        if isinstance(value, str) and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value):
            return value
        raise CellError(f"{value!r} is not a time")
    if kind == "jsonb":
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if kind == "text":
        if not isinstance(value, str):
            raise CellError(f"{value!r} is not text")
        if value == "":
            raise CellError("an empty text, which COPY reads as NULL")
        return value
    raise CellError(f"no format for type {kind}")


def cells(table: Table, row: Mapping[str, Any]) -> list[str | None]:
    out = []
    for column in table.columns:
        try:
            out.append(cell(row[column.name], column.type))
        except CellError as error:
            raise CellError(f"{table.name}.{column.name}: {error}") from None
        if out[-1] is None and not column.nullable:
            raise CellError(f"{table.name}.{column.name}: NULL in a NOT NULL column")
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
