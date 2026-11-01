"""The rows setup and the loop write: one dict per row with exactly its table's columns (`data/inventory.yaml`), in
the inventory's order, so the CSV writer and the load never meet a missing or a stray column. `Tables` collects them
by table, in the order they were made."""

from __future__ import annotations

from collections import defaultdict
from functools import cache
from typing import Any

from hartland_sim.config import InventoryConfig

Row = dict[str, Any]
Tables = defaultdict[str, list[Row]]


@cache
def _columns() -> dict[str, tuple[str, ...]]:
    return {table.name: tuple(column.name for column in table.columns) for table in InventoryConfig.load().tables}


def columns(table: str) -> tuple[str, ...]:
    """`table`'s columns, in the inventory's order (the inventory is read once)."""
    return _columns()[table]


def row(table: str, **values: Any) -> Row:
    """A row of `table` from its columns by name; `KeyError` naming what is missing or not the table's."""
    names = columns(table)
    missing = [name for name in names if name not in values]
    stray = sorted(set(values) - set(names))
    if missing or stray:
        raise KeyError(f"{table}: missing {missing}, not its columns {stray}")
    return {name: values[name] for name in names}


def tables() -> Tables:
    return defaultdict(list)
