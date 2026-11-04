"""The determinism battery's measures (`05` §7, G1 S1.4 T5). A CSV build's identity is its tables, never its
checkpoint: a restored `World` pickles equal, not byte-identical (⚑R201-B, `world.py`).

- `hashes(run)`: the sha256 of each CSV of a build by its table, so two builds of the same inputs compare file by file.
- `closed(run, day)`: the sha256 of each table's rows closed by `day` — what a prolong to `day` must give as a straight
  build to any later day does. Those are setup's tables whole, and of the loop's:
  - the orders of `ordered_on` ≤ `day`, with their lines and their discounts;
  - the agreements, conditions, price lists (with their prices) and weather from a day ≤ `day`;
  - the plan rows made by `day` (a row of month M, version v is made on the day before M − (E − v) months, E the
    months the roll plans).
  The tables a build's end renders (the state tables, the promotions' status) and its own files (`_sim_meta`,
  `_sim_checkpoint`) are not closed documents. Each table's rows count as a sorted list of their text, so the order
  they were appended in does not.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

from hartland_sim.config import WorldConfig
from hartland_sim.dates import add_months
from hartland_sim.setup.opening import STATE_TABLES
from hartland_sim.world import CHECKPOINT_FILE, META_FILE

RENDERED = ("promotion", *STATE_TABLES)  # rendered at a build's end: its last day's, not a closed document
OWN = (CHECKPOINT_FILE, META_FILE)
Keep = Callable[[dict[str, str]], bool]


def hashes(run: Path) -> dict[str, str]:
    """Each CSV's sha256 by its file name."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(run.glob("*.csv"))}


def _digest(path: Path, keep: Keep | None = None, column: str | None = None) -> tuple[str, set[str]]:
    """The sha256 of `path`'s rows that `keep` keeps (module docstring), and their values of `column` — the ids the
    children's rows are kept by, collected in the same pass (none without a `column`)."""
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, [])
        rows = [row for row in reader if keep is None or keep(dict(zip(header, row, strict=True)))]
    out = hashlib.sha256(",".join(header).encode() + b"\n")
    for row in sorted(rows):
        out.update("\x1f".join(row).encode() + b"\n")
    at = header.index(column) if column is not None else None
    return out.hexdigest(), {row[at] for row in rows} if at is not None else set()


def _within(column: str, parents: set[str]) -> Keep:
    """A row whose `column` names one of `parents`."""
    return lambda row: row[column] in parents


def closed(run: Path, day: date, world: WorldConfig | None = None) -> dict[str, str]:
    """Each table's sha256 over its rows closed by `day` (module docstring), by table."""
    world = world or WorldConfig.load()
    last = day.isoformat()
    ahead = len(world.policies.sales_plan.error_pct)

    def made(row: dict[str, str]) -> bool:
        month = date.fromisoformat(row["month"])
        return add_months(month, int(row["version"]) - ahead) - timedelta(days=1) <= day

    by_date: dict[str, Keep] = {
        "sales_order": lambda row: row["ordered_on"] <= last,
        "price_agreement": lambda row: row["valid_from"] <= last,
        "customer_condition": lambda row: row["month"] <= last,
        "standard_price_list": lambda row: row["valid_from"] <= last,
        "channel_price_list": lambda row: row["valid_from"] <= last,
        "weather_daily": lambda row: row["day"] <= last,
        "sales_plan": made,
    }
    keys = {
        "sales_order": "sales_order_id",
        "standard_price_list": "standard_price_list_id",
        "channel_price_list": "channel_price_list_id",
    }
    out = {}
    ids: dict[str, set[str]] = {}
    for name, keep in by_date.items():
        path = run / f"{name}.csv"
        if path.is_file():
            out[name], ids[name] = _digest(path, keep, keys.get(name))
    children = {  # a child's rows by its parent's kept ids, and its own ids when a grandchild needs them
        "sales_order_line": ("sales_order", "sales_order_id", "sales_order_line_id"),
        "sales_order_line_discount": ("sales_order_line", "sales_order_line_id", None),
        "standard_price": ("standard_price_list", "standard_price_list_id", None),
        "channel_price": ("channel_price_list", "channel_price_list_id", None),
    }
    for name, (parent, column, key) in children.items():
        path = run / f"{name}.csv"
        if path.is_file() and (parent in by_date or parent in ids):  # discounts without the lines' file: whole, below
            out[name], ids[name] = _digest(path, _within(column, ids.get(parent, set())), key)
    for path in sorted(run.glob("*.csv")):
        name = path.stem
        if name not in out and name not in RENDERED and path.name not in OWN:
            out[name] = _digest(path)[0]
    return dict(sorted(out.items()))
