"""Row ids from business keys (contracts §2).

Every id is `blake2b(table ‖ business key, digest_size=8)` read as a signed bigint, so a rebuild with the same inputs
gives the same ids and an id never depends on generation order — except where the business key is a sequence:
customer codes number the market in order (`setup.customers`). The table is part of the hash, so a store and its
customer never share an id. The business key of each table is its `id_key` in `data/inventory.yaml`: names of the
key's parts, not columns (`delivery_point`: `customer`, `place`). `row_id(table, **key)` takes the parts by those
names; each is made canonical as `rng`'s key parts are (an `int` whatever numpy type, a date's ISO text), and the
parts are joined in `id_key`'s order by the unit separator, which no part may hold. A one-part key is its own text:
`row_id("store", osm_id="node/1") == business_id("store", "node/1")`.

At ~33 M rows the chance of any two ids colliding is a few in a million; the load's primary keys would refuse it.
"""

from __future__ import annotations

import hashlib
from functools import cache

from hartland_sim.config import inventory
from hartland_sim.rng import canonical

SEPARATOR = "\x1f"


def business_id(table: str, key: str) -> int:
    """The id of `table`'s row whose business key is `key` (e.g. `business_id("customer", "OSM:123456")`)."""
    digest = hashlib.blake2b(f"{table}{SEPARATOR}{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


@cache
def _id_keys() -> dict[str, tuple[str, ...] | None]:
    return {table.name: table.id_key for table in inventory().tables}


def row_id(table: str, **key: object) -> int:
    """The id of `table`'s row from the named parts of its `id_key` (`data/inventory.yaml`)."""
    names = _id_keys()[table]
    if names is None:
        raise KeyError(f"{table} has no id_key: its primary key is its columns")
    if set(key) != set(names):
        raise KeyError(f"{table}'s id_key is {list(names)}, not {sorted(key)}")
    parts = [str(canonical(key[name])) for name in names]
    if any(SEPARATOR in part for part in parts):
        raise ValueError(f"{table}: a key part holds the unit separator: {parts}")
    return business_id(table, SEPARATOR.join(parts))
