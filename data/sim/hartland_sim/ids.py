"""Row ids from business keys (contracts §2).

Every id is `blake2b(business key, digest_size=8)` read as a signed bigint, so a rebuild with the same inputs gives
the same ids and an id never depends on generation order. The business key of each table is its `id_key` in
`data/inventory.yaml`. G1 S1.1 T2.
"""

from __future__ import annotations


def business_id(table: str, key: str) -> int:
    """The id of `table`'s row whose business key is `key` (e.g. `business_id("customer", "OSM:123456")`)."""
    raise NotImplementedError("G1 S1.1 T2")


def row_id(table: str, **key: object) -> int:
    """The id of `table`'s row from the named parts of its `id_key` (`data/inventory.yaml`)."""
    raise NotImplementedError("G1 S1.1 T2")
