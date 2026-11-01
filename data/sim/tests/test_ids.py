"""Row ids from business keys (contracts §2; G1 S1.1 T2)."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date

import numpy as np
import pytest

from hartland_sim.config import InventoryConfig
from hartland_sim.ids import business_id, row_id

BIGINT = (-(2**63), 2**63 - 1)


def test_an_id_is_a_signed_bigint_stable_across_processes() -> None:
    here = business_id("customer", "OSM:123456")
    code = "from hartland_sim.ids import business_id; print(business_id('customer', 'OSM:123456'))"
    for seed in ("0", "1", "random"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=os.environ | {"PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        assert int(out.stdout) == here
    assert BIGINT[0] <= here <= BIGINT[1]


def test_another_key_or_another_table_gives_another_id() -> None:
    assert business_id("customer", "OSM:123456") != business_id("customer", "OSM:123457")
    assert business_id("customer", "OSM:123456") != business_id("store", "OSM:123456")


def test_ids_take_both_signs() -> None:
    ids = [business_id("customer", f"C{n}") for n in range(200)]
    assert min(ids) < 0 < max(ids)
    assert len(set(ids)) == 200


def test_row_id_reads_the_tables_id_key_from_the_inventory() -> None:
    assert row_id("customer", code="RT-000001") == business_id("customer", "RT-000001")
    assert row_id("store", osm_id="node/1") == business_id("store", "node/1")
    two = row_id("delivery_point", customer="RT-000001", place="node/1")
    assert two == row_id("delivery_point", place="node/1", customer="RT-000001")  # by name, not by order
    assert two != row_id("delivery_point", customer="RT-000001", place="node/2")


def test_row_id_parts_are_canonical() -> None:
    day = date(2024, 1, 1)
    assert row_id("lifecycle_phase", customer=np.int64(7), valid_from=day) == row_id(
        "lifecycle_phase", customer=7, valid_from=day
    )
    assert row_id("lifecycle_phase", customer=7, valid_from=day) != row_id(
        "lifecycle_phase", customer=7, valid_from=date(2024, 2, 1)
    )
    # an empty part (an assignment without a target of that kind) is a part
    assert row_id("channel_listing", channel="web", chain="", sku="X", valid_from=day) != row_id(
        "channel_listing", channel="web", chain="X", sku="", valid_from=day
    )


def test_row_id_refuses_a_key_that_is_not_the_inventorys() -> None:
    with pytest.raises(KeyError, match=r"customer's id_key is \['code'\], not \['name'\]"):
        row_id("customer", name="x")
    with pytest.raises(KeyError, match="delivery_point's id_key"):
        row_id("delivery_point", customer="x")
    with pytest.raises(KeyError, match="calendar has no id_key"):
        row_id("calendar", day=date(2024, 1, 1))
    with pytest.raises(ValueError, match="separator"):
        row_id("delivery_point", customer="a\x1fb", place="c")


def test_every_table_with_an_id_key_has_one_id_column_its_pk() -> None:
    for table in InventoryConfig.load().tables:
        if table.id_key is not None:
            (pk,) = table.pk
            assert table.column(pk).type == "bigint", table.name
