"""The opening loads (G1 S1.1 T7, component): every setup CSV COPYs into the `hartland` schema in `load_order`, inside
one transaction, and the commit's deferred foreign keys hold — the seam `load-world` (G5 S5.1) builds on."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from hartland_sim.config import InventoryConfig, StoriesConfig, WorldConfig
from hartland_sim.setup.opening import write
from hartland_sim.setup.run import Setup

pytestmark = pytest.mark.component


def test_the_opening_copies_into_postgres_and_its_keys_hold(
    opening: Setup, hartland_db: psycopg.Connection, tmp_path: Path
) -> None:
    run = tmp_path / "opening"
    write(opening.world, opening.tables, run, WorldConfig.load(), StoriesConfig.load())
    inventory = InventoryConfig.load()
    counts: dict[str, int] = {}
    with hartland_db.transaction():
        for table in inventory.tables:
            path = run / f"{table.name}.csv"
            if not path.exists():
                continue
            statement = sql.SQL("COPY hartland.{} FROM STDIN (FORMAT csv, HEADER true)").format(
                sql.Identifier(table.name)
            )
            with hartland_db.cursor() as cursor, cursor.copy(statement) as copy:
                copy.write(path.read_bytes())
            (counts[table.name],) = hartland_db.execute(
                sql.SQL("SELECT count(*) FROM hartland.{}").format(sql.Identifier(table.name))
            ).fetchone() or (0,)
    assert counts == {name: len(rows) for name, rows in {**opening.tables, **opening.state}.items()}
    opening_day = WorldConfig.load().history.warm_up_from - timedelta(days=1)
    (day,) = hartland_db.execute("SELECT DISTINCT day FROM hartland.stock_day").fetchone() or (None,)
    assert day == opening_day
    leaf = hartland_db.execute(
        "SELECT count(*) FROM hartland.delivery_point dp JOIN hartland.okres o USING (okres_id)"
        " WHERE EXISTS (SELECT FROM hartland.okres child WHERE child.parent_okres_id = o.okres_id)"
    ).fetchone()
    assert leaf == (0,)  # no point in Praha itself, only in its districts
