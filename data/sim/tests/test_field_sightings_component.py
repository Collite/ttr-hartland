"""A build's sightings load (G2 S2.6b T3, component; NH-D24): a story build whose loop writes from the first day (the
determinism battery's `EARLY`) holds real sightings by the 5th; it COPYs whole in `load_order` in one transaction, the
commit's foreign keys hold and each sighting names one of its competitor's products. Without `competitor_product` the
same load is refused at the commit, not skipped."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from hartland_sim.config import InventoryConfig, StoriesConfig, WorldConfig
from hartland_sim.loop.build import build
from hartland_sim.setup.market import Market

pytestmark = pytest.mark.component

WORLD = WorldConfig.load()
EARLY = replace(WORLD, history=replace(WORLD.history, first_day=date(2023, 1, 1)))  # the loop writes from day one


def _load(db: psycopg.Connection, run: Path, without: str = "") -> None:
    with db.transaction():
        for table in InventoryConfig.load().tables:
            path = run / f"{table.name}.csv"
            if path.exists() and table.name != without:
                statement = sql.SQL("COPY hartland.{} FROM STDIN (FORMAT csv, HEADER true)")
                with db.cursor() as cursor, cursor.copy(statement.format(sql.Identifier(table.name))) as copy:
                    copy.write(path.read_bytes())


def test_a_build_s_sightings_load_with_the_competitors_they_name(
    osm_market: Market, hartland_db: psycopg.Connection, tmp_path: Path
) -> None:
    run = build(tmp_path / "run", date(2023, 1, 5), EARLY, StoriesConfig.load(), market=osm_market).run
    with pytest.raises(psycopg.errors.ForeignKeyViolation, match="competitor_sighting_competitor_product_id_fkey"):
        _load(hartland_db, run, without="competitor_product")
    _load(hartland_db, run)
    found = hartland_db.execute(
        "SELECT (SELECT count(*) FROM hartland.competitor), (SELECT count(*) FROM hartland.competitor_product),"
        " count(*), count(*) FILTER (WHERE p.competitor_id = s.competitor_id) FROM hartland.competitor_sighting s"
        " JOIN hartland.competitor_product p USING (competitor_product_id)"
    ).fetchone()
    assert found is not None and found[:2] == (4, 40)
    assert found[2] == found[3] > 0  # real sightings, each of a product of the competitor it names
