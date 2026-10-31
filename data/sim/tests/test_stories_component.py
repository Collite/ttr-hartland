"""G0 S0.4 T3, the component tier: every detector and probe runs on the schema of data/ddl/ (empty), so each names
only real tables and columns and is valid Postgres 16 — G5 S5.2 then judges their numbers on the build."""

from __future__ import annotations

from decimal import Decimal

import psycopg
import pytest

from hartland_sim.config import STORIES_DIR, InventoryConfig, StoriesConfig

pytestmark = pytest.mark.component

STORIES = StoriesConfig.load()
FILES = sorted(
    {detector.sql for story in STORIES.stories for detector in story.detectors}
    | {anomaly.probe.sql for anomaly in STORIES.anomalies}
)


@pytest.fixture(scope="module")
def db(hartland_db: psycopg.Connection) -> psycopg.Connection:
    for view in InventoryConfig.load().views:
        if view.kind == "materialized_view":
            hartland_db.execute(f"REFRESH MATERIALIZED VIEW hartland.{view.name}")
    return hartland_db


@pytest.mark.parametrize("name", FILES)
def test_it_runs_on_the_schema_and_returns_one_number(db: psycopg.Connection, name: str) -> None:
    sql = (STORIES_DIR / name).read_text(encoding="utf-8")
    with db.transaction(force_rollback=True):
        db.execute("SET LOCAL search_path TO hartland")
        cursor = db.execute(sql)
        assert cursor.description is not None and len(cursor.description) == 1, name
        rows = cursor.fetchall()
    assert len(rows) == 1, name
    (value,) = rows[0]
    assert value is None or isinstance(value, int | float | Decimal), name
