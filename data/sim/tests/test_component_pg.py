"""The component tier's own smoke test: `pg_dsn` gives an empty Postgres 16 database (G0 S0.4 applies the DDL on it)."""

import psycopg
import pytest

pytestmark = pytest.mark.component


def test_a_fresh_postgres_16(pg_dsn: str) -> None:
    with psycopg.connect(pg_dsn) as conn:
        assert conn.info.server_version // 10000 == 16
        tables = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
        ).fetchone()
        assert tables == (0,)
