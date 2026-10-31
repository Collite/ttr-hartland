"""The component tier's own smoke test: `pg_dsn` gives a Postgres 16 database a test can create and drop in."""

import secrets

import psycopg
import pytest
from psycopg import sql

pytestmark = pytest.mark.component


def test_a_postgres_16_to_create_and_drop_in(pg_dsn: str) -> None:
    schema = sql.Identifier(f"smoke_{secrets.token_hex(4)}")
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        assert conn.info.server_version // 10000 == 16
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        try:
            conn.execute(sql.SQL("CREATE TABLE {}.probe (id bigint PRIMARY KEY)").format(schema))
            conn.execute(sql.SQL("INSERT INTO {}.probe VALUES (1)").format(schema))
            assert conn.execute(sql.SQL("SELECT count(*) FROM {}.probe").format(schema)).fetchone() == (1,)
        finally:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))
