"""Shared fixtures. The component tier (`-m component`) gets a fresh, empty Postgres 16 database from `pg_dsn`."""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def pg_dsn() -> Iterator[str]:
    """A fresh database: a new one on the server in `HARTLAND_TEST_PG` (CI's service container), dropped afterwards;
    without it, a throwaway testcontainers `postgres:16` (Docker needed).

    The database is shared by every component test of the session — a test that needs isolation works in a schema
    of its own and drops it."""
    server = os.environ.get("HARTLAND_TEST_PG")
    if server:
        name = f"hartland_test_{secrets.token_hex(4)}"
        with psycopg.connect(server, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield make_conninfo(server, dbname=name)
        finally:
            with psycopg.connect(server, autocommit=True) as conn:
                conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        return

    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer("postgres:16", driver=None) as pg:
        yield pg.get_connection_url()
