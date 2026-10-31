"""Shared fixtures: the committed reference files (`ref`, the okres table and its parts); and, for the component
tier (`-m component`), a fresh, empty Postgres 16 database from `pg_dsn`."""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from hartland_sim.io.reference import read

REPO_ROOT = Path(__file__).resolve().parents[3]

PRAHA = "CZ0100"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def ref(repo_root: Path) -> Path:
    """The committed reference files, `data/reference/`."""
    return repo_root / "data" / "reference"


@pytest.fixture(scope="session")
def okres(ref: Path) -> dict[str, dict[str, Any]]:
    """`okres.csv` by code: the 77 okresy and the 22 Prague districts."""
    return {row["code"]: row for row in read("okres", ref)}


@pytest.fixture(scope="session")
def leaves(okres: dict[str, dict[str, Any]]) -> list[str]:
    """The okresy a customer can sit in: every okres but Praha, which its 22 districts replace."""
    parents = {row["parent_okres"] for row in okres.values()}
    return [code for code in okres if code not in parents]


@pytest.fixture(scope="session")
def prague_districts(okres: dict[str, dict[str, Any]]) -> list[str]:
    """The 22 Prague districts (`parent_okres` = Praha, `CZ0100`), in order."""
    return [code for code, row in okres.items() if row["parent_okres"] == PRAHA]


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
