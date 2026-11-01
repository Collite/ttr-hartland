"""Shared fixtures: the committed reference files (`ref`, the okres table and its parts); the OSM market and the
opening world setup builds from the committed inputs (`osm_market`, `opening`, one run per session); and, for the
component tier (`-m component`), a fresh, empty Postgres 16 database from `pg_dsn`, the same with data/ddl/ applied
(`hartland_db`), and `put`, which writes one hand-built row into it."""

from __future__ import annotations

import itertools
import os
import secrets
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from hartland_sim.config import DDL_DIR, Column, InventoryConfig
from hartland_sim.io.ddl import DOCWH, OWNER, READONLY
from hartland_sim.io.reference import read
from hartland_sim.setup import market as market_setup
from hartland_sim.setup.market import Market
from hartland_sim.setup.run import Setup, run

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
def osm_market() -> Market:
    """The market of the committed OSM extract (`setup.market`): stores, chains, the HoReCa sample."""
    return market_setup.build()


@pytest.fixture(scope="session")
def opening(osm_market: Market) -> Setup:
    """The opening world and setup's rows, from the committed inputs (G1 S1.1). Read it, never change it."""
    return run(market=osm_market)


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


@pytest.fixture(scope="module")
def hartland_db(pg_dsn: str) -> Iterator[psycopg.Connection]:
    """`pg_dsn`'s database with data/ddl/'s schema.sql, indexes.sql and grants applied, as `load-world` will (G5 S5.1),
    the roles of contracts §8 created when missing; the schema is dropped after the module."""
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        for role in (OWNER, READONLY, DOCWH):
            conn.execute(
                f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') "
                f"THEN CREATE ROLE {role}; END IF; END $$"
            )
        for name in ("schema.sql", "indexes.sql", "views/grants.sql"):
            conn.execute((DDL_DIR / name).read_text(encoding="utf-8"))
        try:
            yield conn
        finally:
            conn.execute("DROP SCHEMA hartland CASCADE")


Put = Callable[..., None]


@pytest.fixture(scope="session")
def put() -> Put:
    """`put(db, "store", store_id=2, area_id=3)` inserts one row of `hartland.store`. Every NOT NULL column not given
    gets a filler (a coded column its first code, a text its name and a running number, a number that number), so a
    component case writes only what its SQL reads. Use it inside a transaction that rolls back: the foreign keys are
    deferred, so the rows a case leaves out are never missed."""
    inventory = InventoryConfig.load()
    count = itertools.count(1_000_000)

    def insert(db: psycopg.Connection, table: str, **values: Any) -> None:
        columns = inventory.table(table).columns
        unknown = set(values) - {column.name for column in columns}
        assert not unknown, f"{table} has no {sorted(unknown)}"
        row = dict(values)
        for column in columns:
            if column.name not in row and not column.nullable:
                row[column.name] = _filler(column, inventory.codes, next(count))
        db.execute(
            sql.SQL("INSERT INTO hartland.{} ({}) VALUES ({})").format(
                sql.Identifier(table),
                sql.SQL(", ").join(sql.Identifier(name) for name in row),
                sql.SQL(", ").join(sql.Placeholder() for _ in row),
            ),
            list(row.values()),
        )

    return insert


def _filler(column: Column, codes: Mapping[str, Mapping[str, str]], n: int) -> Any:
    if column.coded is not None:
        return next(iter(codes[column.coded]))
    fillers: dict[str, Any] = {
        "bigint": n,
        "integer": n,
        "boolean": False,
        "date": date(2024, 1, 1),
        "timestamptz": datetime(2024, 1, 1, tzinfo=UTC),
        "time": time(8),
        "text": f"{column.name}-{n}",
        "jsonb": Jsonb({}),
    }
    return fillers.get(column.type, 0)  # numeric(p,s)
