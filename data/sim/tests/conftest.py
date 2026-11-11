"""Shared fixtures: the committed reference files (`ref`, the okres table and its parts); the OSM market and the
opening world setup builds from the committed inputs (`osm_market`, `opening`, one run per session); for the build
tier (`-m build`), the story and the null world built to 2025 (`story_world`, `null_world`, once a session); and, for
the component tier (`-m component`), a fresh, empty Postgres 16 database from `pg_dsn`, the same with data/ddl/ applied
(`hartland_db`), and `put`, which writes one hand-built row into it; and S6's hand-built case (`s6_case`), the rows
both its twin and its SQL read (G1 S1.5 T1), with each way it leaves its agreement out (`s6_left_out`, a parameter)."""

from __future__ import annotations

import csv
import itertools
import os
import secrets
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, ClassVar

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from hartland_sim.config import DDL_DIR, Column, InventoryConfig, StoriesConfig
from hartland_sim.ids import row_id
from hartland_sim.io.ddl import DOCWH, OWNER, READONLY
from hartland_sim.io.reference import read
from hartland_sim.loop.build import Built, build
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


BUILT_UNTIL = date(2025, 12, 31)  # the build tier's worlds: S1's window is 2024–2025, S4's 2025


@pytest.fixture(scope="session")
def story_world(tmp_path_factory: pytest.TempPathFactory, osm_market: Market) -> Built:
    """The build tier's world (`-m build`): `build-world` from the warm-up to `BUILT_UNTIL`, once a session."""
    return build(tmp_path_factory.mktemp("story-world") / "run", BUILT_UNTIL, market=osm_market)


@pytest.fixture(scope="session")
def null_world(tmp_path_factory: pytest.TempPathFactory, osm_market: Market) -> Built:
    """The build tier's null world: every story and planted case off, to `BUILT_UNTIL`."""
    return build(tmp_path_factory.mktemp("null-world") / "run", BUILT_UNTIL, null=True, market=osm_market)


Rows = dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class S6Case:
    """`s6_case`: the rows by table, and what made them."""

    tables: Rows
    start: date  # each agreement's first day
    end: date  # and its last
    pct: float  # its discount, %
    elasticity: Mapping[int, float]  # by customer: the lift of its agreed line is 1 + e × pct ÷ 100
    margin: Mapping[int, float]  # by customer: the agreed line's Customer Margin a case at list
    daily: float  # the agreed line's cases a day, unlifted

    # T1(b), one case each: how customer 1's agreement is left out → the twin's reason
    LEFT_OUT: ClassVar[Mapping[str, str]] = {
        "no order before": "no order",
        "an agreement ending before": "before",
        "an agreement starting after": "after",
        "past the history": "history",
        "no cases of the line before": "no cases",
        "no cases of the others in the window": "no cases",
    }

    def leave_out(self, case: str) -> None:
        """Change the rows so that customer 1's agreement is left out as `case` (`LEFT_OUT`) says: no order of its
        in the 8 weeks before; another agreement of its line ending in them, or starting within 14 days after the end;
        the history ending before the fortnight after it (every order after its last day + 13 dropped, both
        customers'); its line's cases before, or its other lines' in the window, dropped."""
        before, window = (self.start - timedelta(days=56), self.start), (self.start, self.end + timedelta(days=15))

        def days(order: dict[str, Any], span: tuple[date, date], customer: int | None = 1) -> bool:
            return customer in (None, order["customer_id"]) and span[0] <= order["ordered_on"] < span[1]

        orders = {o["sales_order_id"]: o for o in self.tables["sales_order"]}
        agreement = self.tables["price_agreement"][0] | {"price_agreement_id": 3}
        if case == "no order before":
            gone = {n for n, o in orders.items() if days(o, before)}
        elif case == "past the history":
            gone = {n for n, o in orders.items() if days(o, (self.end + timedelta(days=14), date.max), None)}
        else:
            gone = set()
        if case == "an agreement ending before":
            first = self.start - timedelta(days=90)
            self.tables["price_agreement"].append(agreement | {"valid_from": first, "valid_to": before[0]})
        elif case == "an agreement starting after":
            first = self.end + timedelta(days=14)
            self.tables["price_agreement"].append(agreement | {"valid_from": first, "valid_to": first})
        lines = self.tables["sales_order_line"]
        if case == "no cases of the line before":
            lines[:] = [r for r in lines if not (r["sku_id"] == 1 and days(orders[r["sales_order_id"]], before))]
        elif case == "no cases of the others in the window":
            lines[:] = [r for r in lines if not (r["sku_id"] != 1 and days(orders[r["sales_order_id"]], window))]
        self.tables["sales_order"] = [o for n, o in orders.items() if n not in gone]
        lines[:] = [r for r in lines if r["sales_order_id"] not in gone]
        kept = {r["sales_order_line_id"] for r in lines}
        discounts = self.tables["sales_order_line_discount"]
        discounts[:] = [r for r in discounts if r["sales_order_line_id"] in kept]

    def write(self, run: Path) -> None:
        """The tables as a CSV build's files (the CSV writer's format: ISO dates, `true`/`false`, empty = NULL)."""
        run.mkdir(parents=True, exist_ok=True)
        for table, rows in self.tables.items():
            if not rows:
                continue
            with (run / f"{table}.csv").open("w", encoding="utf-8", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: _cell(v) for k, v in row.items()})


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value.isoformat() if isinstance(value, date) else str(value)


def a6_case(seen: int = 5, lift: float = 1.5, dip: float = 0.88) -> Rows:
    """G2 S2.3 T7's A-6 arc, hand-built (no build run): the rows `a6.sql` and its twin read. Two regions — A-6's
    (store 1, customer 1) and JMV (store 2, customer 2), a visit at each every Monday — `seen` (at most 6) Kavalír
    promotions at visits in A-6's from its sightings on; Gold's competitor facings beside it 10 in the 8 weeks
    before, 10 × `lift` from the sightings on; each customer 6 cases of Gold a week, A-6's × `dip` from its `from`
    (the season the same in both)."""
    held = StoriesConfig.load().anomaly("A-6").parameters
    first, start, last = (date.fromisoformat(str(held[key])) for key in ("sightings_from", "from", "until"))
    gold, kavalir = 1, row_id("competitor", code=held["competitor"])  # the twin reads the reference's id
    rows: Rows = {
        "region": [{"region_id": 1, "code": held["region"]}, {"region_id": 2, "code": "JMV"}],
        "area": [{"area_id": 1, "region_id": 1}, {"area_id": 2, "region_id": 2}],
        "okres": [{"okres_id": 1}],
        "area_okres": [{"area_id": 1, "okres_id": 1}],
        "store": [{"store_id": 1, "area_id": 1}, {"store_id": 2, "area_id": 2}],
        "customer": [{"customer_id": 1, "store_id": 1}, {"customer_id": 2, "store_id": 2}],
        "delivery_point": [{"delivery_point_id": n, "customer_id": n, "store_id": n, "okres_id": 1} for n in (1, 2)],
        "product_line": [{"product_line_id": gold, "code": held["line"]}],
        "product": [{"product_id": gold, "product_line_id": gold}],
        "sku": [{"sku_id": gold, "product_id": gold}],
        "competitor": [{"competitor_id": kavalir, "code": held["competitor"]}],
        "visit": [],
        "visit_observation": [],
        "competitor_sighting": [],
        "sales_order": [],
        "sales_order_line": [],
        "sales_order_line_discount": [],
    }
    ids = itertools.count(1000)
    day = first - timedelta(days=56)
    while day <= last:  # a visit at each store every Monday, observing Gold
        for store in (1, 2):
            visit = next(ids)
            rows["visit"].append({"visit_id": visit, "store_id": store, "customer_id": store, "day": day})
            facings = 10 * (lift if store == 1 and day >= first else 1)
            rows["visit_observation"].append(
                {"visit_id": visit, "product_line_id": gold, "competitor_facings": round(facings)}
            )
            if store == 1 and day >= first and seen > 0:
                seen -= 1
                rows["competitor_sighting"].append(
                    {"competitor_sighting_id": visit, "visit_id": visit, "competitor_id": kavalir}
                    | {"activity_kind": "promotion"}
                )
        day += timedelta(days=7)
    day = start - timedelta(days=56)
    while day <= last:  # each customer's Gold order every Wednesday
        for customer in (1, 2):
            order = next(ids)
            cases = 6.0 * (dip if customer == 1 and day >= start else 1)
            rows["sales_order"].append(
                {"sales_order_id": order, "customer_id": customer, "delivery_point_id": customer}
                | {"channel": "small_retail", "ordered_on": day + timedelta(days=2)}
            )
            rows["sales_order_line"].append(
                {"sales_order_line_id": order, "sales_order_id": order, "sku_id": gold, "qty_cases": round(cases, 3)}
                | {"is_free_goods": False, "list_price_case": 1000.0}
            )
        day += timedelta(days=7)
    return rows


def write_rows(run: Path, rows: Rows) -> None:
    """`rows` as a CSV build's files (the CSV writer's format); an empty table under the inventory's header."""
    run.mkdir(parents=True, exist_ok=True)
    inventory = InventoryConfig.load()
    for table, found in rows.items():
        names = list(found[0]) if found else [column.name for column in inventory.table(table).columns]
        with (run / f"{table}.csv").open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=names, lineterminator="\n")
            writer.writeheader()
            for row in found:
                writer.writerow({k: _cell(v) for k, v in row.items()})


A6Case = Callable[..., Rows]  # `a6_case`
WriteRows = Callable[..., None]  # `write_rows`


@pytest.fixture(scope="session")
def a6_rows() -> tuple[A6Case, WriteRows]:
    """`a6_case` and `write_rows`, for the CSV and the component tier alike (the tests are not a package)."""
    return a6_case, write_rows


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """`s6_left_out`: every case of `S6Case.LEFT_OUT`, so the CSV and the component tier run the same ones
    (review-225 R3)."""
    if "s6_left_out" in metafunc.fixturenames:
        metafunc.parametrize("s6_left_out", list(S6Case.LEFT_OUT))


@pytest.fixture
def s6_case() -> S6Case:
    """G1 S1.5 T1's S6 case, hand-built (no build run): the rows the S6 twin and `s6.sql` read, fresh for each test.

    Two small-retail customers, each with five lines (1–5) and an order every 14 days: line 1 10 cases an order at
    1 000 CZK a case, lines 2–5 40 cases each at 500. Each has an agreed discount on line 1 that starts mid-cycle
    (2025-03-10, 7 days after an order), runs 45 days at 6 % and lifts the line's daily demand by 1 + e × 0.06 from its
    first day: customer 1 of SJC-02's rep (disciplined), e = 3.5, a delivery cost of 30 and a warehouse cost of 20 a
    case; customer 2 of SMS-02's (undisciplined), e = 0.4, no costs (NULL). An order carries the demand of the 14 days
    before it, so the first order inside the agreement carries 7 days of unlifted demand at the discount, and the first
    after it the lifted remainder at list. The orders run from 9 weeks before the agreement to 11 weeks after its
    start, past its fortnight after."""
    start, days, pct = date(2025, 3, 10), 45, 6.0
    elasticity, costs = {1: 3.5, 2: 0.4}, {1: (30.0, 20.0), 2: (None, None)}
    price, daily = {1: 1000.0, 2: 500.0, 3: 500.0, 4: 500.0, 5: 500.0}, 10 / 14
    tables: Rows = {
        "product_line": [{"product_line_id": n, "code": f"L{n}"} for n in price],
        "product": [{"product_id": n, "product_line_id": n} for n in price],
        "sku": [{"sku_id": n, "product_id": n} for n in price],
        "okres": [{"okres_id": 1}],
    }
    for table in ("area", "assignment", "customer", "delivery_point", "price_agreement", "sales_order"):
        tables[table] = []
    tables["sales_order_line"], tables["sales_order_line_discount"] = [], []
    end = start + timedelta(days=days - 1)
    for n, slot in ((1, "SJC-02"), (2, "SMS-02")):
        tables["area"].append({"area_id": n, "code": slot})
        tables["assignment"].append(
            {"assignment_id": n, "employee_id": n, "kind": "rep_area", "area_id": n}
            | {"valid_from": date(2023, 1, 1), "valid_to": date(9999, 12, 31)}
        )
        tables["customer"].append({"customer_id": n, "channel": "small_retail"})
        tables["delivery_point"].append({"delivery_point_id": n, "customer_id": n, "store_id": None, "okres_id": 1})
        tables["price_agreement"].append(
            {"price_agreement_id": n, "customer_id": n, "product_line_id": 1, "kind": "agreed_discount"}
            | {"discount_pct": pct, "valid_from": start, "valid_to": end, "agreed_by_employee_id": n}
        )
        lift = 1 + elasticity[n] * pct / 100
        delivery, warehouse = costs[n]
        for k in range(-5, 6):  # day −63 … 77 of the agreement
            day = start + timedelta(days=14 * k + 7)
            order = 100 * n + k + 5
            tables["sales_order"].append(
                {"sales_order_id": order, "customer_id": n, "delivery_point_id": n, "channel": "small_retail"}
                | {"ordered_on": day}
            )
            carried = [day - timedelta(days=back) for back in range(1, 15)]
            agreed = start <= day <= end
            for line, list_price in price.items():
                cases = 40.0
                if line == 1:
                    cases = round(daily * sum(lift if start <= d <= end else 1.0 for d in carried), 3)
                off = round(list_price * pct / 100, 2) if line == 1 and agreed else 0.0
                tables["sales_order_line"].append(
                    {"sales_order_line_id": 10 * order + line, "sales_order_id": order, "line_no": line}
                    | {"sku_id": line, "qty_cases": cases, "list_price_case": list_price}
                    | {"invoice_price_case": list_price - off, "purchase_price_case": 0.45 * list_price}
                    | {"delivery_cost_case": delivery, "warehouse_cost_case": warehouse, "is_free_goods": False}
                )
                if off:
                    tables["sales_order_line_discount"].append(
                        {"sales_order_line_id": 10 * order + line, "seq": 1, "kind": "agreed", "price_agreement_id": n}
                        | {"pct": pct, "amount_case": off, "amount_line": round(off * cases, 2)}
                    )
    margin = {n: price[1] - sum(cost or 0.0 for cost in costs[n]) for n in costs}
    return S6Case(tables, start, end, pct, elasticity, margin, daily)


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
