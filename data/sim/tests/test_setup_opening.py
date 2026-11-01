"""Setup step 7 and the writer (G1 S1.1 T7): the opening stock, the CSV writer, and the DONE of the stage — setup run
twice gives byte-identical files, the opening checkpoint round-trips."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import subprocess
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.config import InventoryConfig, StoriesConfig, WorldConfig
from hartland_sim.io.csv_writer import CellError, CsvWriter, cell
from hartland_sim.setup.opening import rates, write
from hartland_sim.setup.run import Setup
from hartland_sim.world import CHECKPOINT_FILE, META_FILE, Checkpoint, Meta

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
INVENTORY = InventoryConfig.load()
OPENING_DAY = WORLD.history.warm_up_from - timedelta(days=1)


@pytest.fixture(scope="module")
def written(opening: Setup, tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("build") / "opening"
    write(opening.world, opening.tables, run, WORLD, STORIES)
    return run


# ── the writer ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "kind", "text"),
    [
        (12, "bigint", "12"),
        (-(2**63), "bigint", str(-(2**63))),
        (3, "integer", "3"),
        (0.1 + 0.2, "numeric(14,2)", "0.30"),
        (10, "numeric(5,1)", "10.0"),
        (50.0870123, "numeric(9,6)", "50.087012"),
        (True, "boolean", "true"),
        (date(2024, 2, 29), "date", "2024-02-29"),
        ("06:00", "time", "06:00"),
        (time(7, 5), "time", "07:05"),
        (datetime(2024, 1, 1, 8, tzinfo=UTC), "timestamptz", "2024-01-01T08:00:00+00:00"),
        ({"b": 1, "a": "č"}, "jsonb", '{"a":"č","b":1}'),
        ("Kolín", "text", "Kolín"),
        (None, "text", None),
    ],
)
def test_a_cell_is_its_value_in_its_type_s_one_spelling(value: Any, kind: str, text: str | None) -> None:
    assert cell(value, kind) == text


@pytest.mark.parametrize(
    ("value", "kind"),
    [(True, "integer"), (1.5, "bigint"), ("1", "numeric(14,2)"), (1, "boolean"), (datetime(2024, 1, 1), "date")]
    + [(datetime(2024, 1, 1), "timestamptz"), ("6:00", "time"), ("", "text"), (3, "text")],
)
def test_a_value_its_type_cannot_hold_is_refused(value: Any, kind: str) -> None:
    with pytest.raises(CellError):
        cell(value, kind)


def test_the_writer_appends_under_one_header_and_refuses_a_null_in_a_not_null_column(tmp_path: Path) -> None:
    row = {"day": date(2024, 1, 1), "year": 2024, "quarter": 1, "month": 1, "week": 1, "iso_year": 2024}
    row |= {"weekday": 1, "weekday_cs": "pondělí", "is_working_day": False, "holiday_cs": "Nový rok", "season": "zima"}
    with CsvWriter(tmp_path) as writer:
        writer.append("calendar", [row])
    with CsvWriter(tmp_path) as writer:
        writer.append("calendar", [row | {"day": date(2024, 1, 2), "holiday_cs": None}])
    lines = (tmp_path / "calendar.csv").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "day,year,quarter,month,week,iso_year,weekday,weekday_cs,is_working_day,holiday_cs,season",
        "2024-01-01,2024,1,1,1,2024,1,pondělí,false,Nový rok,zima",
        "2024-01-02,2024,1,1,1,2024,1,pondělí,false,,zima",
    ]
    with CsvWriter(tmp_path) as writer, pytest.raises(CellError, match=r"calendar\.season: NULL"):
        writer.append("calendar", [row | {"season": None}])
    with CsvWriter(tmp_path) as writer, pytest.raises(KeyError, match="not in the inventory"):
        writer.write_all({"no_such_table": []})


def test_write_all_goes_in_load_order(opening: Setup, tmp_path: Path) -> None:
    with CsvWriter(tmp_path) as writer:
        names = writer.write_all(opening.tables)
    orders = [INVENTORY.table(name).load_order for name in names]
    assert orders == sorted(orders) and set(names) == set(opening.tables)


# ── the opening ──────────────────────────────────────────────────────────────


def test_the_opening_stock_sits_at_each_policy_s_midpoint(opening: Setup) -> None:
    first = WORLD.history.warm_up_from
    rate = rates(WORLD, opening.portfolio, opening.customers, opening.geography, first)
    rows = opening.tables["stock_day"]
    live = [sku for sku in opening.portfolio.skus if sku.launched_on <= first <= sku.last_day]
    assert len(rows) == 2 * len(live)
    for item in rows:
        key = (item["sku_id"], item["dc_site_id"])
        low, high = opening.covers[key]
        assert item["day"] == OPENING_DAY
        assert item["on_hand_cases"] == math.ceil(rate[key] * (low + high) / 2) > 0
        assert item["days_cover"] == (low + high) / 2
        assert (item["reserved_cases"], item["in_transit_cases"], item["below_min"], item["above_max"]) == (
            0,
            0,
            False,
            False,
        )
        assert opening.world.stock[key].on_hand == item["on_hand_cases"]
    world = opening.world
    assert not (world.open_orders or world.open_shipments or world.open_invoices or world.loans or world.promotions)


def test_every_setup_table_is_written_with_the_inventory_s_header(opening: Setup, written: Path) -> None:
    for name, rows in opening.tables.items():
        with (written / f"{name}.csv").open(encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            assert next(reader) == [column.name for column in INVENTORY.table(name).columns]
            assert sum(1 for _ in reader) == len(rows), name
    assert {path.stem for path in written.glob("*.csv")} == set(opening.tables) | {"_sim_meta", "_sim_checkpoint"}


def test_the_meta_names_the_inputs(written: Path) -> None:
    meta = Meta.load(written)
    assert meta.master_seed == WORLD.master_seed
    assert meta.stories_sha256 == STORIES.sha256
    assert meta.world_sha256 == hashlib.sha256((Path(__file__).parents[2] / "world.yaml").read_bytes()).hexdigest()
    assert meta.last_day_done == OPENING_DAY
    assert len(meta.build_id) == 16


def test_the_opening_checkpoint_round_trips(opening: Setup, written: Path) -> None:
    checkpoint = Checkpoint.load(written)
    assert checkpoint.day == OPENING_DAY
    assert checkpoint.restore() == opening.world


def test_a_run_folder_is_written_once(opening: Setup, written: Path) -> None:
    with pytest.raises(FileExistsError):
        write(opening.world, opening.tables, written, WORLD, STORIES)


def _hashes(run: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(run.iterdir())}


def test_setup_run_twice_writes_byte_identical_files(written: Path, tmp_path: Path) -> None:
    """The stage's DONE: a fresh process, another hash seed, the OSM market rebuilt — the same bytes, file by file."""
    again = tmp_path / "again"
    code = (
        "import sys; from pathlib import Path; from hartland_sim.config import StoriesConfig, WorldConfig; "
        "from hartland_sim.setup.opening import write; from hartland_sim.setup.run import run; "
        "s = run(); write(s.world, s.tables, Path(sys.argv[1]), WorldConfig.load(), StoriesConfig.load())"
    )
    subprocess.run([sys.executable, "-c", code, str(again)], env=os.environ | {"PYTHONHASHSEED": "12345"}, check=True)
    assert _hashes(again) == _hashes(written)
    assert {CHECKPOINT_FILE, META_FILE} <= set(_hashes(written))
