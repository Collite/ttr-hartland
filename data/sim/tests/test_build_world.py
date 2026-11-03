"""`build-world` on the build tier (G1 S1.4 T4; ⚑R213-A): the story world to 2025-12-31 (`conftest.story_world`)
writes every table G1 writes; each written year holds 4 standard and 12 × 7 channel list versions, from January 2024 on
(the next quarter's and month's issued on the last day); a budget row per rep and KAM per quarter from 2024-Q1; grants
in every quarter, the refusals by reason in `build.json`, and no quarter refused wholesale for want of a budget."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from hartland_sim.config import WorldConfig
from hartland_sim.loop.build import BUILD_JSON, Built
from hartland_sim.loop.demand import CHANNELS

pytestmark = pytest.mark.build

WORLD = WorldConfig.load()
LOOP_TABLES = (
    "sales_order",
    "sales_order_line",
    "sales_order_line_discount",
    "price_agreement",
    "customer_condition",
    "sales_plan",
    "standard_price_list",
    "standard_price",
    "channel_price_list",
    "channel_price",
    "weather_daily",
)
STATE = ("employee", "assignment", "customer", "lifecycle_phase", "discount_budget")
QUARTERS = [date(year, month, 1) for year in (2024, 2025) for month in (1, 4, 7, 10)]


def _rows(run: Path, table: str) -> list[dict[str, str]]:
    with (run / f"{table}.csv").open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def test_every_table_g1_writes_has_its_rows(story_world: Built) -> None:
    for table in (*LOOP_TABLES, *STATE, "promotion", "campaign", "store", "delivery_point"):
        assert _rows(story_world.run, table), table
    assert not (story_world.run / "stock_day.csv").exists()  # G3's, from the first written day
    assert {"_sim_meta.csv", "_sim_checkpoint.csv", BUILD_JSON} <= {p.name for p in story_world.run.iterdir()}


def test_each_written_year_has_4_standard_and_84_channel_list_versions_none_before_2024(story_world: Built) -> None:
    standard = Counter(r["valid_from"][:4] for r in _rows(story_world.run, "standard_price_list"))
    channel = Counter(r["valid_from"][:4] for r in _rows(story_world.run, "channel_price_list"))
    assert (standard["2024"], standard["2025"], channel["2024"], channel["2025"]) == (4, 4, 84, 84)
    assert set(standard) == set(channel) == {"2024", "2025", "2026"}  # 2026's first, issued on 2025-12-31
    assert (standard["2026"], channel["2026"]) == (1, len(CHANNELS))
    first = min(r["valid_from"] for r in _rows(story_world.run, "standard_price_list"))
    assert first == "2024-01-01" and {r["code"] for r in _rows(story_world.run, "standard_price_list")} >= {
        "SC-2024-Q1",
        "SC-2025-Q2",
    }


def test_a_budget_row_per_rep_and_kam_per_quarter_from_2024(story_world: Built) -> None:
    budgets = _rows(story_world.run, "discount_budget")
    kams = {r["employee_id"] for r in _rows(story_world.run, "employee") if r["role"] == "kam"}
    assignments = _rows(story_world.run, "assignment")
    for first in QUARTERS:
        day = first.isoformat()
        reps = {
            r["employee_id"] for r in assignments if r["kind"] == "rep_area" and r["valid_from"] <= day <= r["valid_to"]
        }
        held = Counter(r["employee_id"] for r in budgets if r["quarter"] == day)
        assert set(held) == reps | kams and set(held.values()) == {1}, day
        assert sum(float(r["budget_amount"]) for r in budgets if r["quarter"] == day) > 0
    assert min(r["quarter"] for r in budgets) == "2024-01-01"


def test_grants_in_every_quarter_and_none_refused_wholesale_for_want_of_a_budget(story_world: Built) -> None:
    report = json.loads((story_world.run / BUILD_JSON).read_text(encoding="utf-8"))
    quarters = report["quarters"]
    for first in QUARTERS:
        counts = quarters[f"{first.year}-Q{(first.month - 1) // 3 + 1}"]
        refused = {k: v for k, v in counts.items() if k.startswith("refused:")}
        assert counts["granted"] > 1000 and refused, counts
        assert counts.get("refused:no_budget", 0) < counts["granted"]
    agreements = Counter(r["valid_from"][:7] for r in _rows(story_world.run, "price_agreement"))
    assert all(agreements[f"{first:%Y-%m}"] > 0 for first in QUARTERS)
    assert (report["master_seed"], report["null"]) == (WORLD.master_seed, False)
    assert report["git_sha"] and set(report["timings_s"]) == {"setup", "loop", "end", "total"}
