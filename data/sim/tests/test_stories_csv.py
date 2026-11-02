"""`stories-check --csv` (G1 S1.2 T7): the pandas twins of S1, S1-weather, S2, S2-affluence and S4 on small hand-made
CSV builds, the k-means check's parts, the runner's verdicts and report, and the warm-up switch."""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hartland_sim.checks.stories import (
    Build,
    _ntile,
    adjusted_rand,
    check,
    kmeans,
    read,
    report,
    s1,
    s1_note,
    s1_weather,
    s2_affluence,
    s4,
)
from hartland_sim.cli import main
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop.warmup import Output
from hartland_sim.setup.promotions import Templates

STORIES = StoriesConfig.load()
BIG = 9_200_000_000_000_000_001  # a 19-digit id: a float would round it


def _write(run: Path, table: str, rows: list[dict[str, Any]]) -> None:
    run.mkdir(parents=True, exist_ok=True)
    with (run / f"{table}.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _text(v) for k, v in row.items()})


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value.isoformat() if isinstance(value, date) else str(value)


LINES = {"CLASSIC": (1, False), "GOLD": (2, True), "POLEVKY": (3, False), "SMESI": (4, False)}


def _catalogue(run: Path) -> None:
    _write(run, "product_line", [{"product_line_id": i, "code": c, "is_premium": p} for c, (i, p) in LINES.items()])
    _write(run, "product", [{"product_id": 10 + i, "product_line_id": i} for i, _ in LINES.values()])
    _write(run, "sku", [{"sku_id": 100 + i, "product_id": 10 + i} for i, _ in LINES.values()])


class Orders:
    """Hand-made orders: one line each."""

    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []
        self.lines: list[dict[str, Any]] = []

    def add(self, day: date, line: str, cases: float, point: int = 1, channel: str = "small_retail") -> None:
        n = len(self.orders) + 1
        self.orders.append({"sales_order_id": n, "ordered_on": day, "channel": channel, "delivery_point_id": point})
        sku = 100 + LINES[line][0]
        self.lines.append({"sales_order_id": n, "sku_id": sku, "qty_cases": cases, "is_free_goods": False})

    def write(self, run: Path) -> None:
        _write(run, "sales_order", self.orders)
        _write(run, "sales_order_line", self.lines)


def test_read_keeps_19_digit_ids_nulls_booleans_and_dates(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "delivery_point",
        [
            {"delivery_point_id": BIG, "store_id": BIG - 1, "okres_id": 1},
            {"delivery_point_id": -BIG + 2 * 10**18, "store_id": None, "okres_id": 2},
        ],
    )
    frame = read(tmp_path / "delivery_point.csv", "delivery_point")
    assert frame["delivery_point_id"].tolist() == [BIG, -BIG + 2 * 10**18]
    assert frame["store_id"].iloc[0] == BIG - 1 and frame["store_id"].isna().iloc[1]
    _write(tmp_path, "store", [{"store_id": 1, "sunday_open": True, "opened_on": date(2024, 5, 1)}])
    store = read(tmp_path / "store.csv", "store")
    assert store["sunday_open"].tolist() == [True] and store["opened_on"].iloc[0].date() == date(2024, 5, 1)


def test_s1_is_the_smallest_peak_over_trough_of_the_lines_with_data(tmp_path: Path) -> None:
    _catalogue(tmp_path)
    orders = Orders()
    for week in range(1, 53):
        monday = date.fromisocalendar(2024, week, 1)
        classic = 13.5 if week in (45, 46, 47, 48, 49, 50, 51, 52, 1, 2, 3, 4, 5, 6, 7, 8) else 10.0
        gold = 16.0 if week in (46, 47, 48, 49, 50, 51, 52, 2, 3, 4, 5, 6, 7, 8) else 10.0
        orders.add(monday, "CLASSIC", classic)
        orders.add(monday + timedelta(days=1), "GOLD", gold)
    orders.add(date(2023, 7, 3), "CLASSIC", 1000.0)  # ISO 2023: outside the window
    orders.write(tmp_path)
    build = Build(tmp_path)
    assert s1(build, STORIES) == pytest.approx(1.35)  # no soup orders: its NULL ratio is skipped
    note = s1_note(build, STORIES)  # each line's ratio, the YAML's order: the line that sets the margin shows
    assert note.startswith("CLASSIC 1.350 · GOLD 1.600 · ESPRESSO —") and "BEZKOFEIN" not in note


def _geography(run: Path, okresy: dict[int, float]) -> None:
    """Leaf okresy with their affluence, one area and region each."""
    _write(run, "okres", [{"okres_id": o, "parent_okres_id": None, "affluence_index": a} for o, a in okresy.items()])
    _write(run, "area_okres", [{"area_id": o, "okres_id": o} for o in okresy])
    _write(run, "area", [{"area_id": o, "region_id": o} for o in okresy])


def test_s1_weather_correlates_a_region_s_soup_with_its_cold_weeks(tmp_path: Path) -> None:
    _catalogue(tmp_path)
    _geography(tmp_path, {1: 1.0, 2: 1.0})
    _write(
        tmp_path,
        "delivery_point",
        [
            {"delivery_point_id": 1, "store_id": 11, "okres_id": 1},
            {"delivery_point_id": 2, "store_id": 12, "okres_id": 2},
            {"delivery_point_id": 3, "store_id": None, "okres_id": 1},  # a depot: not where it is sold
        ],
    )
    orders, weather = Orders(), []
    for week in range(1, 40):
        monday = date.fromisocalendar(2024, week, 1)
        temp = 20 - week / 2
        for region, point in ((1, 1), (2, 2)):
            weather += [{"region_id": region, "day": monday + timedelta(days=d), "temp_mean_c": temp} for d in range(7)]
            orders.add(monday, "POLEVKY", 10 + (20 - temp) * point)
        orders.add(monday, "POLEVKY", 100 * (week % 2), point=3, channel="horeca")
    orders.write(tmp_path)
    _write(tmp_path, "weather_daily", weather)
    assert s1_weather(Build(tmp_path), STORIES) == pytest.approx(-1.0)


def _tv_promotion(run: Path, channels: list[str]) -> None:
    start, end = date(2024, 9, 2), date(2024, 9, 29)
    _write(
        run,
        "promotion",
        [{"promotion_id": 1, "campaign_id": 1, "target_product_line_id": 2, "valid_from": start, "valid_to": end}],
    )
    _write(run, "campaign", [{"campaign_id": 1, "has_tv": True}])
    _write(
        run, "promotion_eligibility", [{"promotion_id": 1, "scope_kind": "channel", "scope_value": c} for c in channels]
    )


def _affluence_build(run: Path, horeca: bool) -> None:
    """Four leaf okresy, one per quarter; a store in the poorest and the richest; Gold every day of the promotion and
    its 8 weeks before — the rich store's lift 2, the poor one's 1; with `horeca`, a café in the rich okres buying
    Gold flat, a channel the promotion does not name."""
    _catalogue(run)
    _geography(run, {1: 0.8, 2: 0.9, 3: 1.1, 4: 1.3})
    points = [
        {"delivery_point_id": 1, "store_id": 11, "okres_id": 1},
        {"delivery_point_id": 4, "store_id": 14, "okres_id": 4},
    ]
    points.append({"delivery_point_id": 5, "store_id": 15, "okres_id": 4})
    _write(run, "delivery_point", points)
    _tv_promotion(run, ["small_retail", "big_retail"])
    orders = Orders()
    day = date(2024, 9, 2) - timedelta(days=56)
    while day <= date(2024, 9, 29):
        during = day >= date(2024, 9, 2)
        orders.add(day, "GOLD", 1.0, point=1)
        orders.add(day, "GOLD", 2.0 if during else 1.0, point=4)
        if horeca:
            orders.add(day, "GOLD", 1.0, point=5, channel="horeca")
        day += timedelta(days=1)
    orders.write(run)


@pytest.mark.parametrize("horeca", [False, True])
def test_s2_affluence_counts_only_the_channels_the_promotion_names(tmp_path: Path, horeca: bool) -> None:
    """ttr-hartland#8 item 1: a café in the rich okres buying flat would pull the rich lift to 1.5."""
    _affluence_build(tmp_path, horeca)
    assert s2_affluence(Build(tmp_path), STORIES) == pytest.approx(2.0)


def test_s2_affluence_skips_a_quarter_with_no_order_during_the_promotion(tmp_path: Path) -> None:
    """The rich store stops ordering when the promotion starts: its quarter's lift is NULL in SQL, so the metric is."""
    _affluence_build(tmp_path, horeca=False)
    orders = [row for row in csv.DictReader((tmp_path / "sales_order.csv").open(encoding="utf-8"))]
    lines = [row for row in csv.DictReader((tmp_path / "sales_order_line.csv").open(encoding="utf-8"))]
    gone = {o["sales_order_id"] for o in orders if o["delivery_point_id"] == "4" and o["ordered_on"] >= "2024-09-02"}
    _write(tmp_path, "sales_order", [o for o in orders if o["sales_order_id"] not in gone])
    _write(tmp_path, "sales_order_line", [line for line in lines if line["sales_order_id"] not in gone])
    assert math.isnan(s2_affluence(Build(tmp_path), STORIES))


def test_s2_affluence_leaves_out_the_days_of_a_promotion_without_a_campaign(tmp_path: Path) -> None:
    """A promotion of the line with no campaign (`campaign_id` NULL) makes its days unclean, as any other does: the
    rich store's surge in it is not counted before the TV promotion, and the metric stays 2."""
    _affluence_build(tmp_path, horeca=False)
    rows = [row for row in csv.DictReader((tmp_path / "promotion.csv").open(encoding="utf-8"))]
    rows.append(
        {
            "promotion_id": 2,
            "campaign_id": None,
            "target_product_line_id": 2,
            "valid_from": date(2024, 7, 15),
            "valid_to": date(2024, 7, 21),
        }
    )
    _write(tmp_path, "promotion", rows)
    orders = Orders()
    for row in csv.DictReader((tmp_path / "sales_order.csv").open(encoding="utf-8")):
        orders.orders.append(row)
    orders.lines = [row for row in csv.DictReader((tmp_path / "sales_order_line.csv").open(encoding="utf-8"))]
    for day in range(15, 22):
        orders.orders.append(
            {
                "sales_order_id": 1000 + day,
                "ordered_on": date(2024, 7, day),
                "channel": "small_retail",
                "delivery_point_id": 4,
            }
        )
        orders.lines.append({"sales_order_id": 1000 + day, "sku_id": 102, "qty_cases": 50.0, "is_free_goods": False})
    orders.write(tmp_path)
    assert s2_affluence(Build(tmp_path), STORIES) == pytest.approx(2.0)


def test_s2_affluence_ends_the_history_on_the_last_order_whatever_its_place(tmp_path: Path) -> None:
    """Gold's TV promotion runs to 09-29 and Classic's to 09-08; the stores order to 09-15 and a depot on 09-29. The
    history ends on the depot's day, as the SQL's `MAX(ordered_on) FROM sales_order`: Gold has 28 days during (the
    rich store's lift 1, the poor one's 0.5), Classic 7 (1 and 1) — the metric (1 + 1) ÷ (0.5 + 1). Ended on the
    stores' last day, it would be (2 + 1) ÷ (1 + 1)."""
    _catalogue(tmp_path)
    _geography(tmp_path, {1: 0.8, 2: 0.9, 3: 1.1, 4: 1.3})
    points: list[dict[str, Any]] = [
        {"delivery_point_id": 1, "store_id": 11, "okres_id": 1},
        {"delivery_point_id": 4, "store_id": 14, "okres_id": 4},
        {"delivery_point_id": 9, "store_id": None, "okres_id": 1},  # a depot: not where it is sold
    ]
    _write(tmp_path, "delivery_point", points)
    start = date(2024, 9, 2)
    windows = [(1, 2, date(2024, 9, 29)), (2, 1, date(2024, 9, 8))]  # (promotion, line, last day)
    _write(
        tmp_path,
        "promotion",
        [
            {"promotion_id": n, "campaign_id": 1, "target_product_line_id": line, "valid_from": start, "valid_to": end}
            for n, line, end in windows
        ],
    )
    _write(tmp_path, "campaign", [{"campaign_id": 1, "has_tv": True}])
    _write(
        tmp_path,
        "promotion_eligibility",
        [{"promotion_id": n, "scope_kind": "channel", "scope_value": "small_retail"} for n, _, _ in windows],
    )
    orders = Orders()
    day = start - timedelta(days=56)
    while day <= date(2024, 9, 15):
        during = day >= start
        orders.add(day, "GOLD", 1.0, point=1)
        orders.add(day, "GOLD", 2.0 if during else 1.0, point=4)
        if day <= date(2024, 9, 8):
            orders.add(day, "CLASSIC", 1.0, point=1)
            orders.add(day, "CLASSIC", 1.0, point=4)
        day += timedelta(days=1)
    orders.add(date(2024, 9, 29), "SMESI", 1.0, point=9, channel="vending")
    orders.write(tmp_path)
    assert s2_affluence(Build(tmp_path), STORIES) == pytest.approx(2 / 1.5)


def test_every_tv_template_scopes_by_channel_only() -> None:
    """ttr-hartland#8 item 1: S2-affluence's channel rule is all of a TV promotion's eligibility, so a TV template
    names its channels and nothing else (a template without them would match no order in the twin)."""
    tv = [template for template in Templates.load().templates if template.tv]
    assert len(tv) == 8
    for template in tv:
        assert set(template.eligibility) == {"channel"}, template.key


def test_ntile_cuts_like_sql() -> None:
    import pandas as pd

    frame = pd.DataFrame({"x": [5, 1, 9, 3, 7, 2, 8, 4, 6, 10]})
    groups = _ntile(frame, "x", 4)
    assert [int(groups[frame["x"] == v].iloc[0]) for v in range(1, 11)] == [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]


def test_s4_is_the_smallest_share_of_stores_led_by_their_street_s_line(tmp_path: Path) -> None:
    _catalogue(tmp_path)
    streets = {11: "transport_hub", 12: "transport_hub", 13: "village", 14: "high_street", 15: "high_street"}
    _write(tmp_path, "store", [{"store_id": s, "street_type": t} for s, t in streets.items()])
    _write(tmp_path, "delivery_point", [{"delivery_point_id": s, "store_id": s, "okres_id": 1} for s in streets])
    orders = Orders()
    leaders = {11: "SMESI", 12: "SMESI", 13: "POLEVKY", 14: "GOLD", 15: "CLASSIC"}
    for store, line in leaders.items():
        orders.add(date(2025, 3, 3), line, 5.0, point=store)
        orders.add(date(2025, 3, 3), "CLASSIC" if line != "CLASSIC" else "GOLD", 1.0, point=store)
        orders.add(date(2024, 3, 4), "CLASSIC", 100.0, point=store)  # 2024: outside S4's window
    orders.write(tmp_path)
    assert s4(Build(tmp_path), STORIES) == pytest.approx(0.5)


def test_s4_is_nan_when_a_street_type_has_no_ordering_store(tmp_path: Path) -> None:
    """s4.sql's NULL: no high-street store orders in 2025, so the detector cannot say the high street leads."""
    _catalogue(tmp_path)
    streets = {11: "transport_hub", 13: "village", 14: "high_street"}
    _write(tmp_path, "store", [{"store_id": s, "street_type": t} for s, t in streets.items()])
    _write(tmp_path, "delivery_point", [{"delivery_point_id": s, "store_id": s, "okres_id": 1} for s in streets])
    orders = Orders()
    orders.add(date(2025, 3, 3), "SMESI", 5.0, point=11)
    orders.add(date(2025, 3, 3), "POLEVKY", 5.0, point=13)
    orders.add(date(2024, 3, 4), "GOLD", 5.0, point=14)  # 2024: outside S4's window
    orders.write(tmp_path)
    assert math.isnan(s4(Build(tmp_path), STORIES))


def test_kmeans_finds_separate_groups_and_the_adjusted_rand_index() -> None:
    blobs = np.concatenate([np.full((20, 3), k, dtype=float) + np.linspace(0, 0.1, 20)[:, None] for k in range(3)])
    truth = np.repeat(["a", "b", "c"], 20)
    assert adjusted_rand(kmeans(blobs, 3), truth) == pytest.approx(1.0)
    assert adjusted_rand(np.array([0, 0, 1, 1]), np.array(["x", "y", "x", "y"])) == pytest.approx(-0.5)
    assert adjusted_rand(np.array([0, 0, 1, 1]), np.array(["x", "x", "y", "y"])) == pytest.approx(1.0)


def test_the_runner_judges_what_it_can_and_names_what_it_cannot(tmp_path: Path) -> None:
    _affluence_build(tmp_path, horeca=False)
    verdicts = {v.check: v for v in check(tmp_path, STORIES)}
    assert verdicts["detectors/s2_affluence.sql"].passed is True
    assert verdicts["detectors/s2.sql"].passed is None and "promotion_result" in verdicts["detectors/s2.sql"].note
    assert verdicts["detectors/s3.sql"].passed is None and verdicts["detectors/s3.sql"].note == "no pandas twin yet"
    s1_found = verdicts["detectors/s1.sql"]
    assert s1_found.passed is False and math.isnan(s1_found.value or math.nan)  # Gold only: no curve's weeks both
    assert verdicts["kmeans"].passed is None
    null = {v.check: v for v in check(tmp_path, STORIES, null=True)}
    assert null["detectors/s2_affluence.sql"].passed is False  # 2.0 is no null world
    text = report(list(verdicts.values()), tmp_path, null=False)
    assert "| S2 | `detectors/s2_affluence.sql` | tv_rich_over_poor_lift | >= 1.35 | 2.000 | pass |" in text
    assert "not run (no promotion_result.csv (G4 S4.3 writes it))" in text
    assert "| S4 | `kmeans` | adjusted_rand_index | >= 0.6 |" in text


def _disabled(*names: str) -> StoriesConfig:
    return replace(
        STORIES, stories=tuple(replace(s, enabled=False) if s.story in names else s for s in STORIES.stories)
    )


def test_a_disabled_story_is_not_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Contracts §3: every *enabled* detector. With S1 and S4 off, the build that fails S1 passes."""
    _affluence_build(tmp_path, horeca=False)
    found = [v for v in check(tmp_path, _disabled("S1", "S4")) if v.story in ("S1", "S4")]
    assert [v.check for v in found if v.story == "S4"] == ["detectors/s4.sql", "kmeans"]
    assert all(v.passed is None and v.value is None and v.note == "disabled" for v in found)
    monkeypatch.setattr(StoriesConfig, "load", classmethod(lambda cls: _disabled("S1", "S4")))
    assert main(["stories-check", "--csv", str(tmp_path)]) == 0
    assert "| S4 | `kmeans` | adjusted_rand_index | >= 0.6 | — | not run (disabled) |" in capsys.readouterr().out


def test_stories_check_csv_writes_the_report_and_fails_on_a_failed_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _affluence_build(tmp_path, horeca=False)
    assert main(["stories-check", "--csv", str(tmp_path)]) == 1  # S1 cannot pass on Gold alone
    assert (tmp_path / "stories.md").read_text(encoding="utf-8") == capsys.readouterr().out


def test_the_warm_up_writes_nothing_before_the_first_day(tmp_path: Path) -> None:
    world = WorldConfig.load()
    with CsvWriter(tmp_path) as writer:
        output = Output.of(writer, world)
        first = world.history.first_day
        rows = [
            {"region_id": 1, "day": d, "temp_mean_c": 1.0, "precip_mm": 0.0, "snow": False, "severe": False}
            for d in (first - timedelta(days=1), first)
        ]
        assert output.append(first - timedelta(days=1), "weather_daily", rows[:1]) == 0
        assert output.append(first, "weather_daily", rows[1:]) == 1
        assert not output.writes(first - timedelta(days=1)) and output.writes(first)
    written = (tmp_path / "weather_daily.csv").read_text(encoding="utf-8").splitlines()
    assert len(written) == 2 and written[1].split(",")[1] == first.isoformat()
