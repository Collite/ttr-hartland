"""The month-end's G1 half (G1 S1.4 T3, `05` §3): M+1's conditions from what month M booked (S1.3's tiers), the
phases from the last six months (R-4), the rolling plan of M+1…M+3 (the demand's expectation × a planning error,
stream `plan`), M+1's channel lists; each row written under the day it becomes valid."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import add_months
from hartland_sim.ids import row_id
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import acquisition
from hartland_sim.loop.demand import CHANNELS
from hartland_sim.loop.pricing import Rules, standard_list
from hartland_sim.loop.warmup import Output
from hartland_sim.period import Closing, month_end
from hartland_sim.rng import uniforms
from hartland_sim.setup.run import Setup
from hartland_sim.world import Bought, Checkpoint, Customer, Episode, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
LIFE = WORLD.policies.lifecycle
DAY = date(2023, 1, 31)
FEB = date(2023, 2, 1)
NEVER = date(2100, 1, 1)  # an `Output` that writes nothing


@pytest.fixture
def world(opening: Setup) -> World:
    return Checkpoint.of(opening.world).restore()


@pytest.fixture(scope="module")
def closing(opening: Setup) -> Closing:
    rules = Rules.of(opening.world, WORLD, STORIES)
    return Closing.of(opening.world, WORLD, STORIES, opening.portfolio.skus, rules)


def _retail(world: World, n: int = 1) -> list[Customer]:
    return [c for c in world.customers.values() if c.channel == "small_retail" and c.active_on(FEB)][:n]


def _bits(closing: Closing, *lines: str) -> int:
    return sum(1 << closing.rules.lines[line] for line in lines)


def _rows(run: Path, table: str) -> list[dict[str, str]]:
    path = run / f"{table}.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def test_m_plus_1_s_conditions_come_from_what_month_m_booked(world: World, closing: Closing, tmp_path: Path) -> None:
    rich, poor = _retail(world, 2)
    lines = _bits(closing, "CLASSIC", "GOLD", "SMESI", "POLEVKY", "NUDLE")  # 5 lines, coffee and culinary
    world.bought[DAY.replace(day=1)] = {rich.customer_id: Bought(51_000.0, 3, lines)}
    with CsvWriter(tmp_path) as writer:
        rows = month_end.conditions(world, closing, DAY, Output(writer, FEB))
    assert rich.condition is not None and rich.condition.month == FEB
    assert (rich.condition.volume_tier, rich.condition.volume_pct) == ("VOL-T2", 4)
    assert (rich.condition.cross_sell_tier, rich.condition.cross_sell_pct) == ("XS-T2", 3)  # 2 % + 1 % both brands
    assert poor.condition is not None and (poor.condition.volume_pct, poor.condition.cross_sell_pct) == (0, 0)
    tiered = [c for c in world.customers.values() if c.channel == "small_retail" and c.active_on(FEB)]
    assert [r["customer_id"] for r in rows] == [c.customer_id for c in tiered]
    written = {int(r["customer_id"]): r for r in _rows(tmp_path, "customer_condition")}
    assert written[rich.customer_id]["month"] == "2023-02-01"
    assert written[rich.customer_id]["prev_month_net_revenue"] == "51000.00"
    assert (written[rich.customer_id]["prev_month_lines_bought"], written[rich.customer_id]["both_brands"]) == (
        "5",
        "true",
    )


def test_a_row_valid_before_the_first_written_day_is_not_written(
    world: World, closing: Closing, tmp_path: Path
) -> None:
    with CsvWriter(tmp_path) as writer:
        month_end.conditions(world, closing, DAY, Output(writer, date(2023, 3, 1)))
    assert not _rows(tmp_path, "customer_condition")


def test_the_phases_follow_the_last_six_months(world: World, closing: Closing) -> None:
    vip, regular, falls, stays = _retail(world, 4)
    months = [add_months(FEB, -k) for k in range(1, LIFE.lookback_months + 1)]  # Jan 2023 back to Aug 2022
    for month in months:
        world.bought[month] = {vip.customer_id: Bought(LIFE.vip_min_net_revenue / 6, 2, 1)}
    for month in months[: LIFE.regular_min_order_months]:
        world.bought[month][regular.customer_id] = Bought(1_000.0, 1, 1)
    for month in months[: LIFE.regular_min_order_months - 1]:
        world.bought[month][falls.customer_id] = Bought(1_000.0, 1, 1)
    falls.episodes.append(Episode("regular", date(2022, 12, 1), "test"))
    stays.episodes.append(Episode("occasional", date(2022, 12, 1), "test"))
    opened = dict(month_end.phases(world, closing, DAY))
    assert (opened[vip.customer_id].phase, opened[vip.customer_id].valid_from) == ("vip", FEB)
    assert opened[regular.customer_id].phase == "regular"
    assert opened[falls.customer_id].phase == "occasional"
    assert stays.customer_id not in opened
    assert vip.phase_on(DAY) != "vip" and vip.phase_on(FEB) == "vip"
    assert opened[regular.customer_id].reason == "Objednávky alespoň v 4 z posledních 6 měsíců"
    assert all(world.customers[c].phase_on(DAY) not in ("potential", "lost") for c in opened)


def test_a_loss_on_the_first_replaces_the_month_end_s_phase_of_that_day(world: World) -> None:
    customer = next(c for c in _retail(world, 50) if not c.traits)
    customer.episodes.append(Episode("regular", FEB, "test"))
    reps = acquisition.Reps.of(world, STORIES)
    reps.rates = acquisition.Rates(0.0, 1e6, 0.0)  # every unpinned customer is lost, whatever its rep
    acquisition.step(world, reps, FEB)
    assert [e.valid_from for e in customer.episodes].count(FEB) == 1
    assert (customer.episodes[-1].phase, customer.phase_on(FEB)) == ("lost", "lost")


def test_the_plan_rolls_m_plus_1_to_m_plus_3_by_key_account_area_and_channel(
    world: World, closing: Closing, tmp_path: Path
) -> None:
    with CsvWriter(tmp_path) as writer:
        rows = month_end.plan(world, closing, DAY, Output(writer, FEB))
    months = {r["month"]: r["version"] for r in rows}
    assert months == {date(2023, 2, 1): 3, date(2023, 3, 1): 2, date(2023, 4, 1): 1}
    by_kind = {(r["customer_id"] is not None, r["area_id"] is not None, r["channel"]) for r in rows}
    assert by_kind == {
        (True, False, "big_retail"),
        (True, False, "gas_ka"),
        (False, True, "small_retail"),
        *((False, False, channel) for channel in ("own_shops", "web", "vending", "horeca")),
    }
    errors = WORLD.policies.sales_plan.error_pct
    lines = {row_id("product_line", code=line): line for line in closing.rules.lines}
    customers = {row_id("customer", code=t[1]): t[1] for t in closing.targets if t[1]}
    areas = {row_id("area", code=t[2]): t[2] for t in closing.targets if t[2]}
    for month, ahead in ((FEB, 1), (date(2023, 4, 1), 3)):
        cases, gross = month_end.expectation(world, closing, DAY, month)
        planned = [r for r in rows if r["month"] == month]
        for r in planned:
            assert r["sales_plan_id"] == row_id(
                "sales_plan",
                month=month,
                channel=r["channel"],
                product_line=lines[r["product_line_id"]],
                customer=customers.get(r["customer_id"], ""),
                area=areas.get(r["area_id"], ""),
                version=len(errors) - ahead + 1,
            )
        total = sum(r["planned_cases"] for r in planned)
        assert total == pytest.approx(cases.sum(), rel=errors[ahead - 1] / 100)
        assert sum(r["planned_net_revenue"] for r in planned) == pytest.approx(gross.sum(), rel=errors[ahead - 1] / 100)
    written = _rows(tmp_path, "sales_plan")
    assert len(written) == len(rows) and {r["month"] for r in written} == {"2023-02-01", "2023-03-01", "2023-04-01"}


def test_the_plan_s_error_is_drawn_per_row_within_its_months_ahead(
    world: World, closing: Closing, tmp_path: Path
) -> None:
    with CsvWriter(tmp_path) as writer:
        rows = month_end.plan(world, closing, DAY, Output(writer, NEVER))
    cases, _ = month_end.expectation(world, closing, DAY, FEB)
    n = len(closing.targets)
    sums = np.stack([np.bincount(closing.point_target, cases[:, i], n) for i in range(cases.shape[1])], axis=1)
    lines = list(closing.rules.lines)
    for r in rows:
        if r["month"] != FEB:
            continue
        t = next(
            k
            for k, (channel, customer, area) in enumerate(closing.targets)
            if channel == r["channel"]
            and (row_id("customer", code=customer) if customer else None) == r["customer_id"]
            and (row_id("area", code=area) if area else None) == r["area_id"]
        )
        i = next(k for k, line in enumerate(lines) if row_id("product_line", code=line) == r["product_line_id"])
        error = 1 + (2 * float(uniforms("plan", ids=np.array([r["sales_plan_id"]]))[0]) - 1) * 4 / 100
        assert r["planned_cases"] == pytest.approx(sums[t, i] * error, abs=1e-3)


def test_a_quarter_s_last_month_end_keeps_q_plus_1_s_planned_gross_by_budget_holder(
    world: World, closing: Closing, opening: Setup, tmp_path: Path
) -> None:
    with CsvWriter(tmp_path) as writer:
        month_end.plan(world, closing, DAY, Output(writer, NEVER))
        assert world.planned_gross == {}  # only a quarter's last month-end plans the budgets
        day = date(2023, 3, 31)
        world.standard_list = standard_list(opening.portfolio.skus, day + timedelta(days=1))
        rows = month_end.plan(world, closing, day, Output(writer, NEVER))
    planned = world.planned_gross
    assert set(planned) == set(closing.holders)
    for code in ("KA-TESCO", "KA-LIDL"):
        customer = row_id("customer", code=code)
        assert planned[code] == pytest.approx(
            sum(r["planned_net_revenue"] for r in rows if r["customer_id"] == customer)
        )
    area = next(c.area for c in _retail(world) if c.area)
    assert area is not None
    area_id = row_id("area", code=area)
    assert planned[area] == pytest.approx(sum(r["planned_net_revenue"] for r in rows if r["area_id"] == area_id))
    horeca = [c.code for c in world.customers.values() if c.channel == "horeca" and c.code in planned]
    channel_rows = sum(r["planned_net_revenue"] for r in rows if r["channel"] == "horeca")
    assert sum(planned[code] for code in horeca) == pytest.approx(channel_rows)
    assert {r["month"] for r in rows} == {date(2023, 4, 1), date(2023, 5, 1), date(2023, 6, 1)}


def test_m_plus_1_s_channel_lists_need_the_standard_of_its_quarter(
    world: World, closing: Closing, tmp_path: Path
) -> None:
    with CsvWriter(tmp_path) as writer:
        month_end.lists(world, closing, FEB, Output(writer, FEB))
    assert {(c, listed.valid_from) for c, listed in world.channel_lists.items()} == {(c, FEB) for c in CHANNELS}
    heads = _rows(tmp_path, "channel_price_list")
    assert [h["channel"] for h in heads] == list(CHANNELS) and {h["valid_from"] for h in heads} == {"2023-02-01"}
    with pytest.raises(ValueError, match="roll the standard list first"):
        month_end.lists(world, closing, date(2023, 4, 1), Output(writer, FEB))


def test_the_warm_up_s_last_month_end_writes_january_2024_and_nothing_before(
    world: World, closing: Closing, opening: Setup, tmp_path: Path
) -> None:
    first = WORLD.history.first_day
    skus = opening.portfolio.skus
    world.standard_list = standard_list(skus, date(2023, 12, 1))
    with CsvWriter(tmp_path) as writer:
        month_end.close(world, closing, date(2023, 11, 30), Output(writer, first))
    assert not _rows(tmp_path, "customer_condition") and not _rows(tmp_path, "channel_price_list")
    plans = _rows(tmp_path, "sales_plan")
    assert {(r["month"], r["version"]) for r in plans} == {("2024-01-01", "2"), ("2024-02-01", "1")}
    world.standard_list = standard_list(skus, first)  # the quarter-end's roll
    with CsvWriter(tmp_path) as writer:
        month_end.close(world, closing, date(2023, 12, 31), Output(writer, first))
    assert {r["month"] for r in _rows(tmp_path, "customer_condition")} == {"2024-01-01"}
    assert {r["valid_from"] for r in _rows(tmp_path, "channel_price_list")} == {"2024-01-01"}
    assert {(r["month"], r["version"]) for r in _rows(tmp_path, "sales_plan")} >= {("2024-01-01", "3")}


def test_close_keeps_the_months_the_next_month_end_reads(world: World, closing: Closing, tmp_path: Path) -> None:
    for k in range(10):
        world.bought[add_months(FEB, -k - 1)] = {}
    with CsvWriter(tmp_path) as writer:
        month_end.close(world, closing, DAY, Output(writer, NEVER))
        assert sorted(world.bought) == [add_months(FEB, -k) for k in range(LIFE.lookback_months - 1, 0, -1)]
        with pytest.raises(ValueError, match="last day"):
            month_end.close(world, closing, date(2023, 1, 30), Output(writer, NEVER))
