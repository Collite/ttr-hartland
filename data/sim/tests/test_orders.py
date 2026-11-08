"""The orders (G1 S1.4 T1, `03` §6, `05` §3 step 4): a basket becomes one `sales_order` — its number and ids from the
business keys, its source by the channel's mix (stream `order`), the next working day requested, the credit check's
status — with its lines priced by S1.3 and each discount a row; and the customers' asks, once a quarter each, before
the day's orders."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import next_working_day, quarter_end
from hartland_sim.ids import row_id
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import orders
from hartland_sim.loop.demand import BasketReady
from hartland_sim.loop.orders import Desk, order, order_no, source
from hartland_sim.loop.pricing import Refusal, Rules, condition, granter, price_order
from hartland_sim.loop.warmup import Output
from hartland_sim.rng import uniforms
from hartland_sim.setup.run import Setup
from hartland_sim.world import Agreement, Budget, Checkpoint, Customer, OpenInvoice, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
DAY = date(2023, 1, 13)  # a Friday of the opening month, whose lists the World holds
QUARTER = date(2023, 1, 1)
DCS = {region: dc for dc, regions in WORLD.sites.dcs.items() for region in regions}


@pytest.fixture
def world(opening: Setup) -> World:
    return Checkpoint.of(opening.world).restore()


@pytest.fixture
def desk(world: World, opening: Setup) -> Desk:
    return Desk.of(world, Rules.of(opening.world, WORLD, STORIES), WORLD)


def _customers(world: World, channel: str, sub_channel: str | None = None) -> list[Customer]:
    return [
        c
        for c in world.customers.values()
        if c.channel == channel and c.sub_channel == sub_channel and c.active_on(DAY)
    ]


def _basket(world: World, customer: Customer, point: int = 0, day: date = DAY, skus: int = 3) -> BasketReady:
    """`skus` of the channel list's SKUs, 2 cases each, at the customer's `point`-th delivery point."""
    demand = world.demand
    assert demand is not None
    c = int(np.flatnonzero(demand.customer_ids == customer.customer_id)[0])
    point_id = int(demand.point_ids[int(demand.customer_start[c]) + point])
    prices = world.channel_lists[customer.channel].prices
    units = dict(zip(demand.catalogue.sku_ids.tolist(), demand.catalogue.units.tolist(), strict=True))
    lines = tuple((sku, 2.0, 2.0 * units[sku]) for sku in list(prices)[:skus])
    return BasketReady(day, customer.customer_id, point_id, customer.channel, lines)


def _head(world: World, desk: Desk, basket: BasketReady, draw: float) -> dict[str, Any]:
    """The `sales_order` row of `basket` as its customer's first order of the day."""
    head = order(world, desk, basket, 1, draw).head
    assert head is not None
    return head


def test_a_basket_becomes_one_order_with_its_lines_priced_by_s1_3(world: World, desk: Desk) -> None:
    customer = _customers(world, "small_retail")[0]
    customer.condition = condition(WORLD.pricing.tiers, QUARTER, 60_000, 5, True)  # VOL-T2 4 % + XS-T2 3 %
    basket = _basket(world, customer)
    made = order(world, desk, basket, 1, 0.99)
    number = f"20230113-{customer.code}-1"
    assert made.head is not None
    assert made.head == {
        "sales_order_id": row_id("sales_order", order_no=number),
        "order_no": number,
        "customer_id": customer.customer_id,
        "delivery_point_id": basket.delivery_point_id,
        "channel": "small_retail",
        "ordered_on": DAY,
        "requested_on": date(2023, 1, 16),  # Friday → Monday
        "source": "portal",
        "taken_by_employee_id": None,
        "visit_id": None,
        "status": "released",
        "hold_reason": None,
        "dc_site_id": made.head["dc_site_id"],
    }
    demand = world.demand
    assert demand is not None
    spot = demand.spots[desk.rules.points[basket.delivery_point_id]]
    assert made.head["dc_site_id"] == row_id("site", code=DCS[spot.region])
    priced = price_order(world, desk.rules, basket)
    assert made.priced == priced
    standard = world.standard_list
    assert standard is not None
    for line_no, (line, written) in enumerate(zip(priced, made.lines, strict=True), start=1):
        assert written["sales_order_line_id"] == row_id("sales_order_line", order_no=number, line_no=line_no)
        assert (written["sales_order_id"], written["line_no"], written["sku_id"]) == (
            made.head["sales_order_id"],
            line_no,
            line.sku_id,
        )
        assert (written["qty_cases"], written["list_price_case"], written["invoice_price_case"]) == (
            line.cases,
            line.list_price_case,
            line.invoice_price_case,
        )
        assert (written["discount_total_case"], written["discount_total"]) == (
            line.discount_total_case,
            line.discount_total,
        )
        assert written["purchase_price_case"] == standard.prices[line.sku_id].purchase_case
        assert written["vat_rate"] == desk.rules.vat[line.sku_id]
        assert written["inbound_cost_case"] == 0.0 and written["warehouse_cost_case"] is None
    kinds = Counter((d["sales_order_line_id"], d["kind"]) for d in made.discounts)
    assert len(made.discounts) == 2 * len(made.lines)  # the volume and the cross-sell tier on every line
    assert {kind for _, kind in kinds} == {"volume", "cross_sell"} and set(kinds.values()) == {1}
    assert {d["sales_order_line_id"] for d in made.discounts} == {line["sales_order_line_id"] for line in made.lines}


def test_the_number_of_an_order_is_its_customer_s_count_of_the_day(world: World, desk: Desk, tmp_path: Path) -> None:
    (chain,) = [c for c in _customers(world, "big_retail") if c.code == "KA-TESCO"]
    baskets = [_basket(world, chain, point) for point in range(3)]
    with CsvWriter(tmp_path) as writer:
        done = orders.step(world, desk, DAY, baskets, Output(writer, DAY))
    assert done.orders == 3
    with (tmp_path / "sales_order.csv").open(encoding="utf-8") as source_file:
        written = list(csv.DictReader(source_file))
    assert [r["order_no"] for r in written] == [order_no("KA-TESCO", DAY, n) for n in (1, 2, 3)]
    assert {r["source"] for r in written} == {"edi"} and {r["taken_by_employee_id"] for r in written} == {""}
    assert len({r["delivery_point_id"] for r in written}) == 3


def test_the_next_working_day_skips_weekends_and_the_czech_days_off() -> None:
    assert next_working_day(date(2023, 1, 13)) == date(2023, 1, 16)  # Friday → Monday
    assert next_working_day(date(2023, 1, 16)) == date(2023, 1, 17)
    assert next_working_day(date(2023, 4, 6)) == date(2023, 4, 11)  # Good Friday, the weekend, Easter Monday
    assert next_working_day(date(2023, 12, 22)) == date(2023, 12, 27)  # the weekend, 24–26 December
    assert next_working_day(date(2024, 12, 31)) == date(2025, 1, 2)  # New Year's Day


def test_the_source_follows_the_channel_s_mix_and_a_station_shop_takes_special_retail_s(
    world: World, desk: Desk
) -> None:
    # small retail's mix holds no visit: a visit's order is the field's (G2 S2.1)
    assert [source(desk, "small_retail", u) for u in (0.0, 0.639, 0.64, 0.999)] == [
        "phone",
        "phone",
        "portal",
        "portal",
    ]
    # nor HoReCa's: its KAMs' visits are not modelled (review-231 R5)
    assert [source(desk, "horeca", u) for u in (0.0, 0.62, 0.63, 0.999)] == ["phone", "phone", "portal", "portal"]
    assert {source(desk, channel, u) for channel in ("big_retail", "gas_ka") for u in (0.0, 0.5, 0.999)} == {"edi"}
    assert (source(desk, "web", 0.3), source(desk, "own_shops", 0.3)) == ("web", "replenishment")
    shop = _customers(world, "small_retail", "special_retail")[0]
    store = _customers(world, "small_retail")[0]
    assert _head(world, desk, _basket(world, shop), 0.6)["source"] == "portal"  # special retail: phone 0.57
    assert _head(world, desk, _basket(world, store), 0.6)["source"] == "phone"  # small retail: phone 0.64


def test_the_sources_of_a_day_s_orders_are_drawn_per_order(world: World, desk: Desk) -> None:
    stores = _customers(world, "small_retail")[:600]
    ids = np.array([row_id("sales_order", order_no=order_no(c.code, DAY, 1)) for c in stores])
    draws = uniforms("order", "source", ids=ids)
    assert uniforms("order", "source", ids=ids[::-1])[::-1].tolist() == draws.tolist()  # an id's own draw
    counts = Counter(source(desk, "small_retail", float(u)) for u in draws)
    for name, share in WORLD.channels.small_retail.order_sources.items():
        assert counts[name] / len(stores) == pytest.approx(share, abs=0.06)


def test_a_visit_or_a_phone_order_is_taken_by_the_customer_s_rep_or_kam(world: World, desk: Desk) -> None:
    store = _customers(world, "small_retail")[0]
    venue = _customers(world, "horeca")[0]
    rep, kam = granter(world, store, DAY), granter(world, venue, DAY)
    assert rep is not None and kam is not None
    assert _head(world, desk, _basket(world, store), 0.0)["taken_by_employee_id"] == rep  # visit
    assert _head(world, desk, _basket(world, store), 0.5)["taken_by_employee_id"] == rep  # phone
    assert _head(world, desk, _basket(world, store), 0.9)["taken_by_employee_id"] is None  # portal
    assert _head(world, desk, _basket(world, venue), 0.1)["taken_by_employee_id"] == kam  # phone


def test_a_visit_or_a_phone_order_no_one_can_take_draws_again_among_the_others(
    world: World, desk: Desk, opening: Setup
) -> None:
    """review-221 N3: the same draw, the mix's other sources renormalised (small retail's: the portal alone)."""
    store = _customers(world, "small_retail")[0]
    held = _head(world, desk, _basket(world, store), 0.0)
    assert (held["source"], held["taken_by_employee_id"]) == ("phone", granter(world, store, DAY))
    store.area = None  # its area unheld
    for draw in (0.0, 0.5, 0.9):
        head = _head(world, desk, _basket(world, store), draw)
        assert (head["source"], head["taken_by_employee_id"]) == ("portal", None)
    channels = WORLD.channels
    taken = replace(channels, horeca=replace(channels.horeca, order_sources={"visit": 0.5, "phone": 0.5}))
    with pytest.raises(ValueError, match="horeca.order_sources: an order no one takes"):
        Desk.of(world, Rules.of(opening.world, WORLD, STORIES), replace(WORLD, channels=taken))


def test_the_re_pick_follows_the_untaken_shares_when_there_are_several(world: World, opening: Setup) -> None:
    """RR4: the draw is rescaled within the visit's interval, so {visit .5, portal .25, web .25} re-picks half each."""
    channels = WORLD.channels
    mix = {"visit": 0.5, "portal": 0.25, "web": 0.25}
    config = replace(WORLD, channels=replace(channels, small_retail=replace(channels.small_retail, order_sources=mix)))
    desk = Desk.of(world, Rules.of(opening.world, WORLD, STORIES), config)
    store = _customers(world, "small_retail")[0]
    store.area = None
    draws = [(k + 0.5) / 200 for k in range(100)]  # evenly over the visit's [0, 0.5)
    counts = Counter(_head(world, desk, _basket(world, store), draw)["source"] for draw in draws)
    assert set(counts) == {"portal", "web"} and abs(counts["portal"] - counts["web"]) <= 2


def test_an_order_of_a_customer_with_an_invoice_overdue_is_held(world: World, desk: Desk) -> None:
    store = _customers(world, "small_retail")[0]
    due = DAY - timedelta(days=WORLD.finance.hold.overdue_days_over + 1)
    world.open_invoices = {1: OpenInvoice(1, store.customer_id, due - timedelta(days=14), due, 500.0)}
    head = _head(world, desk, _basket(world, store), 0.5)
    assert (head["status"], head["hold_reason"]) == ("held", "overdue")


def test_an_order_needs_the_standard_list_of_its_day(world: World, desk: Desk) -> None:
    store = _customers(world, "small_retail")[0]
    basket = _basket(world, store)
    later = BasketReady(date(2023, 4, 3), basket.customer_id, basket.delivery_point_id, basket.channel, basket.lines)
    with pytest.raises(ValueError, match="standard list"):
        order(world, desk, later, 1, 0.5)


# ── the asks ───────────────────────────────────────────────────────────────────


def test_each_asking_customer_asks_once_a_quarter_on_a_day_of_it(world: World, desk: Desk) -> None:
    by_day, lines = desk.plan(world, DAY)
    asked = np.concatenate(list(by_day.values()))
    assert sorted(asked.tolist()) == list(range(len(desk.asking)))
    assert min(by_day) == QUARTER.toordinal() and max(by_day) == quarter_end(DAY).toordinal()
    assert desk.plan(world, date(2023, 3, 31)) is desk.plan(world, DAY)
    again = Desk.of(world, desk.rules, WORLD).plan(world, DAY)
    assert {d: k.tolist() for d, k in again[0].items()} == {d: k.tolist() for d, k in by_day.items()}
    assert again[1].tolist() == lines.tolist()
    customers = list(world.customers.values())
    channels = Counter(customers[int(c)].channel for c in desk.asking)
    assert set(channels) == {"small_retail", "big_retail", "gas_ka", "vending", "horeca"}
    assert channels["big_retail"] == 9 and "web" not in channels


def test_a_day_s_asks_are_the_active_customers_lines_they_buy(world: World, desk: Desk) -> None:
    demand = world.demand
    assert demand is not None
    index = {int(customer): c for c, customer in enumerate(demand.customer_ids)}
    lines = {line.code: i for i, line in enumerate(demand.lines)}
    answers: list[Agreement | Refusal] = []
    for k in range(31):
        day = QUARTER + timedelta(days=k)
        for answer in orders.asks(world, desk, day):
            customer = world.customers[answer.customer_id]
            assert customer.active_on(day)
            assert demand.carried[index[customer.customer_id], lines[answer.line]]
            answers.append(answer)
    assert len({a.customer_id for a in answers}) == len(answers)  # once a quarter
    granted = [a for a in answers if isinstance(a, Agreement)]
    assert 500 < len(answers) < 1500 and len(granted) > len(answers) / 3
    assert all(a.valid_to == quarter_end(DAY) for a in granted)
    assert {a.code for a in world.agreements.values()} == {a.code for a in granted}


def test_the_asks_of_a_day_come_before_its_orders_and_are_written(world: World, desk: Desk, tmp_path: Path) -> None:
    by_day, _ = desk.plan(world, DAY)
    customers = list(world.customers.values())
    day = min(
        d
        for d, asking in by_day.items()
        if QUARTER.toordinal() + 2 <= d <= DAY.toordinal()
        and any(customers[int(desk.asking[k])].channel == "small_retail" for k in asking)
    )
    for actor in world.actors.values():  # every ask that reaches a rep is granted, within any budget
        actor.skills.discount_discipline = 0.5
    for key in world.budgets:
        world.budgets[key] = Budget(1e12)
    probe = Checkpoint.of(world).restore()
    answered = orders.asks(probe, Desk.of(probe, desk.rules, WORLD), date.fromordinal(day))
    retail = [a for a in answered if isinstance(a, Agreement) and a.kind == "agreed_discount"]
    assert retail
    asked = world.customers[retail[0].customer_id]
    demand = world.demand
    assert demand is not None
    goods = demand.catalogue
    skus = [
        int(s) for s, line in zip(goods.sku_ids, goods.line, strict=True) if demand.lines[line].code == retail[0].line
    ]
    on_list = [s for s in skus if s in world.channel_lists["small_retail"].prices]
    point = int(
        demand.point_ids[int(demand.customer_start[int(np.flatnonzero(demand.customer_ids == asked.customer_id)[0])])]
    )
    basket = BasketReady(date.fromordinal(day), asked.customer_id, point, "small_retail", ((on_list[0], 2.0, 12.0),))
    with CsvWriter(tmp_path) as writer:
        done = orders.step(world, desk, date.fromordinal(day), [basket], Output(writer, QUARTER))
    assert done.granted == len([a for a in answered if isinstance(a, Agreement)])
    with (tmp_path / "sales_order_line_discount.csv").open(encoding="utf-8") as source_file:
        rows = list(csv.DictReader(source_file))
    assert [(r["kind"], r["code"]) for r in rows] == [("agreed", retail[0].code)]
    with (tmp_path / "price_agreement.csv").open(encoding="utf-8") as source_file:
        written = list(csv.DictReader(source_file))
    assert {r["code"] for r in written} == {a.code for a in world.agreements.values()}
    assert rows[0]["price_agreement_id"] == str(row_id("price_agreement", code=retail[0].code))


def test_the_warm_up_s_orders_and_asks_move_the_world_and_write_nothing(
    world: World, desk: Desk, tmp_path: Path
) -> None:
    by_day, _ = desk.plan(world, DAY)
    day = date.fromordinal(min(by_day))
    store = _customers(world, "small_retail")[0]
    with CsvWriter(tmp_path) as writer:
        done = orders.step(world, desk, day, [_basket(world, store, day=day)], Output(writer, WORLD.history.first_day))
    assert done.orders == 1 and done.granted + sum(done.refused.values()) > 0
    assert len(world.agreements) == done.granted
    assert not list(tmp_path.glob("*.csv"))
