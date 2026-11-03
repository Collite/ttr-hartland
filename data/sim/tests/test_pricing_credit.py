"""The credit check (G1 S1.3 T7, A-14): an order is held when an open invoice is more than `finance.hold.
overdue_days_over` days past due, or when the open invoices and the order would pass the credit limit of the
customer's terms; else released. The `World`'s ledger is empty until G4, so a fixture ledger plays A-14: its customer
crosses its limit in the month `anomalies.yaml` names, and is released once it pays."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.loop.pricing import Priced, credit_check, order_value
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, Customer, OpenInvoice, World

WORLD = WorldConfig.load()
LIMITS = {str(channel): limit for channel, limit in WORLD.policies.credit_limit.items()}
OVERDUE = WORLD.finance.hold.overdue_days_over
MONTH = StoriesConfig.load().anomaly("A-14").parameters["month"]


@pytest.fixture
def world(opening: Setup) -> World:
    return Checkpoint.of(opening.world).restore()


def _a14(world: World) -> Customer:
    (found,) = [c for c in world.customers.values() if "crosses_credit_limit" in c.traits]
    return found


def _owes(world: World, customer: Customer, *invoices: tuple[date, int, float]) -> None:
    """`customer`'s open invoices: (issued on, days to pay, open amount)."""
    world.open_invoices = {
        n: OpenInvoice(n, customer.customer_id, issued, issued + timedelta(days=days), amount)
        for n, (issued, days, amount) in enumerate(invoices, start=1)
    }


def test_an_order_with_nothing_owed_is_released(world: World) -> None:
    customer = _a14(world)
    assert world.open_invoices == {}
    assert credit_check(world, WORLD, customer, MONTH, 10_000) == ("released", None)


def test_the_a14_customer_crosses_its_limit_in_the_planted_month_and_is_released_once_it_pays(world: World) -> None:
    customer = _a14(world)
    assert MONTH == date(2026, 6, 1)
    limit = LIMITS["special_retail" if customer.sub_channel == "special_retail" else customer.channel]
    order = 0.3 * limit
    earlier = MONTH - timedelta(days=20)
    _owes(world, customer, (earlier, 14, 0.3 * limit))
    assert credit_check(world, WORLD, customer, earlier + timedelta(days=10), order) == ("released", None)
    # June: two invoices within their terms and a big order pass the limit — twice, as the probe wants.
    _owes(world, customer, (MONTH - timedelta(days=6), 14, 0.4 * limit), (MONTH + timedelta(days=4), 14, 0.35 * limit))
    for day in (MONTH + timedelta(days=7), MONTH + timedelta(days=11)):
        assert credit_check(world, WORLD, customer, day, order) == ("held", "credit_limit")
    world.open_invoices = {}  # paid
    assert credit_check(world, WORLD, customer, MONTH + timedelta(days=14), order) == ("released", None)


@pytest.mark.parametrize(("late", "expected"), [(OVERDUE, ("released", None)), (OVERDUE + 1, ("held", "overdue"))])
def test_an_invoice_more_than_the_days_overdue_holds_the_order(
    world: World, late: int, expected: tuple[str, str | None]
) -> None:
    customer = _a14(world)
    day = date(2025, 5, 20)
    _owes(world, customer, (day - timedelta(days=14 + late), 14, 100.0))
    assert credit_check(world, WORLD, customer, day, 100.0) == expected


def test_overdue_is_the_reason_when_both_hold(world: World) -> None:
    customer = _a14(world)
    day = date(2025, 5, 20)
    _owes(world, customer, (day - timedelta(days=60), 14, 10 * LIMITS["small_retail"]))
    assert credit_check(world, WORLD, customer, day, 100.0) == ("held", "overdue")


def test_the_limit_is_the_terms_one_and_a_channel_without_one_is_never_over_it(world: World) -> None:
    day = date(2025, 5, 20)
    station = next(c for c in world.customers.values() if c.sub_channel == "special_retail")
    assert credit_check(world, WORLD, station, day, LIMITS["special_retail"] + 1) == ("held", "credit_limit")
    assert credit_check(world, WORLD, station, day, LIMITS["special_retail"]) == ("released", None)
    web = next(c for c in world.customers.values() if c.channel == "web")
    assert "web" not in LIMITS
    assert credit_check(world, WORLD, web, day, 1e9) == ("released", None)
    other = replace(_a14(world), customer_id=-1)
    _owes(world, other, (day, 14, 10 * LIMITS["small_retail"]))  # someone else's debt
    assert credit_check(world, WORLD, _a14(world), day, 100.0) == ("released", None)


def test_an_order_s_value_is_its_invoice_prices_times_its_cases() -> None:
    lines = [
        Priced(1, 3, 18, 612.0, 178.88, 536.64, 433.12, False, ()),
        Priced(2, 0.167, 1, 100.0, 0.0, 0.0, 100.0, False, ()),
        Priced(1, 2, 12, 612.0, 612.0, 1224.0, 0.0, True, ()),
    ]
    assert order_value(lines) == 1316.06  # 3 × 433.12 + 0.167 × 100.00; the free line adds nothing
