"""Individual prices (G1 S1.3 T4, `01` §5.3): a KAM's customer — big retail, gas KA, vending, HoReCa — buys at its
individual price, the SKU's own or the line's percentage off the channel list; the discount row takes the list down to
it (kind individual, from the sales budget). Small retail never has one."""

from __future__ import annotations

from datetime import date

import pytest

from hartland_sim.ids import row_id
from hartland_sim.loop.pricing import Term, apply, individual
from hartland_sim.setup.run import Setup
from hartland_sim.world import Agreement, Checkpoint, Customer, World

DAY = date(2023, 2, 15)
END = date(2023, 3, 31)


@pytest.fixture
def world(opening: Setup) -> World:
    return Checkpoint.of(opening.world).restore()


def _customer(world: World, channel: str) -> Customer:
    return next(c for c in world.customers.values() if c.channel == channel and c.active_on(DAY))


def _sku(opening: Setup, line: str) -> tuple[int, float]:
    """A SKU of `line` sold on DAY, with its list price in big retail."""
    listed = opening.world.channel_lists["big_retail"].prices
    sku = next(s for s in opening.portfolio.skus if s.line == line and s.sku_id in listed)
    return sku.sku_id, listed[sku.sku_id].case


def _add(world: World, agreement: Agreement) -> int:
    agreement_id = row_id("price_agreement", code=agreement.code)
    world.agreements[agreement_id] = agreement
    return agreement_id


@pytest.mark.parametrize("channel", ["big_retail", "gas_ka", "vending", "horeca"])
def test_a_kam_customer_s_sku_price_takes_the_list_down_to_it(opening: Setup, world: World, channel: str) -> None:
    customer = _customer(world, channel)
    sku_id, _ = _sku(opening, "GOLD")
    listed = world.channel_lists[channel].prices[sku_id].case
    price = round(listed * 0.9, 2)
    agreement = Agreement(
        customer.customer_id, "GOLD", 10, DAY, END, "AG-2023-00007", "individual_price", 5, sku_id, price
    )
    agreement_id = _add(world, agreement)
    term = individual(world, customer, sku_id, "GOLD", listed, DAY)
    assert term == Term("individual", "AG-2023-00007", "sales", price_case=price, price_agreement_id=agreement_id)
    (row,) = apply(listed, 4, [term])
    assert (row.kind, row.code, row.budget, row.pct) == ("individual", "AG-2023-00007", "sales", None)
    assert row.amount_case == round(listed - price, 2)
    assert row.amount_line == round(4 * (listed - price), 2)
    assert row.price_agreement_id == agreement_id and row.promotion_id is None


def test_a_line_s_percentage_prices_each_of_its_skus_and_a_sku_s_own_price_wins(opening: Setup, world: World) -> None:
    chain = _customer(world, "big_retail")
    sku_id, listed = _sku(opening, "GOLD")
    line_wide = Agreement(chain.customer_id, "GOLD", 6, DAY, END, "AG-2023-00008", "individual_price", 5)
    line_id = _add(world, line_wide)
    term = individual(world, chain, sku_id, "GOLD", listed, DAY)
    assert term == Term(
        "individual", "AG-2023-00008", "sales", price_case=round(listed * 0.94, 2), price_agreement_id=line_id
    )
    own = Agreement(chain.customer_id, "GOLD", 3, DAY, END, "AG-2023-00009", "individual_price", 5, sku_id, listed - 20)
    own_id = _add(world, own)
    assert individual(world, chain, sku_id, "GOLD", listed, DAY) == Term(
        "individual", "AG-2023-00009", "sales", price_case=listed - 20, price_agreement_id=own_id
    )
    other_id, other_price = next(
        (s.sku_id, world.channel_lists["big_retail"].prices[s.sku_id].case)
        for s in opening.portfolio.skus
        if s.line == "GOLD" and s.sku_id != sku_id and s.sku_id in world.channel_lists["big_retail"].prices
    )
    found = individual(world, chain, other_id, "GOLD", other_price, DAY)
    assert found is not None and found.code == "AG-2023-00008"


def test_small_retail_never_has_an_individual_price(opening: Setup, world: World) -> None:
    shop = _customer(world, "small_retail")
    sku_id, listed = _sku(opening, "GOLD")
    _add(world, Agreement(shop.customer_id, "GOLD", 6, DAY, END, "AG-2023-00010", "individual_price", 5))
    assert individual(world, shop, sku_id, "GOLD", listed, DAY) is None


def test_no_term_outside_its_validity_or_from_an_agreed_discount(opening: Setup, world: World) -> None:
    chain = _customer(world, "big_retail")
    sku_id, listed = _sku(opening, "GOLD")
    _add(world, Agreement(chain.customer_id, "GOLD", 6, DAY, END, "AG-2023-00011", "agreed_discount", 5))
    assert individual(world, chain, sku_id, "GOLD", listed, DAY) is None
    _add(world, Agreement(chain.customer_id, "GOLD", 6, DAY, END, "AG-2023-00012", "individual_price", 5))
    assert individual(world, chain, sku_id, "GOLD", listed, date(2023, 4, 1)) is None
    assert individual(world, chain, sku_id, "CLASSIC", listed, DAY) is None


def test_an_individual_price_above_the_price_reached_is_no_row(opening: Setup, world: World) -> None:
    chain = _customer(world, "big_retail")
    sku_id, listed = _sku(opening, "GOLD")
    _add(
        world, Agreement(chain.customer_id, "GOLD", 0, DAY, END, "AG-2023-00013", "individual_price", 5, sku_id, listed)
    )
    term = individual(world, chain, sku_id, "GOLD", listed, DAY)
    assert term is not None and apply(listed, 2, [term]) == ()
