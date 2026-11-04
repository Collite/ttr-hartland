"""The agreed discount (G1 S1.3 T3, `01` §5.2, `04` S6 as amended 11-04, S1.5 T2): who grants an ask and how a
discipline chooses (one chance for every granter) and sizes it,
the cap (the line's maximum × the phase's concession), the quarter's budget (refused past it, reserved within it, at
the carried line's expected gross and the discount's own lift), the refusals, the agreements' index, and the running
agreement's term on the order, never above the line's maximum."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import quarter_end
from hartland_sim.ids import row_id
from hartland_sim.loop.pricing import (
    Refusal,
    Rules,
    Term,
    agreed,
    agreement_row,
    chance,
    expected_gross,
    grant,
)
from hartland_sim.rng import rng
from hartland_sim.setup.run import Setup
from hartland_sim.world import Agreement, Budget, Checkpoint, Customer, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
S6 = STORIES.story("S6").parameters
DAY = date(2023, 2, 15)  # in the opening quarter, the one with budgets
QUARTER = date(2023, 1, 1)
SHARES = {str(phase): share for phase, share in WORLD.policies.lifecycle.concession_share.items()}


@pytest.fixture
def world(opening: Setup) -> World:
    return Checkpoint.of(opening.world).restore()


@pytest.fixture
def rules(opening: Setup) -> Rules:
    return Rules.of(opening.world, WORLD, STORIES)


def _rep(world: World, area: str) -> int:
    duty = world.duty("rep_area", area, DAY)
    assert duty is not None
    return duty.employee_id


def _asks(world: World, area: str) -> list[tuple[Customer, str]]:
    """Every (customer, line it buys) of `area`'s active small-retail customers."""
    demand = world.demand
    assert demand is not None
    index = {int(customer): c for c, customer in enumerate(demand.customer_ids)}
    out = []
    for customer in world.customers.values():
        if customer.channel == "small_retail" and customer.area == area and customer.active_on(DAY):
            carried = demand.carried[index[customer.customer_id]]
            out += [(customer, line.code) for line, buys in zip(demand.lines, carried, strict=True) if buys]
    return out


def _cap(rules: Rules, customer: Customer, line: str) -> float:
    return rules.max_pct(line) * SHARES.get(customer.phase_on(DAY), 0.0)


def _plenty(world: World, employee: int) -> None:
    world.budgets[employee, QUARTER] = Budget(1e12)


def test_a_disciplined_rep_grants_the_elastic_asks_below_the_cap(world: World, rules: Rules) -> None:
    employee = _rep(world, "SJC-02")
    assert world.actors[employee].skills.discount_discipline == 1.4
    _plenty(world, employee)
    high = S6["elasticity"]["high"]
    granted = {True: [0, 0], False: [0, 0]}  # elastic? → [granted, asked]
    for customer, line in _asks(world, "SJC-02"):
        if _cap(rules, customer, line) <= 0:
            continue
        answer = grant(world, rules, customer.customer_id, line, DAY)
        elastic = bool(rules.elasticity[rules.customers[customer.customer_id], rules.lines[line]] == high)
        granted[elastic][1] += 1
        if isinstance(answer, Agreement):
            granted[elastic][0] += 1
            low, top = S6["disciplined_share_of_max"]
            cap = _cap(rules, customer, line)
            assert low * cap - 0.05 <= answer.discount_pct <= top * cap + 0.05 < cap + 0.05
            assert (answer.kind, answer.employee_id) == ("agreed_discount", employee)
        else:
            assert answer.reason == "declined"
    assert granted[True][1] >= 50 and granted[False][1] >= 100
    assert granted[True][0] == granted[True][1]  # (high ÷ high) ^ anything = 1
    assert granted[False][0] / granted[False][1] <= 0.05  # (0.4 ÷ 3.0) ^ 2.7 ≈ 0.004


def test_an_undisciplined_rep_grants_every_ask_flat(world: World, rules: Rules) -> None:
    employee = _rep(world, "SMS-02")
    assert world.actors[employee].skills.discount_discipline == 0.5
    _plenty(world, employee)
    low, top = S6["undisciplined_discount_pct"]
    asked = inelastic = 0
    for customer, line in _asks(world, "SMS-02"):
        cap = _cap(rules, customer, line)
        if cap <= 0:
            continue
        answer = grant(world, rules, customer.customer_id, line, DAY)
        assert isinstance(answer, Agreement), answer
        assert low <= answer.discount_pct <= top or answer.discount_pct == cap
        assert answer.discount_pct <= cap
        asked += 1
        inelastic += bool(rules.elasticity[rules.customers[customer.customer_id], rules.lines[line]] < 1)
    assert asked >= 100 and inelastic >= 50  # at the range's floor a low elasticity is granted too, never declined


def test_every_granter_grants_with_the_chance_of_its_discipline() -> None:
    """R2 (Bora 11-04): (e ÷ e_high) ^ (γ × (discipline − the range's floor)), γ = 3 — to whoever asks at the floor,
    a decline of a low elasticity likelier with each step of discipline, a high elasticity always granted."""
    high, low = S6["elasticity"]["high"], S6["elasticity"]["low"]
    floor, top = S6["discipline_range"]
    assert (S6["disciplined_choice_exponent"], floor, top) == (3, 0.5, 1.4)
    assert chance(low, floor, S6) == chance(high, floor, S6) == 1.0
    assert chance(low, 1.4, S6) == pytest.approx((low / high) ** 2.7)
    steps = [chance(low, floor + k / 10, S6) for k in range(10)]  # 0.5 … 1.4
    assert all(earlier > later for earlier, later in zip(steps, steps[1:], strict=False))
    assert all(chance(high, floor + k / 10, S6) == 1.0 for k in range(10))


@pytest.mark.parametrize("area", ["SJC-02", "VCH-05", "SMS-02"])
def test_a_grant_s_first_draw_decides_it_and_the_discipline_sizes_it(world: World, rules: Rules, area: str) -> None:
    """The draws of S1.4: stream `pricing_agree`, key ("grant", customer, line, day) — the first decides against
    `chance`, the next sizes the grant (a share of the cap from `disciplined_from`, else a flat 5–8 %)."""
    employee = _rep(world, area)
    discipline = world.actors[employee].skills.discount_discipline
    _plenty(world, employee)
    answers = {True: 0, False: 0}
    for customer, line in _asks(world, area):
        cap = _cap(rules, customer, line)
        if cap <= 0:
            continue
        hidden = rules.elasticity[rules.customers[customer.customer_id], rules.lines[line]]
        draw = rng("pricing_agree", "grant", customer.customer_id, line, DAY)
        granted = draw.random() < chance(hidden, discipline, S6)
        answer = grant(world, rules, customer.customer_id, line, DAY)
        answers[granted] += 1
        if not granted:
            assert answer == Refusal(customer.customer_id, line, DAY, "declined", employee)
            continue
        assert isinstance(answer, Agreement)
        if discipline >= S6["disciplined_from"]:
            assert answer.discount_pct == round(cap * draw.uniform(*S6["disciplined_share_of_max"]), 1)
        else:
            assert answer.discount_pct == min(round(draw.uniform(*S6["undisciplined_discount_pct"]), 1), cap)
    assert answers[True] >= 20 and (answers[False] == 0) == (discipline == 0.5)


def test_a_grant_reserves_its_expected_cost_and_one_past_the_budget_is_refused(world: World, rules: Rules) -> None:
    employee = _rep(world, "SMS-02")
    customer, line = next((c, line) for c, line in _asks(world, "SMS-02") if _cap(rules, c, line) > 0)
    budget = world.budgets[employee, QUARTER]
    assert budget.reserved == 0 and budget.used_amount == 0
    world.budgets[employee, QUARTER] = Budget(budget.budget_amount, used_amount=budget.budget_amount - 1.0)
    refused = grant(world, rules, customer.customer_id, line, DAY)
    assert refused == Refusal(customer.customer_id, line, DAY, "budget", employee)
    assert world.agreements == {} and world.budgets[employee, QUARTER].reserved == 0
    world.budgets[employee, QUARTER] = budget
    agreement = grant(world, rules, customer.customer_id, line, DAY)
    assert isinstance(agreement, Agreement)
    expected = expected_gross(world, rules, customer.customer_id, line, DAY, quarter_end(DAY))
    lift = 1 + rules.elasticity[rules.customers[customer.customer_id], rules.lines[line]] * agreement.discount_pct / 100
    assert agreement.budget_charge == round(agreement.discount_pct / 100 * expected * lift, 2) > 0
    assert (budget.reserved, budget.used_amount) == (agreement.budget_charge, 0)  # S4.1 posts it at invoicing
    assert (agreement.valid_from, agreement.valid_to) == (DAY, date(2023, 3, 31))


def test_the_charge_follows_the_line_s_price_index_as_the_budget_does(world: World, rules: Rules) -> None:
    """review-221 R4: the budgets are at the drifted list, so the charges against them are too."""
    customer, line = next((c, line) for c, line in _asks(world, "SMS-02") if _cap(rules, c, line) > 0)
    drifted = Checkpoint.of(world).restore()
    drifted.price_index = {**drifted.price_index, line: 1.3}
    flat = expected_gross(world, rules, customer.customer_id, line, DAY, quarter_end(DAY))
    moved = expected_gross(drifted, rules, customer.customer_id, line, DAY, quarter_end(DAY))
    assert world.price_index[line] == 1.0 and moved == pytest.approx(1.3 * flat)
    one, other = (grant(w, rules, customer.customer_id, line, DAY) for w in (world, drifted))
    assert isinstance(one, Agreement) and isinstance(other, Agreement) and one.discount_pct == other.discount_pct
    lift = 1 + rules.elasticity[rules.customers[customer.customer_id], rules.lines[line]] * one.discount_pct / 100
    assert one.budget_charge == round(one.discount_pct / 100 * flat * lift, 2)
    assert other.budget_charge == round(one.discount_pct / 100 * moved * lift, 2)
    assert one.budget_charge is not None and other.budget_charge is not None
    assert abs(other.budget_charge - 1.3 * one.budget_charge) <= 0.01


def test_an_ask_without_a_budget_a_phase_or_a_granter_is_refused(world: World, rules: Rules) -> None:
    employee = _rep(world, "SMS-02")
    customer, line = _asks(world, "SMS-02")[0]
    later = date(2023, 4, 3)  # the quarters after the opening have no budget until S1.4's quarter-end rolls them
    no_budget = Refusal(customer.customer_id, line, later, "no_budget", employee)
    assert grant(world, rules, customer.customer_id, line, later) == no_budget
    held = WORLD.policies.lifecycle.concession_share
    phase = next(phase for phase in held if phase == customer.phase_on(DAY))
    lifecycle = replace(WORLD.policies.lifecycle, concession_share={**held, phase: 0.0})  # its phase allows none
    strict = replace(rules, world=replace(WORLD, policies=replace(WORLD.policies, lifecycle=lifecycle)))
    no_phase = Refusal(customer.customer_id, line, DAY, "phase", employee)
    assert grant(world, strict, customer.customer_id, line, DAY) == no_phase
    web = next(c for c in world.customers.values() if c.channel == "web")
    no_granter = Refusal(web.customer_id, "CLASSIC", DAY, "no_granter")
    assert grant(world, rules, web.customer_id, "CLASSIC", DAY) == no_granter


def test_an_ask_of_a_customer_not_active_or_on_a_line_it_does_not_buy_reserves_nothing(
    world: World, rules: Rules
) -> None:
    kam = {duty.target: duty.employee_id for duty in world.duties if duty.kind == "kam_customer" and duty.on(DAY)}
    venue = next(
        c for c in world.customers.values() if c.code in kam and c.channel == "horeca" and not c.active_on(DAY)
    )
    world.budgets[kam[venue.code], QUARTER] = Budget(1e12)
    inactive = Refusal(venue.customer_id, "CLASSIC", DAY, "inactive", kam[venue.code])
    assert grant(world, rules, venue.customer_id, "CLASSIC", DAY) == inactive
    shop, _ = _asks(world, "SMS-02")[0]
    bought = {line for c, line in _asks(world, "SMS-02") if c is shop}
    line = next(line for line in rules.lines if line not in bought)
    not_bought = Refusal(shop.customer_id, line, DAY, "not_bought", _rep(world, "SMS-02"))
    assert grant(world, rules, shop.customer_id, line, DAY) == not_bought
    assert expected_gross(world, rules, shop.customer_id, line, DAY, quarter_end(DAY)) == 0
    assert world.agreements == {} and all(budget.reserved == 0 for budget in world.budgets.values())


def test_grants_extend_the_agreements_index_and_a_restored_world_finds_the_same(world: World, rules: Rules) -> None:
    _plenty(world, _rep(world, "SMS-02"))
    asks = [(c.customer_id, line) for c, line in _asks(world, "SMS-02") if _cap(rules, c, line) > 0][:200]
    assert world.agreements_on(*asks[0], DAY) == []
    index = world.__dict__["_agreement_index"][2]
    made = [answer for answer in (grant(world, rules, *ask, DAY) for ask in asks) if isinstance(answer, Agreement)]
    assert len(made) >= 100
    assert world.__dict__["_agreement_index"][2] is index  # extended by each grant, never built again
    found = {ask: world.agreements_on(*ask, DAY) for ask in asks}
    assert sum(len(running) for running in found.values()) == len(made)
    restored = Checkpoint.of(world).restore()
    assert "_agreement_index" not in restored.__dict__
    assert {ask: restored.agreements_on(*ask, DAY) for ask in asks} == found
    agreement_id = next(iter(world.agreements))
    with pytest.raises(ValueError, match="in the World already"):
        world.add_agreement(agreement_id, world.agreements[agreement_id])


def test_a_running_agreement_holds_the_line_and_the_codes_number_within_the_year(world: World, rules: Rules) -> None:
    _plenty(world, _rep(world, "SMS-02"))
    asks = [(c, line) for c, line in _asks(world, "SMS-02") if _cap(rules, c, line) > 0][:3]
    made = [grant(world, rules, c.customer_id, line, DAY) for c, line in asks]
    assert [a.code for a in made if isinstance(a, Agreement)] == ["AG-2023-00001", "AG-2023-00002", "AG-2023-00003"]
    assert list(world.agreements) == [row_id("price_agreement", code=a.code) for a in made if isinstance(a, Agreement)]
    customer, line = asks[0]
    again = grant(world, rules, customer.customer_id, line, date(2023, 3, 1))
    assert isinstance(again, Refusal) and again.reason == "held"


def test_the_same_ask_on_the_same_day_gets_the_same_answer(opening: Setup, rules: Rules) -> None:
    answers = []
    for _ in range(2):
        world = Checkpoint.of(opening.world).restore()
        customer, line = _asks(world, "SJC-02")[5]
        answers.append(grant(world, rules, customer.customer_id, line, DAY))
    assert answers[0] == answers[1]


def test_a_kam_grants_individual_prices(world: World, rules: Rules) -> None:
    """The KAMs' customers ask on three lines each: what the KAMs grant (by `chance`, as a rep) is individual prices."""
    codes = {c.code: c for c in world.customers.values()}
    made = []
    for duty in [d for d in world.duties if d.kind == "kam_customer" and d.on(DAY)][:30]:
        world.budgets[duty.employee_id, QUARTER] = Budget(1e12)
        customer = codes[duty.target]
        answers = [grant(world, rules, customer.customer_id, line, DAY) for line in ("CLASSIC", "GOLD", "PROFIKAVA")]
        made += [(a, duty.employee_id) for a in answers if isinstance(a, Agreement)]
    assert made and all(a.kind == "individual_price" and a.employee_id == kam for a, kam in made)
    assert all(a.discount_pct <= rules.max_pct(a.line) for a, _ in made)


def test_the_order_takes_the_running_agreed_discount_up_to_the_line_s_maximum(world: World, rules: Rules) -> None:
    customer, _ = _asks(world, "SMS-02")[0]
    employee = _rep(world, "SMS-02")
    gold = Agreement(customer.customer_id, "GOLD", 12.0, DAY, date(2023, 3, 31), "AG-2023-00042", employee_id=employee)
    gold_id = row_id("price_agreement", code=gold.code)
    world.add_agreement(gold_id, gold)
    term = agreed(world, rules, customer, "GOLD", DAY)
    assert term == Term("agreed", "AG-2023-00042", "sales", pct=5.0, price_agreement_id=gold_id)
    assert agreed(world, rules, customer, "GOLD", date(2023, 4, 1)) is None
    assert agreed(world, rules, customer, "CLASSIC", DAY) is None
    chain = next(c for c in world.customers.values() if c.channel == "big_retail")
    world.add_agreement(1, Agreement(chain.customer_id, "GOLD", 4.0, DAY, DAY, "AG-2023-00043", employee_id=employee))
    assert agreed(world, rules, chain, "GOLD", DAY) is None


def test_the_agreement_renders_its_row(world: World) -> None:
    gold = Agreement(7, "GOLD", 4.5, DAY, date(2023, 3, 31), "AG-2023-00042", employee_id=11, budget_charge=812.5)
    assert agreement_row(99, gold) == {
        "price_agreement_id": 99,
        "code": "AG-2023-00042",
        "customer_id": 7,
        "sku_id": None,
        "product_line_id": row_id("product_line", code="GOLD"),
        "kind": "agreed_discount",
        "discount_pct": 4.5,
        "price_case": None,
        "valid_from": DAY,
        "valid_to": date(2023, 3, 31),
        "agreed_by_employee_id": 11,
        "budget_charge": 812.5,
    }
    with pytest.raises(ValueError, match="who agreed it"):
        agreement_row(99, Agreement(7, "GOLD", 4.5, DAY, DAY, "AG-2023-00043"))
