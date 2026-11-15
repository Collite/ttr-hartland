"""S5's opt-in pilot (G2 S2.6 T1–T3, `loop.pilot`; design08 §§3–4, NH-D22) — the five behaviour groups: the pair
assignment frozen before any exposure; no PoS at a holdout; one arm per customer, kept through acquisition, loss and a
restart; the declared score, the all-manager gradient and the half budgets; complete audit rows, each pilot loan's
decision, and a month-end selection cancelled on its effective day. The `World` is the opening's run in the story
mode through the last PoS-free day's month-end (`assigned`), once; a case continues it from its checkpoint."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.io.csv_writer import CsvWriter
from hartland_sim.loop import pilot, pos
from hartland_sim.loop.day import Loop
from hartland_sim.loop.pilot import Mapped, Pilot
from hartland_sim.loop.pricing import channel_list, standard_list
from hartland_sim.loop.warmup import Output
from hartland_sim.protocol import Protocol
from hartland_sim.rng import uniforms
from hartland_sim.setup.opening import state_rows
from hartland_sim.setup.run import Setup
from hartland_sim.world import Checkpoint, World

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
PROTOCOL = Protocol.load()
END, FIRST = date(2023, 2, 28), date(2023, 3, 1)  # the last PoS-free day (the assignment's month-end) and the next
MARCH = date(2023, 3, 31)
H = Fraction(1461, 8)  # (3 + 9) ÷ 2 months × 365.25 ÷ 12 days


def _loop(opening: Setup, world: World, run: Path, first_day: date = FIRST) -> tuple[Loop, CsvWriter]:
    writer = CsvWriter(run)
    return Loop.of(world, WORLD, STORIES, opening.portfolio.skus, Output(writer, first_day)), writer


@pytest.fixture(scope="module")
def assigned(opening: Setup, tmp_path_factory: pytest.TempPathFactory) -> Checkpoint:
    """The opening run as a story pilot through `END`: assigned, its first placements lent from `FIRST`."""
    run = tmp_path_factory.mktemp("assigned")
    world = Checkpoint.of(opening.world).restore()
    world.pilot = pilot.start("story", PROTOCOL)
    loop, writer = _loop(opening, world, run)
    with writer:
        loop.until(END - timedelta(days=1))
        assert not world.pilot.assigned and not world.loans and set(world.display_budget.values()) == {0.0}
        loop.run(END)
    Checkpoint.of(loop.world).save(run)
    return Checkpoint.load(run, END)


@pytest.fixture
def world(assigned: Checkpoint) -> World:
    return assigned.restore()


def _pilot(world: World) -> Pilot:
    assert world.pilot is not None
    return world.pilot


def _arm(world: World, arm: str) -> set[int]:
    return {c for c, held in _pilot(world).assigned.items() if held.mapped.arm == arm}


def _decided(world: World, day: date = END) -> dict[int, pilot.Decision]:
    return {point: d for (on, point), d in _pilot(world).decisions.items() if on == day}


def _lose(monkeypatch: pytest.MonkeyPatch, day: date, *customers: int) -> None:
    """The churn draws of `customers` on `day` forced to 0, so each is lost that day (`test_pos_chronology`'s)."""
    lost = np.array(customers, dtype=np.int64)

    def forced(stream: str, *key: object, ids: np.ndarray) -> np.ndarray:
        out = uniforms(stream, *key, ids=ids)
        return np.where(np.isin(ids, lost), 0.0, out) if (stream, key) == ("churn", (day,)) else out

    monkeypatch.setattr("hartland_sim.loop.acquisition.uniforms", forced)


# ── 1. the pair assignment, frozen before any exposure ─────────────────────


def test_every_registered_small_retail_customer_is_assigned_in_pairs_of_one_region(world: World) -> None:
    held = _pilot(world).assigned
    small = {c.customer_id for c in world.customers.values() if c.channel == "small_retail"}
    assert set(held) == small and len(held) == 8160
    pairs: dict[int, list[Mapped]] = {}
    for a in held.values():
        if a.mapped.arm == "outside":
            assert a.mapped.pair_id is None and a.mapped.outside_reason in ("unpaired", "cross_region")
        else:
            pairs.setdefault(a.mapped.pair_id, []).append(a.mapped)  # type: ignore[arg-type]
    assert all(sorted(m.arm for m in two) == ["holdout", "programme"] for two in pairs.values())
    assert all(len({m.region for m in two}) == 1 for two in pairs.values())
    potential = {c.customer_id for c in world.customers.values() if c.channel == "small_retail" and c.since is None}
    assert potential and all(held[c].baseline == 0 for c in potential)  # a zero baseline is a valid one


def test_the_pairs_are_consecutive_in_the_declared_order_and_the_coin_is_keyed(world: World) -> None:
    p = _pilot(world)
    for region in ("PHA", "VCH"):
        mine = [(c, a) for c, a in p.assigned.items() if a.mapped.region == region and a.mapped.arm != "outside"]
        mine.sort(key=lambda ca: (_portfolio(ca[1].mapped), ca[1].baseline, ca[0]))
        order = [a.mapped.pair_id for _, a in mine]
        assert order[0::2] == order[1::2]  # each pair two consecutive customers
        ids = np.array(order[0::2], dtype=np.int64)
        coins = uniforms("pos", "s5-pilot-v1", 1, region, ids=ids)
        assert [a.mapped.arm for _, a in mine[0::2]] == ["programme" if c < 0.5 else "holdout" for c in coins]


def _portfolio(m: Mapped) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(outlet.split("/")) for outlet in m.portfolio_key.split(";"))


def test_nothing_is_lent_or_handed_out_before_the_assignment_day(world: World) -> None:
    p = _pilot(world)
    assert all(loan.lent_on == FIRST for loan in world.loans.values())
    member = world.customers[next(iter(_arm(world, "programme")))]
    assert not pilot.hands_out(p, member, END) and pilot.hands_out(p, member, FIRST)
    assert pilot.hands_out(p, member, date(2025, 12, 31)) and not pilot.hands_out(p, member, date(2026, 1, 1))


def test_the_pilot_gates_give_aways_in_its_universe_only(world: World) -> None:
    """⚑R295-A: a key account's station hands out as the legacy build does, a holdout never, a no-PoS mode nowhere."""
    p = _pilot(world)
    holdout = world.customers[next(iter(_arm(world, "holdout")))]
    station = next(c for c in world.customers.values() if c.channel == "gas_ka")
    day = date(2024, 1, 2)
    assert pilot.hands_out(p, station, day) and not pilot.hands_out(p, holdout, day)
    reference = replace(p, mode="no-pos")
    assert not pilot.hands_out(reference, station, day) and not pilot.hands_out(reference, holdout, day)
    pilot.handed(p, station, day, 12.5)
    assert day.toordinal() not in p.handouts  # only the universe's give-aways are its history


def test_an_odd_customer_is_outside_and_another_mode_takes_the_story_map(world: World) -> None:
    p = _pilot(world)
    demand = world.demand
    assert demand is not None
    terms = pos.Terms.of(WORLD, STORIES, demand.lines)
    pha = [c for c in world.customers.values() if c.channel == "small_retail" and _region(terms, c.area) == "PHA"]
    odd = pha[: 2 * (len(pha) // 2) - 1]
    drawn = pilot.draw(p, odd, demand.spots, terms)
    alone = [c for c, m in drawn.items() if m.arm == "outside"]
    assert len(alone) == 1 and drawn[alone[0]].outside_reason == "unpaired"
    story_map = {c: a.mapped for c, a in p.assigned.items()}
    other = pilot.start("full-null", PROTOCOL, story_map)
    other.baseline = {c: 7 for c in story_map}  # its own baselines
    pilot.assign(world, other, terms)
    assert other.digest == p.digest and {c: a.mapped for c, a in other.assigned.items()} == story_map
    assert {a.baseline for a in other.assigned.values()} == {7}
    with pytest.raises(ValueError, match="differ in 1"):
        pilot.assign(world, pilot.start("no-pos", PROTOCOL, dict(list(story_map.items())[1:])), terms)
    with pytest.raises(ValueError, match="takes the story pre-pilot"):
        pilot.start("skill-neutral", PROTOCOL)


def _region(terms: pos.Terms, area: str | None) -> str | None:
    return terms.region_of.get(area) if area else None


# ── 2. no PoS at a holdout ─────────────────────────────────────────────────


def test_a_holdout_gets_no_display_no_give_away_and_no_display_prompt(
    opening: Setup, assigned: Checkpoint, tmp_path: Path
) -> None:
    world = assigned.restore()
    loop, writer = _loop(opening, world, tmp_path)
    with writer:
        loop.until(MARCH + timedelta(days=1))  # March written, its month-end placed from April
    others = _arm(world, "holdout") | _arm(world, "outside")
    assert {loan.customer_id for loan in world.loans.values()}.isdisjoint(others)
    assert {d.customer_id for d in _pilot(world).decisions.values()}.isdisjoint(others)
    customer_of = {r["visit_id"]: int(r["customer_id"]) for r in _rows(tmp_path / "visit.csv") if r["customer_id"]}
    given = {customer_of[r["visit_id"]] for r in _rows(tmp_path / "pos_handout.csv")}
    small = {c for c in given if world.customers[c].channel == "small_retail"}  # the pilot's universe (⚑R295-A)
    assert small and small <= _arm(world, "programme")
    assert {world.customers[c].channel for c in given - small} == {"gas_ka"}  # a key account's stations, as legacy
    prompted = {int(r["customer_id"]) for r in _rows(tmp_path / "rep_prompt.csv") if r["kind"] == "pos_placement"}
    assert prompted and prompted <= _arm(world, "programme")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


# ── 3. one arm per customer: outlets, acquisition, loss, restart ──────────


def test_a_customer_has_one_arm_whatever_its_stores_and_one_across_regions_is_outside(world: World) -> None:
    demand = world.demand
    assert demand is not None
    terms = pos.Terms.of(WORLD, STORIES, demand.lines)
    one, other = [
        c for c in world.customers.values() if c.channel == "small_retail" and _region(terms, c.area) == "PHA"
    ][:2]
    far = next(s for s in demand.spots if s.region == "VCH" and s.channel == "small_retail")
    mine = [s for s in demand.spots if s.customer_id == one.customer_id]
    twice = [*mine, replace(mine[0], delivery_point_id=1)]
    drawn = pilot.draw(_pilot(world), [one, other], twice, terms)
    assert drawn[one.customer_id].portfolio_key.count(";") == 1  # both stores in its portfolio, one arm
    split = pilot.draw(_pilot(world), [one], [*mine, replace(far, customer_id=one.customer_id)], terms)
    assert split[one.customer_id] == Mapped(
        "PHA", "outside", None, "cross_region", split[one.customer_id].portfolio_key
    )


def test_the_arms_hold_through_acquisition_loss_and_a_restart(
    opening: Setup, assigned: Checkpoint, tmp_path: Path
) -> None:
    world = assigned.restore()
    before = {c: a.mapped for c, a in _pilot(world).assigned.items()}
    loop, writer = _loop(opening, world, tmp_path)
    with writer:
        loop.until(MARCH)
    p = _pilot(world)
    assert {c: a.mapped for c, a in p.assigned.items()} == before  # frozen
    won = [c for c in world.customers.values() if c.since is not None and FIRST < c.since <= MARCH]
    lost = [c for c in world.customers.values() if c.until is not None and FIRST <= c.until < MARCH]
    assert won and lost
    march = _decided(world, MARCH)
    deciding = {d.customer_id for d in march.values()}
    april = MARCH + timedelta(days=1)
    held = {loan.customer_id for loan in world.loans.values() if loan.lent_on < april and loan.returned_on is None}
    for c in won:  # a programme store acquired in March is a candidate at its month-end, a holdout never
        expected = pilot.programme(p, c.customer_id) and c.active_on(april) and c.customer_id not in held
        assert (c.customer_id in deciding) == expected
    assert all(pilot.programme(p, c.customer_id) == (before[c.customer_id].arm == "programme") for c in lost)
    assert not deciding & {c.customer_id for c in lost}
    restored = Checkpoint.of(world).restore()
    assert restored == world and _pilot(restored).assigned == p.assigned


# ── 4. the declared score, the all-manager gradient, half the budget ──────


def test_each_score_is_the_declared_one_from_its_snapshot(world: World) -> None:
    for d in _decided(world).values():
        assert d.forecast_days == 183 and d.fit == pytest.approx(
            world.display_uplift[d.delivery_point_id][_line(world, d)]
        )
        if d.material is None:
            assert (d.asset_cost, d.gain, d.cost, d.score, d.outcome) == (None, None, None, 0.0, "no_asset")
            continue
        gain = round(max(0.0, d.prior56_line / 100 / 56) * d.fit * float(H), 2)
        cost = round(d.asset_cost * float(H) / (36 * 30.4) + d.prior56_handouts / 100 / 56 * float(H), 2)  # type: ignore[operator]
        assert (d.gain, d.cost) == (gain, cost)
        assert d.score == (round(gain / cost, 8) if gain > 0 else 0.0)
        assert d.outcome != "nonpositive_score" or d.prior56_line <= 0


def _line(world: World, d: pilot.Decision) -> str:
    return next(iter(world.display_uplift[d.delivery_point_id]))


def test_every_manager_draws_by_score_to_the_skill_times_three_and_spends_half_the_budget(world: World) -> None:
    decided = _decided(world)
    assert Counter(d.outcome for d in decided.values())["selected"] > 0
    terms = pos.Terms.of(WORLD, STORIES, world.demand.lines)  # type: ignore[union-attr]
    for region in terms.regions:
        mine = [d for d in decided.values() if d.region == region]
        employee_id, skill, factor = terms.manager(world, region, FIRST)  # type: ignore[misc]
        assert {d.manager_id for d in mine} == {employee_id}
        ids = np.array([d.delivery_point_id for d in mine], dtype=np.int64)
        draw = uniforms("pos", "s5-pilot-v1", "place", region, FIRST, ids=ids)
        weight = np.array([d.score for d in mine]) ** (skill * 3)
        key = np.where(weight > 0, np.log(draw) / np.where(weight > 0, weight, 1.0), -np.inf)
        ranked = [mine[k] for k in np.argsort(-key, kind="stable") if np.isfinite(key[k])]
        reached = [d.outcome for d in ranked]
        stop = reached.index("no_budget") if "no_budget" in reached else len(reached)
        assert set(reached[:stop]) <= {"selected", "no_asset"} and set(reached[stop:]) <= {"no_budget"}
        spent = sum(world.assets[world.loans[d.loan_id].pos_asset_id].cost for d in ranked[:stop] if d.loan_id)
        half = WORLD.field.pos_budget[region] * factor * 0.5
        assert 0 < spent <= half and world.display_budget[region] == pytest.approx(half - spent, abs=0.005)


def test_a_missing_material_and_an_invalid_cost_are_no_offers(world: World) -> None:
    demand = world.demand
    assert demand is not None
    terms = pos.Terms.of(WORLD, STORIES, demand.lines)
    p = _pilot(world)
    history = pilot.History.of(p, END, {line.code: i for i, line in enumerate(demand.lines)}, WORLD)
    point, d = next((point, d) for point, d in _decided(world).items() if d.outcome == "selected")
    t = next(t for t in world.tourplans.values() if t.delivery_point_id == point)
    manager = terms.manager(world, d.region, FIRST)
    none = pilot.forecast(world, terms, history, t, d.region, manager, {})
    assert (none.material, none.asset_cost, none.gain, none.cost, none.score, none.outcome) == (
        None, None, None, None, 0.0, "no_asset"
    )  # fmt: skip
    assert none.prior56_line == d.prior56_line  # the line's history still kept
    free = next(i for i, a in world.assets.items() if a.material == d.material and a.region == d.region)
    world.assets[free] = replace(world.assets[free], cost=0.0)
    zero = pilot.forecast(world, terms, history, t, d.region, manager, {(d.region, d.material): free})  # type: ignore[dict-item]
    assert (zero.asset_cost, zero.cost, zero.score, zero.outcome) == (0.0, 0.0, 0.0, "ineligible")
    world.assets[free] = replace(world.assets[free], cost=math.nan)
    nan = pilot.forecast(world, terms, history, t, d.region, manager, {(d.region, d.material): free})  # type: ignore[dict-item]
    assert (nan.score, nan.outcome) == (0.0, "ineligible")
    for start in ({(d.region, d.material): free}, {}):  # a region without a manager decides nothing
        unheld = pilot.forecast(world, terms, history, t, d.region, None, start)  # type: ignore[arg-type]
        assert (unheld.manager_id, unheld.score, unheld.outcome) == (None, 0.0, "ineligible")


def test_the_pilot_ends_with_its_window(world: World, tmp_path: Path) -> None:
    """The window's last month-end lends nothing for the day after it: no decision, nil budgets."""
    demand = world.demand
    assert demand is not None
    terms = pos.Terms.of(WORLD, STORIES, demand.lines)
    held = len(_pilot(world).decisions)
    with CsvWriter(tmp_path) as writer:
        pilot.close(world, terms, date(2025, 12, 31), Output(writer, date(2100, 1, 1)))
    assert len(_pilot(world).decisions) == held and set(world.display_budget.values()) == {0.0}


# ── 5. complete audit rows, provenance, the effective day (NH-D22) ─────────


def test_every_programme_candidate_has_one_decision_and_every_loan_one_selection(world: World) -> None:
    p = _pilot(world)
    candidates = {
        t.delivery_point_id
        for t in (world.tourplans[i] for i in world.tourplan_of.values())
        if world.customers[t.customer_id].channel == "small_retail"
        and pilot.programme(p, t.customer_id)
        and world.customers[t.customer_id].active_on(FIRST)
    }
    decided = _decided(world)
    assert set(decided) == candidates
    selected = {d.loan_id: d for d in decided.values() if d.outcome == "selected" and d.loan_id is not None}
    assert (
        set(selected) == set(world.loans) and len(selected) == Counter(d.outcome for d in decided.values())["selected"]
    )
    for loan_id, d in selected.items():
        loan = world.loans[loan_id]
        assert (loan.customer_id, loan.delivery_point_id, loan.lent_on) == (d.customer_id, d.delivery_point_id, FIRST)
        assert loan.approved_by == d.manager_id and world.assets[loan.pos_asset_id].region == d.region
    assert all(d.loan_id is None for d in decided.values() if d.outcome != "selected")


def test_a_selection_cancelled_on_its_effective_day_keeps_its_decision(
    opening: Setup, assigned: Checkpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = assigned.restore()
    p = _pilot(world)
    selected = {point: d for point, d in _decided(world).items() if d.outcome == "selected"}
    point, victim = next((point, d) for point, d in selected.items() if not world.customers[d.customer_id].traits)
    kept = {k: replace(d) for k, d in p.decisions.items()}
    loan_id = victim.loan_id
    assert loan_id is not None
    asset = world.loans[loan_id].pos_asset_id
    budget = world.display_budget[victim.region]
    _lose(monkeypatch, FIRST, victim.customer_id)
    loop, writer = _loop(opening, world, tmp_path)
    with writer:
        loop.run(FIRST)
    held = p.decisions[(END, point)]
    assert (held.outcome, held.loan_id) == ("cancelled", None)
    assert replace(held, outcome="selected", loan_id=loan_id) == kept[(END, point)]  # its key, day and inputs
    assert sum(d.outcome == "cancelled" for d in p.decisions.values()) == 1
    assert loan_id not in world.loans and world.assets[asset].loan is None  # no loan; the asset in stock (S2.5)
    assert world.display_budget[victim.region] == round(budget + world.assets[asset].cost, 2)
    for k, d in p.decisions.items():  # the valid selections and the never-selected ones as they were
        if k != (END, point):
            assert d == kept[k]
    rows = state_rows(world, WORLD)["pos_placement_decision"]
    assert {r["loan_id"] for r in rows if r["loan_id"] is not None} <= set(world.loans)  # no dangling link
    (row,) = [r for r in rows if r["delivery_point_id"] == point]
    assert (row["outcome"], row["loan_id"], row["decision_day"]) == ("cancelled", None, END)


# ── T6: a restart at the month and the year edge, with a cancellation ──────


def _moved(opening: Setup, world: World, day: date) -> World:
    """`world` moved to `day`, its lists those of `day + 1`'s month (`test_pos_chronology._at`) and its observed
    history by as many days, so the next month-end forecasts from the same 56 days."""
    shift = (day - world.day).days
    skus = opening.portfolio.skus
    world.day = day
    following = day + timedelta(days=1)
    world.standard_list = standard_list(skus, following)
    world.channel_lists = {c: channel_list(world.standard_list, skus, WORLD, c, following) for c in world.channel_lists}
    p = _pilot(world)
    p.lines = {d + shift: held for d, held in p.lines.items()}
    p.handouts = {d + shift: held for d, held in p.handouts.items()}
    return world


def _written(*runs: Path) -> dict[str, list[str]]:
    """Each file's header and the rows of `runs` after it, in order (a run cut at a checkpoint, then resumed)."""
    out: dict[str, list[str]] = {}
    for run in runs:
        for path in sorted(run.glob("*.csv")):
            header, *rows = path.read_text(encoding="utf-8").splitlines()
            out.setdefault(path.name, [header]).extend(rows)
    return out


@pytest.mark.parametrize("edge", ["month", "year"])
def test_a_restart_at_the_edge_is_the_straight_run_cancellation_and_export_included(
    edge: str, opening: Setup, assigned: Checkpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The straight run over a month-end and its effective day — a selection's customer lost that day — and the run
    cut at the month-end's checkpoint, then restarted from it, reach the same `World`, the same state rows and the
    same files."""
    day = END if edge == "month" else date(2023, 12, 31)
    first_day = day + timedelta(days=1)
    if edge == "month":  # the month-end itself is the fixture's: run it again from its eve
        start = Checkpoint.of(_eve(opening, tmp_path / "before"))
    else:
        start = Checkpoint.of(_moved(opening, assigned.restore(), day - timedelta(days=1)))
    straight, eve, held, restarted = (tmp_path / name for name in ("straight", "eve", "held", "restarted"))
    loop, writer = _loop(opening, start.restore(), straight, first_day)
    with writer:
        loop.run(day)
        point, victim = next(
            (point, d)
            for point, d in _decided(loop.world, day).items()
            if d.outcome == "selected" and not loop.world.customers[d.customer_id].traits
        )
        _lose(monkeypatch, first_day, victim.customer_id)
        loop.run(first_day)
    cut, writer = _loop(opening, start.restore(), eve, first_day)
    with writer:
        cut.run(day)
    Checkpoint.of(cut.world).save(held)
    again, writer = _loop(opening, Checkpoint.load(held, day).restore(), restarted, first_day)
    with writer:
        again.run(first_day)
    assert again.world == loop.world
    assert _pilot(again.world).decisions[(day, point)].outcome == "cancelled"
    assert state_rows(again.world, WORLD) == state_rows(loop.world, WORLD)
    files = _written(straight)
    assert files == _written(eve, restarted)
    assert len(files["pos_handout.csv"]) > 1 or first_day == date(2024, 1, 1)  # New Year's Day: no visit at all


def _eve(opening: Setup, run: Path) -> World:
    """The opening run as a story pilot to the day before `END` (the `assigned` fixture's, stopped a day short)."""
    world = Checkpoint.of(opening.world).restore()
    world.pilot = pilot.start("story", PROTOCOL)
    loop, writer = _loop(opening, world, run)
    with writer:
        loop.until(END - timedelta(days=1))
    return loop.world
