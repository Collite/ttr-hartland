"""Setup steps 1 and 3 (G1 S1.1 T4): the reference tables and the organisation — counts, names and codes, the duties
with the two area moves, the hidden skills."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import pytest

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import read
from hartland_sim.setup.market import Market
from hartland_sim.setup.org import rep_areas
from hartland_sim.setup.run import Setup, run
from hartland_sim.world import OPEN, Actor

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
MOVES = STORIES.story("S3").parameters["area_move_days"]


@pytest.fixture(scope="module")
def setup(opening: Setup) -> Setup:
    return opening


def _rows(setup: Setup, table: str) -> list[dict[str, Any]]:
    return setup.rows(table)


def _by_slot(setup: Setup, prefix: str) -> dict[str, Actor]:
    return {slot.split(":", 1)[1]: actor for slot, actor in setup.org.actors.items() if slot.startswith(prefix)}


def test_the_reference_tables_count_as_the_files(setup: Setup) -> None:
    counts = {table: len(_rows(setup, table)) for table in ("legal_entity", "kraj", "okres", "site", "calendar")}
    assert counts == {"legal_entity": 4, "kraj": 14, "okres": 99, "site": 18, "calendar": 1826}
    assert Counter(site["kind"] for site in _rows(setup, "site")) == {"hq": 1, "factory": 3, "dc": 2, "shop": 12}
    assert len(_rows(setup, "area")) == 50
    assert len(_rows(setup, "region")) == 6
    assert len(_rows(setup, "area_okres")) == 98  # every okres but Praha, which its 22 districts replace
    praha = row_id("okres", code="CZ0100")
    assert sum(okres["parent_okres_id"] == praha for okres in _rows(setup, "okres")) == 22
    assert all(0.8 <= okres["affluence_index"] <= 1.3 for okres in _rows(setup, "okres"))


def test_the_org_units_are_the_five_and_their_teams_each_on_its_unit_s_cost_centre(setup: Setup) -> None:
    units = _rows(setup, "org_unit")
    assert 20 <= len(units) <= 30
    centres = {centre["code"]: centre["unit"] for centre in read("cost_centre")}
    for unit in read("org_unit"):
        assert centres[unit["cost_centre"]] == unit["unit"], unit["code"]
    roots = [unit for unit in units if unit["parent_org_unit_id"] is None]
    assert [root["code"] for root in roots] == ["OU-HCZ"]
    top = {unit["code"] for unit in units if unit["parent_org_unit_id"] == row_id("org_unit", code="OU-HCZ")}
    assert top == {"OU-FIN", "OU-ZAS", "OU-DOP", "OU-OBC", "OU-MKT"}


def test_one_employee_per_post_of_world_yaml(setup: Setup) -> None:
    staff = WORLD.salesforce
    roles = Counter(employee["role"] for employee in _rows(setup, "employee"))
    assert roles == {
        "rep": staff.reps,
        "area_manager": staff.area_managers,
        "head_of_key_accounts": staff.head_of_key_accounts,
        "kam": sum(staff.kams.values()),
        "product_manager": staff.product_managers,
        "dispatcher": staff.dispatchers,
        "buyer": staff.supply,
        "finance": staff.finance,
        "marketing": staff.marketing,
        "management": staff.management,
        "shop_manager": staff.shop_managers,
    }
    assert roles["rep"] == 50 and roles["area_manager"] == 6 and roles["kam"] == 16 and roles["dispatcher"] == 6
    assert len(setup.world.actors) == len(_rows(setup, "employee"))


def test_names_are_czech_from_the_lists_and_codes_are_ascii_logins(setup: Setup) -> None:
    first = {row["nominative"]: row["gender"] for row in read("names_first")}
    last = {row["male"]: "m" for row in read("names_last")} | {row["female"]: "f" for row in read("names_last")}
    employees = _rows(setup, "employee")
    for employee in employees:
        assert first[employee["first_name"]] == last[employee["last_name"]], employee  # a woman's surname is hers
        assert re.fullmatch(r"[a-z]+[0-9]{2}", employee["code"]), employee["code"]
    assert len({employee["code"] for employee in employees}) == len(employees)
    assert len({(e["first_name"], e["last_name"]) for e in employees}) == len(employees)
    women = sum(first[employee["first_name"]] == "f" for employee in employees) / len(employees)
    assert 0.3 <= women <= 0.6
    assert all(employee["hired_on"] < WORLD.history.warm_up_from for employee in employees)
    assert all(employee["left_on"] is None for employee in employees)


def test_each_reports_to_its_post_s_manager(setup: Setup) -> None:
    by_id = {employee["employee_id"]: employee for employee in _rows(setup, "employee")}
    area_managers = {region: actor.employee_id for region, actor in _by_slot(setup, "area_manager:").items()}
    region_of = setup.geography.region_of
    for area, rep in _by_slot(setup, "rep:").items():
        assert by_id[rep.employee_id]["manager_employee_id"] == area_managers[region_of[area]]
    (head,) = _by_slot(setup, "head_of_key_accounts:").values()
    for actor in setup.org.actors.values():
        if actor.role == "kam":
            assert by_id[actor.employee_id]["manager_employee_id"] == head.employee_id
    bosses = [employee for employee in by_id.values() if employee["manager_employee_id"] is None]
    assert len(bosses) == 1 and bosses[0]["role"] == "management"
    assert all(e["manager_employee_id"] in by_id for e in by_id.values() if e["manager_employee_id"] is not None)
    managed = {region["code"]: region["area_manager_employee_id"] for region in _rows(setup, "region")}
    assert managed == area_managers


def _areas_on(setup: Setup, day: date) -> Counter[str]:
    return Counter(duty.target for duty in setup.world.duties if duty.kind == "rep_area" and duty.on(day))


@pytest.mark.parametrize(
    "day",
    [
        date(2023, 1, 1),
        MOVES[0] - timedelta(days=1),
        MOVES[0],
        MOVES[1] - timedelta(days=1),
        MOVES[1],
        date(2026, 10, 31),
    ],
)
def test_every_area_has_one_rep_on_every_day(setup: Setup, day: date) -> None:
    assert _areas_on(setup, day) == dict.fromkeys(setup.geography.region_of, 1)


def test_each_rep_moves_twice_within_its_region_and_never_back(setup: Setup) -> None:
    region_of = setup.geography.region_of
    for first, rep in _by_slot(setup, "rep:").items():
        held = [duty for duty in setup.world.duties if duty.kind == "rep_area" and duty.employee_id == rep.employee_id]
        assert [duty.valid_from for duty in held] == [rep.hired_on, *MOVES]
        assert [duty.valid_to for duty in held] == [MOVES[0] - timedelta(days=1), MOVES[1] - timedelta(days=1), OPEN]
        areas = [duty.target for duty in held]
        assert areas[0] == first
        assert {region_of[area] for area in areas} == {region_of[first]}
        assert areas[0] != areas[1] != areas[2] != areas[0]
    rows = [row for row in _rows(setup, "assignment") if row["kind"] == "rep_area"]
    assert len(rows) == 50 * 3
    assert all(row["area_id"] is not None and row["region_id"] is None for row in rows)


def test_the_other_duties_one_each(setup: Setup) -> None:
    kinds = Counter(duty.kind for duty in setup.world.duties if duty.kind != "kam_customer")  # the KAMs': customers
    assert kinds == {"rep_area": 150, "area_manager_region": 6, "pm_line": 11, "dispatcher_dc": 6}
    dispatch = Counter(duty.target for duty in setup.world.duties if duty.kind == "dispatcher_dc")
    assert dispatch == {"DC-JIRNY": 3, "DC-MODRICE": 3}
    lines = {duty.target for duty in setup.world.duties if duty.kind == "pm_line"}
    assert lines == {line["code"] for line in read("product_line")}


def test_the_eight_pinned_reps_carry_their_skills_and_traits(setup: Setup) -> None:
    reps = _by_slot(setup, "rep:")
    for pin in STORIES.pins.reps:
        skills = reps[pin.slot].skills
        assert (skills.conversion, skills.acquisition, skills.retention, skills.cross_sell) == (
            pin.conversion,
            pin.acquisition,
            pin.retention,
            pin.cross_sell,
        )
        assert skills.profile == pin.profile
        assert (skills.skips_visits, skills.records_nothing) == (pin.skips_visits, pin.records_nothing)
    assert sum(rep.skills.skips_visits for rep in reps.values()) == 3  # A-7
    assert sum(rep.skills.records_nothing for rep in reps.values()) == 1  # A-8


def test_every_other_rep_draws_its_skills_in_the_range(setup: Setup) -> None:
    low, high = STORIES.story("S3").parameters["skill_range"]
    pinned = {pin.slot for pin in STORIES.pins.reps}
    drawn = [rep.skills for area, rep in _by_slot(setup, "rep:").items() if area not in pinned]
    assert len(drawn) == 42
    values = [v for s in drawn for v in (s.conversion, s.acquisition, s.retention, s.cross_sell)]
    assert all(low <= v <= high for v in values)
    assert len(set(values)) == len(values)
    assert all(s.profile is None for s in drawn)


def test_discipline_and_pos_skill_follow_s5_and_s6(setup: Setup) -> None:
    s6 = STORIES.story("S6").parameters
    low, high = s6["discipline_range"]
    reps = _by_slot(setup, "rep:")
    for area, discipline in s6["pinned_reps"].items():
        assert reps[area].skills.discount_discipline == discipline
    pms = _by_slot(setup, "pm:")
    for line, discipline in s6["pinned_product_managers"].items():
        assert pms[line].skills.discount_discipline == discipline
    disciplined = [a for a in setup.org.actors.values() if a.role in ("rep", "kam", "product_manager")]
    assert all(low <= a.skills.discount_discipline <= high for a in disciplined)
    others = [a for a in setup.org.actors.values() if a.role not in ("rep", "kam", "product_manager")]
    assert all(a.skills.discount_discipline == 1.0 for a in others)
    s5 = STORIES.story("S5").parameters
    managers = _by_slot(setup, "area_manager:")
    assert {region: m.skills.pos_skill for region, m in managers.items()} == s5["pos_skill"]
    assert {region: m.skills.pos_budget_factor for region, m in managers.items()} == {
        region: s5["budget_factor"].get(region, 1.0) for region in managers
    }


def test_the_null_world_gives_everyone_average_skills(osm_market: Market) -> None:
    null = replace(
        STORIES,
        stories=tuple(replace(story, enabled=False) for story in STORIES.stories),
        anomalies=tuple(replace(anomaly, enabled=False) for anomaly in STORIES.anomalies),
    )
    actors = run(stories=null, market=osm_market).org.actors.values()
    assert {
        (a.skills.conversion, a.skills.retention, a.skills.discount_discipline, a.skills.pos_skill) for a in actors
    } == {(1.0, 1.0, 1.0, 1.0)}
    assert not any(a.skills.skips_visits or a.skills.records_nothing or a.skills.profile for a in actors)


def test_two_runs_give_the_same_rows(setup: Setup, osm_market: Market) -> None:
    again = run(market=osm_market)
    for table in ("employee", "assignment", "org_unit", "region", "okres"):
        assert again.rows(table) == setup.rows(table), table


def test_rep_areas_in_a_ring_of_two_alternate(setup: Setup) -> None:
    geography = replace(setup.geography, region_of={"A": "R", "B": "R"})
    held = rep_areas("A", date(2015, 1, 1), MOVES, geography)
    assert [area for area, _, _ in held] == ["A", "B", "A"]
