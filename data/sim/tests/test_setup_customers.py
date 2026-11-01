"""Setup steps 2 and 5 (G1 S1.1 T6): the market's stores and chains, the customers of every channel with their
delivery points, the KAMs, the pins and the first lifecycle phases — on the full OSM extract."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import pytest

from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import read
from hartland_sim.setup.classifier import osm_order
from hartland_sim.setup.customers import family_plural, small_retail_pool
from hartland_sim.setup.market import Market
from hartland_sim.setup.run import Setup, run
from hartland_sim.world import OPEN

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
START = WORLD.history.warm_up_from


@pytest.fixture(scope="module")
def setup(opening: Setup) -> Setup:
    return opening


def _rows(setup: Setup, table: str) -> list[dict[str, Any]]:
    return setup.tables[table]


def _customers(setup: Setup, channel: str) -> list[dict[str, Any]]:
    return [customer for customer in _rows(setup, "customer") if customer["channel"] == channel]


def test_the_market_is_written_whole(setup: Setup, osm_market: Market) -> None:
    stores = _rows(setup, "store")
    assert len(stores) == len(osm_market.stores)
    assert len({store["store_id"] for store in stores}) == len(stores)
    assert {chain["code"] for chain in _rows(setup, "chain")} == {chain["code"] for chain in osm_market.chains}


def test_forty_percent_of_the_independents_and_station_shops_start_as_customers(
    setup: Setup, osm_market: Market
) -> None:
    retail = _customers(setup, "small_retail")
    assert len(retail) == len(small_retail_pool(osm_market.stores))
    active = [customer for customer in retail if customer["status"] == "active"]
    assert abs(len(active) / len(retail) - WORLD.channels.small_retail.active_share_at_start) < 0.02  # type: ignore[operator]
    assert all(date(2019, 1, 1) <= customer["since"] <= date(2023, 12, 31) for customer in active)
    years = Counter(customer["since"].year for customer in active)
    assert set(years) == {2019, 2020, 2021, 2022, 2023}
    assert all(c["since"] is None for c in retail if c["status"] == "potential")
    stations = [customer for customer in retail if customer["sub_channel"] == "special_retail"]
    by_store = {store["store_id"]: store for store in _rows(setup, "store")}
    assert stations and all(by_store[c["store_id"]]["store_class"] == "petrol" for c in stations)
    assert all(by_store[c["store_id"]]["store_class"] != "petrol" for c in retail if c["sub_channel"] is None)


def test_key_account_and_norma_stores_are_no_small_retail_customers(setup: Setup) -> None:
    by_store = {store["store_id"]: store for store in _rows(setup, "store")}
    chains = {chain["chain_id"]: chain for chain in _rows(setup, "chain")}
    for customer in _customers(setup, "small_retail"):
        chain = chains.get(by_store[customer["store_id"]]["chain_id"])
        assert chain is None or (not chain["is_key_account"] and chain["kind"] in ("convenience_franchise", "petrol"))


def _points(setup: Setup) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in _rows(setup, "delivery_point"):
        out[point["customer_id"]].append(point)
    return out


def test_key_accounts_deliver_to_their_stores_or_their_dcs(setup: Setup) -> None:
    points = _points(setup)
    stores = _rows(setup, "store")
    big = {c["code"]: c for c in _customers(setup, "big_retail")}
    gas = {c["code"]: c for c in _customers(setup, "gas_ka")}
    assert set(big) == {f"KA-{chain}" for chain in WORLD.channels.big_retail.accounts}
    assert set(gas) == {"KA-OMV", "KA-SHELL"}
    dc_level = WORLD.channels.big_retail.dc_level_orders
    for code, customer in (big | gas).items():
        chain = code.removeprefix("KA-")
        mine = points[customer["customer_id"]]
        if chain in dc_level:  # ⚑R196-F: its DCs, one per Hartland DC, in that DC's okres, no store
            sites = {site["code"]: site for site in read("site")}
            assert sorted((p["okres_id"], p["dc_site_id"]) for p in mine) == sorted(
                (row_id("okres", code=sites[dc]["okres"]), row_id("site", code=dc)) for dc in WORLD.sites.dcs
            )
            assert all(p["store_id"] is None and "distribuční centrum" in p["name"] for p in mine)
        else:
            chain_id = row_id("chain", code=chain)
            own = {s["store_id"] for s in stores if s["chain_id"] == chain_id and s["store_class"] != "horeca"}
            assert {p["store_id"] for p in mine} == own and own
        assert customer["okres_id"] is None and customer["status"] == "active"


def test_the_other_channels_count_as_world_yaml(setup: Setup) -> None:
    horeca = Counter((c["status"], c["store_id"] is None) for c in _customers(setup, "horeca"))
    channels = WORLD.channels
    assert horeca == {
        ("active", False): channels.horeca.direct_customers,
        ("potential", False): channels.horeca.potential,
        ("active", True): channels.horeca.wholesalers,
    }
    assert len(_customers(setup, "vending")) == channels.vending.operators
    assert len(_customers(setup, "own_shops")) == WORLD.sites.own_shops
    assert len(_customers(setup, "web")) == channels.web.consumers
    # 03 §3: ≈ 36 700 customers + ≈ 6 500 potential; the small retail of the real extract is smaller (Found line)
    assert abs(len(_rows(setup, "customer")) - 43_200) / 43_200 < 0.10


def test_own_shops_deliver_to_their_site_without_a_store(setup: Setup) -> None:
    points = _points(setup)
    sites = {site["name"]: site for site in read("site") if site["kind"] == "shop"}
    for customer in _customers(setup, "own_shops"):
        (point,) = points[customer["customer_id"]]
        site = sites[customer["name"]]
        assert point["store_id"] is None
        assert point["okres_id"] == customer["okres_id"] == row_id("okres", code=site["okres"])
        assert customer["payment_terms_days"] == 0 and customer["price_list_channel"] == "own_shops"


def test_web_consumers_spread_by_population_and_affluence(setup: Setup, okres: dict[str, dict[str, Any]]) -> None:
    web = _customers(setup, "web")
    praha = [code for code in okres if code.startswith("CZ0100-")]
    in_praha = sum(customer["okres_id"] in {row_id("okres", code=code) for code in praha} for customer in web)
    country = sum(row["population"] for code, row in okres.items() if code != "CZ0100")
    population = okres["CZ0100"]["population"] / country
    share = in_praha / len(web)
    assert population * 0.95 < share < population * 1.5  # Prague is richer than the country
    assert all(date(2019, 1, 1) <= c["since"] <= date(2023, 12, 31) for c in web)
    assert all(c["credit_limit"] is None and c["payment_terms_days"] == 0 for c in web)


def test_every_customer_has_a_point_in_a_leaf_okres_served_by_its_region_s_dc(setup: Setup, leaves: list[str]) -> None:
    points = _points(setup)
    assert set(points) == {customer["customer_id"] for customer in _rows(setup, "customer")}
    ids = [point["delivery_point_id"] for point in _rows(setup, "delivery_point")]
    assert len(set(ids)) == len(ids)
    leaf_ids = {row_id("okres", code=code): code for code in leaves}
    geography = setup.geography
    for point in _rows(setup, "delivery_point"):
        code = leaf_ids[point["okres_id"]]  # a leaf: never Praha itself
        assert point["dc_site_id"] == row_id("site", code=geography.dc_for(code))


def test_windows_follow_the_kind_of_place(setup: Setup) -> None:
    windows = WORLD.delivery.windows
    kinds = Counter((p["time_window_from"], p["unloading_minutes"]) for p in _rows(setup, "delivery_point"))
    assert kinds[windows["web"].opens, windows["web"].unloading_minutes] == WORLD.channels.web.consumers
    assert kinds[windows["chain_dc"].opens, windows["chain_dc"].unloading_minutes] == 6


def test_customer_terms_and_limits_by_channel(setup: Setup) -> None:
    days = {str(channel): n for channel, n in WORLD.finance.payment_terms_days.items()}
    limits = {str(channel): n for channel, n in WORLD.policies.credit_limit.items()}
    for customer in _rows(setup, "customer"):
        terms = "special_retail" if customer["sub_channel"] == "special_retail" else customer["channel"]
        assert customer["payment_terms_days"] == days[terms]
        assert customer["credit_limit"] == limits.get(terms)
        assert customer["price_list_channel"] == customer["channel"]
    codes = [customer["code"] for customer in _rows(setup, "customer")]
    assert len(set(codes)) == len(codes)
    assert all(
        re.fullmatch(r"(KA-[A-Z]+|MO-\d{6}|GA-\d{6}|VO-\d{2}|AU-\d{2}|VP-[A-Z0-9-]+|ES-\d{6})", c) for c in codes
    )


def test_synthetic_names_read_as_czech(setup: Setup) -> None:
    synthetic = [
        c["name"] for c in _rows(setup, "customer") if c["channel"] in ("vending", "web") or c["store_id"] is None
    ]
    places = [p["name"] for p in _rows(setup, "delivery_point") if p["store_id"] is None]
    assert_czech_all(synthetic, what="customer.name", allow=["Harcafé", "Tesco", "Albert", "Lidl", "Penny", "Shell"])
    assert_czech_all(places, what="delivery_point.name", allow=["Harcafé", "Lidl", "Kaufland", "Penny"])


def test_the_pins_resolve_by_rule_to_active_customers(setup: Setup) -> None:
    by_code = {customer.code: customer for customer in setup.world.customers.values()}
    stores = {store["osm_id"]: store for store in setup.customers.stores}
    assert Counter(setup.customers.pins.values()) == Counter(pin.trait for pin in STORIES.pins.customers)
    for pin in STORIES.pins.customers:
        (code,) = [c for c, trait in setup.customers.pins.items() if trait == pin.trait and by_code[c].area == pin.area]
        customer = by_code[code]
        assert customer.status == "active" and customer.channel == pin.channel and pin.trait in customer.traits
        assert pin.chain is None or customer.chain == pin.chain
        assert customer.store is not None
        assert pin.store_class is None or stores[customer.store]["store_class"] == pin.store_class
        rivals = [
            c
            for c in by_code.values()
            if c.status == "active"
            and c.channel == pin.channel
            and c.area == pin.area
            and c.store is not None
            and (pin.chain is None or c.chain == pin.chain)
            and (pin.store_class is None or stores[c.store]["store_class"] == pin.store_class)
        ]
        assert min(rivals, key=lambda c: osm_order(c.store or "")).code == code  # rank 1 in osm_order


def test_a1_and_a14_customers_get_an_invented_name_a13_keeps_its_own(setup: Setup, osm_market: Market) -> None:
    original = {store["osm_id"]: store["name"] for store in osm_market.stores}
    renamed = {store["osm_id"]: store["name"] for store in _rows(setup, "store")}
    customers = {c["code"]: c for c in _rows(setup, "customer")}
    points = {p["customer_id"]: p for p in _rows(setup, "delivery_point")}
    by_code = {customer.code: customer for customer in setup.world.customers.values()}
    for code, trait in setup.customers.pins.items():
        osm_id = by_code[code].store
        assert osm_id is not None
        if trait == "rejects_noodle_prompts":
            assert renamed[osm_id] == original[osm_id] == customers[code]["name"]
            continue
        name = customers[code]["name"]
        assert re.fullmatch(r"(Potraviny|Smíšené zboží|Supermarket|Restaurace|Kavárna|Hostinec|Jídelna) U \w+", name)
        assert renamed[osm_id] == name == points[customers[code]["customer_id"]]["name"] != original[osm_id]
    assert sum(renamed[k] != original[k] for k in original) == 4


def test_family_plural_declines_every_surname_of_the_list() -> None:
    examples = {"Novák": "Nováků", "Černý": "Černých", "Svoboda": "Svobodů", "Marek": "Marků", "Vaněk": "Vaňků"}
    examples |= {"Němec": "Němců", "Kadlec": "Kadlců", "Procházka": "Procházků", "Kříž": "Křížů", "Hájek": "Hájků"}
    assert {name: family_plural(name) for name in examples} == examples
    plurals = [family_plural(row["male"]) for row in read("names_last")]
    assert all(p.endswith(("ů", "ých")) for p in plurals)
    assert len(set(plurals)) == len(plurals)


def test_the_null_world_pins_and_renames_no_one(osm_market: Market) -> None:
    null = replace(STORIES, anomalies=tuple(replace(a, enabled=False) for a in STORIES.anomalies))
    quiet = run(stories=null, market=osm_market)
    assert quiet.customers.pins == {}
    assert {s["name"] for s in quiet.tables["store"]} == {s["name"] for s in osm_market.stores}


def test_kams_hold_their_accounts(setup: Setup) -> None:
    held: dict[int, list[str]] = defaultdict(list)
    for duty in setup.world.duties:
        if duty.kind == "kam_customer":
            held[duty.employee_id].append(duty.target)
    kams = {slot: actor.employee_id for slot, actor in setup.org.actors.items() if slot.startswith("kam:")}
    big = [held[kams[f"kam:big_retail:{n}"]] for n in range(1, 9)]
    assert sorted(len(accounts) for accounts in big) == [1] * 7 + [2]
    assert held[kams["kam:gas_ka:1"]] == ["KA-OMV", "KA-SHELL"]
    assert sorted(len(held[kams[f"kam:vending:{n}"]]) for n in range(1, 4)) == [13, 13, 14]
    horeca = [c["code"] for c in _customers(setup, "horeca")]
    assert sorted(code for n in range(1, 5) for code in held[kams[f"kam:horeca:{n}"]]) == sorted(horeca)
    rows = [row for row in _rows(setup, "assignment") if row["kind"] == "kam_customer"]
    assert len(rows) == sum(len(accounts) for accounts in held.values())
    assert all(row["customer_id"] is not None and row["area_id"] is None for row in rows)


def test_small_retail_customers_open_with_a_phase(setup: Setup) -> None:
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for episode in _rows(setup, "lifecycle_phase"):
        episodes[episode["customer_id"]].append(episode)
    retail = _customers(setup, "small_retail")
    assert set(episodes) == {customer["customer_id"] for customer in retail}
    for customer in retail:
        mine = sorted(episodes[customer["customer_id"]], key=lambda e: e["valid_from"])
        assert mine[-1]["valid_to"] == OPEN
        if customer["status"] == "potential":
            assert [(e["phase"], e["valid_from"]) for e in mine] == [("potential", START)]
        elif customer["since"] > START:
            assert [e["phase"] for e in mine] == ["potential", "occasional"]
            assert mine[0]["valid_to"] == customer["since"] - timedelta(days=1)
        else:
            assert [(e["phase"], e["valid_from"]) for e in mine] == [("occasional", customer["since"])]


def test_the_opening_budgets_cover_every_rep_and_kam(setup: Setup) -> None:
    budgets = {row["employee_id"]: row for row in _rows(setup, "discount_budget")}
    reps = {actor.employee_id for actor in setup.org.actors.values() if actor.role in ("rep", "kam")}
    assert set(budgets) == reps and len(reps) == 66
    assert {row["quarter"] for row in budgets.values()} == {date(2023, 1, 1)}
    active = Counter(
        c.area for c in setup.world.customers.values() if c.channel == "small_retail" and c.active_on(START)
    )
    gross = WORLD.pricing.budgets.opening_quarter_gross["small_retail"]
    for slot, actor in setup.org.actors.items():
        if slot.startswith("rep:"):
            area = slot.removeprefix("rep:")
            assert budgets[actor.employee_id]["budget_amount"] == round(0.02 * gross * active[area], 2)
    assert {(e, q) for e, q in setup.world.budgets} == {(e, date(2023, 1, 1)) for e in budgets}


def test_the_world_holds_every_customer_with_its_traits(setup: Setup) -> None:
    assert len(setup.world.customers) == len(_rows(setup, "customer"))
    traits = {c.code: c.traits for c in setup.world.customers.values() if c.traits}
    assert {code: trait for code, (trait,) in traits.items()} == setup.customers.pins
