"""G0 S0.4 T1 — the catalogue: brands, lines, products, SKUs (01 §3, ⚑NH-1, ⚑NH-4) and the competitors (⚑NH-18)."""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.checks.czech import assert_czech_all
from hartland_sim.io.reference import read

HISTORY = (date(2024, 1, 1), date(2026, 10, 31))

# 01 §3: the SKUs per line in the range as it stands at the end of the history.
LIVE_PER_LINE = {
    "CLASSIC": 14,
    "GOLD": 12,
    "ESPRESSO": 6,
    "SMESI": 12,
    "BEZKOFEIN": 4,
    "PROFIKAVA": 6,
    "POLEVKY": 24,
    "KELIMKY": 10,
    "NUDLE": 13,
    "PREMIUM": 8,
    "PROFIGASTRO": 8,
}


@pytest.fixture(scope="module")
def skus(ref: Path) -> list[dict[str, Any]]:
    return read("catalogue", ref)


@pytest.fixture(scope="module")
def products(ref: Path) -> dict[str, dict[str, Any]]:
    return {row["code"]: row for row in read("product", ref)}


@pytest.fixture(scope="module")
def lines(ref: Path) -> dict[str, dict[str, Any]]:
    return {row["code"]: row for row in read("product_line", ref)}


def _line(sku: dict[str, Any], products: dict[str, dict[str, Any]]) -> str:
    line: str = products[sku["product"]]["product_line"]
    return line


def _live(sku: dict[str, Any], day: date) -> bool:
    return bool(sku["launched_on"] <= day and (sku["delisted_on"] is None or sku["delisted_on"] > day))


def test_117_skus_live_at_the_end_of_the_history(skus: list[dict[str, Any]], products: dict[str, Any]) -> None:
    live = Counter(_line(sku, products) for sku in skus if _live(sku, HISTORY[1]))
    assert live == LIVE_PER_LINE
    assert sum(live.values()) == 117


def test_both_brands_and_their_businesses(ref: Path, lines: dict[str, dict[str, Any]]) -> None:
    brands = {row["code"]: row for row in read("brand", ref)}
    assert {(code, row["name"], row["business"]) for code, row in brands.items()} == {
        ("HARCAFE", "Harcafé", "KAVA"),
        ("HARSOUP", "Harsoup", "KULINAR"),
    }
    assert Counter(line["brand"] for line in lines.values()) == {"HARCAFE": 6, "HARSOUP": 5}


def test_premium_and_professional_lines(lines: dict[str, dict[str, Any]]) -> None:
    assert {code for code, line in lines.items() if line["is_premium"]} == {"GOLD", "ESPRESSO", "PREMIUM"}
    assert {code for code, line in lines.items() if line["is_professional"]} == {"PROFIKAVA", "PROFIGASTRO"}


def test_twelve_launches_and_six_delistings_inside_the_history(
    skus: list[dict[str, Any]], products: dict[str, Any]
) -> None:
    first, last = HISTORY
    launched = [sku for sku in skus if first <= sku["launched_on"] <= last]
    delisted = [sku for sku in skus if sku["delisted_on"] is not None]
    assert len(launched) == 12
    assert len(delisted) == 6
    assert all(first <= sku["delisted_on"] <= last for sku in delisted)
    assert not {sku["sku_code"] for sku in launched} & {sku["sku_code"] for sku in delisted}
    assert all(sku["launched_on"] < first for sku in skus if sku not in launched)


def test_the_a2_novelty_pumpkin_cream_pouch_launches_in_september_2025(skus: list[dict[str, Any]]) -> None:
    (pouch,) = [sku for sku in skus if sku["product"] == "KREM-DYNOVY" and sku["pack_format"] == "pouch"]
    assert pouch["launched_on"] == date(2025, 9, 1)


def test_the_a9_delisting_target_espresso_decaf_100_g_is_live(skus: list[dict[str, Any]]) -> None:
    """A-9 delists it at two chains (a channel listing), not from the range."""
    (jar,) = [sku for sku in skus if sku["product"] == "ESPRESSO-BEZKOFEINU" and sku["net_weight_g"] == 100]
    assert jar["delisted_on"] is None


def test_hodonin_makes_classic_gold_and_3v1_only(
    skus: list[dict[str, Any]], products: dict[str, Any], lines: dict[str, dict[str, Any]]
) -> None:
    """01 §2: Hodonín makes Classic, Gold and 3in1; Kolín the whole coffee range; Chrudim every soup."""
    hodonin = {sku["product"] for sku in skus if "HHO" in sku["suppliers"].split(";")}
    assert hodonin == {"CLASSIC", "GOLD", "3V1"}
    for sku in skus:
        coffee = lines[_line(sku, products)]["brand"] == "HARCAFE"
        assert sku["suppliers"].split(";")[0] == ("HKO" if coffee else "HCH"), sku["sku_code"]


def test_a_case_costs_less_than_it_lists(skus: list[dict[str, Any]]) -> None:
    assert all(0 < sku["cogs_case"] < sku["base_price_case"] for sku in skus)


def test_premium_coffee_lists_dearer_per_gram_than_classic(
    skus: list[dict[str, Any]], products: dict[str, Any]
) -> None:
    def per_gram(line: str) -> float:
        jars = [
            s for s in skus if _line(s, products) == line and s["pack_format"] == "jar" and s["net_weight_g"] == 100
        ]
        cheapest: float = min(s["base_price_case"] / s["units_per_case"] / s["net_weight_g"] for s in jars)
        return cheapest

    assert per_gram("GOLD") > per_gram("CLASSIC") * 1.4
    assert per_gram("ESPRESSO") > per_gram("GOLD")


def test_professional_packs_are_never_prompted(skus: list[dict[str, Any]], products: dict[str, Any]) -> None:
    """R-7: importance 0 is never offered to a retail store; every pack of a Professional line is a Professional
    pack."""
    for sku in skus:
        if _line(sku, products) in {"PROFIKAVA", "PROFIGASTRO"}:
            assert sku["is_professional"]
        if sku["is_professional"]:
            assert sku["importance"] == 0
    assert Counter(sku["importance"] for sku in skus)[10] >= 10


def test_sku_codes_are_ean13_in_the_restricted_range(skus: list[dict[str, Any]]) -> None:
    """GS1's 29x prefix is for restricted circulation: no real product carries these codes."""
    for sku in skus:
        code = sku["sku_code"]
        total = sum(int(digit) * (3 if i % 2 else 1) for i, digit in enumerate(code[:12]))
        assert int(code[12]) == (10 - total % 10) % 10, code


def test_names_and_nicknames_read_as_czech(
    skus: list[dict[str, Any]], products: dict[str, dict[str, Any]], lines: dict[str, dict[str, Any]]
) -> None:
    assert_czech_all([sku["name_cs"] for sku in skus], what="catalogue.name_cs")
    assert_czech_all([product["name_cs"] for product in products.values()], what="product.name_cs")
    assert_czech_all([line["name_cs"] for line in lines.values()], what="product_line.name_cs")
    nicknames = [nick for product in products.values() for nick in product["nicknames_cs"].split(";")]
    assert {"zlatý kafe", "klasik", "trojka"} <= set(nicknames)
    assert_czech_all(nicknames, what="product.nicknames_cs")


def test_the_four_competitors_and_their_equivalents(ref: Path, skus: list[dict[str, Any]]) -> None:
    competitors = {row["code"]: row for row in read("competitor", ref)}
    assert {row["name"] for row in competitors.values()} == {"Kavalír", "Nordcafé", "Polévkárna", "Zlatá vývarka"}
    rows = read("competitor_product", ref)
    assert 35 <= len(rows) <= 45
    assert set(Counter(row["competitor"] for row in rows).values()) == {10}
    live_products = {sku["product"] for sku in skus if _live(sku, HISTORY[1])}
    assert {row["equivalent_product"] for row in rows} <= live_products
    by_business = {code: row["business"] for code, row in competitors.items()}
    coffee = {sku["product"] for sku in skus if sku["suppliers"] != "HCH"}
    for row in rows:
        assert (row["equivalent_product"] in coffee) is (by_business[row["competitor"]] == "KAVA"), row["code"]
