"""G0 S0.2 T6 — names, legal entities, sites, cost centres, GL accounts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from hartland_sim.io.reference import load_schema, read

# 01 §2 (⚑NH-5): the twelve own shops.
SHOP_TOWNS = Counter(
    {"Praha": 3, "Brno": 2}
    | dict.fromkeys(["Ostrava", "Plzeň", "Olomouc", "Liberec", "Hradec Králové", "České Budějovice", "Zlín"], 1)
)


@pytest.fixture(scope="module")
def ref(repo_root: Path) -> Path:
    return repo_root / "data" / "reference"


def table(ref: Path, name: str) -> list[dict[str, Any]]:
    return read(name, ref)


# ── T6 ──────────────────────────────────────────────────────────────────────


def test_the_group_four_entities_one_center(ref: Path) -> None:
    entities = table(ref, "legal_entity")
    assert [(row["name"], row["role"]) for row in entities] == [
        ("Hartland CZ s.r.o.", "center"),
        ("Harcafé Kolín a.s.", "factory"),
        ("Harcafé Hodonín a.s.", "factory"),
        ("Harsoup Chrudim a.s.", "factory"),
    ]


@pytest.mark.parametrize("ico", ["93102453", "93102461", "93102470", "93102488"])
def test_a_fictional_ico_is_well_formed(ref: Path, ico: str) -> None:
    """The IČO checksum (mod 11) holds, so a validator accepts them; ARES knew none of them on 2026-10-31."""
    assert ico in {row["ico"] for row in table(ref, "legal_entity")}
    remainder = sum(int(digit) * weight for digit, weight in zip(ico[:7], range(8, 1, -1), strict=True)) % 11
    assert int(ico[7]) == {0: 1, 1: 0}.get(remainder, 11 - remainder)


def test_eighteen_sites_and_the_twelve_shops(ref: Path) -> None:
    sites = table(ref, "site")
    assert Counter(row["kind"] for row in sites) == {"hq": 1, "factory": 3, "dc": 2, "shop": 12}
    assert Counter(row["town"] for row in sites if row["kind"] == "shop") == SHOP_TOWNS
    factories = {row["town"]: row["legal_entity"] for row in sites if row["kind"] == "factory"}
    assert factories == {"Kolín": "HKO", "Hodonín": "HHO", "Chrudim": "HCH"}
    assert {row["town"] for row in sites if row["kind"] == "dc"} == {"Jirny", "Modřice"}


def test_a_cost_centre_for_every_unit_region_and_shop(ref: Path) -> None:
    centres = table(ref, "cost_centre")
    assert {row["unit"] for row in centres} == {"supply", "delivery", "sales", "marketing", "finance_admin"}
    regions = {row["name"] for row in table(ref, "region")}
    assert {row["name"].removeprefix("Terénní prodej – ") for row in centres if "Terénní" in row["name"]} == regions
    shops = {row["code"] for row in table(ref, "site") if row["kind"] == "shop"}
    assert shops <= {row["site"] for row in centres}


def test_the_pl_line_to_account_map_is_total(ref: Path) -> None:
    """Every P/L line of contracts §4 has an account; every revenue or expense account has a line; no balance
    account has one."""
    lines = {line for line in load_schema(ref / "schemas" / "gl_account.schema.json")["properties"]["pl_line"]["enum"]}
    accounts = table(ref, "gl_account")
    assert {row["pl_line"] for row in accounts} == lines
    for row in accounts:
        assert (row["pl_line"] is not None) == (row["kind"] in {"revenue", "revenue_deduction", "expense"}), row


def test_vat_accounts_for_both_rates_both_ways(ref: Path) -> None:
    vat = {(row["kind"], row["vat_rate"]) for row in table(ref, "gl_account") if row["vat_rate"] is not None}
    assert vat == {("liability", 12), ("liability", 21), ("asset", 12), ("asset", 21)}


def test_enough_names_of_both_genders(ref: Path) -> None:
    first = table(ref, "names_first")
    assert Counter(row["gender"] for row in first) == {"m": 59, "f": 59}
    assert len(table(ref, "names_last")) >= 60


def test_the_female_surname_is_the_male_one_declined(ref: Path) -> None:
    for row in table(ref, "names_last"):
        male, female = row["male"], row["female"]
        assert female.endswith(("ová", "á")), row
        assert female[:2] == male[:2], row


def test_the_cases_differ_where_czech_declines(ref: Path) -> None:
    indeclinable = {"Marie", "Lucie", "Natálie", "Libuše", "Dagmar"}
    for row in table(ref, "names_first"):
        if row["nominative"] not in indeclinable:
            assert row["genitive"] != row["nominative"], row
