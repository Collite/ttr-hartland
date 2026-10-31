"""G0 S0.3 T3 — chain matching: a golden table of the name variants in the OSM store set."""

from __future__ import annotations

import pytest

from hartland_sim.setup.classifier.chains import BY_CODE, CHAINS, fold, match

# (brand, name, operator, fuel) → chain code; None = an independent, which keeps its OSM name.
GOLDEN: list[tuple[str, str, str, bool, str | None]] = [
    ("Albert", "Albert Supermarket", "", False, "ALBERT"),
    ("", "Albert Hypermarket", "", False, "ALBERT"),
    ("", "Potraviny", "AHOLD Czech Republic, a.s.", False, "ALBERT"),
    ("", "BILLA", "", False, "BILLA"),
    ("Billa", "Billa Stop & Shop", "", False, "BILLA"),
    ("Coop", "Coop Jednota", "", False, "COOP"),
    ("COOP TIP", "", "COOP Hořovice, družstvo", False, "COOP"),
    ("Coop Tuty", "Tuty", "", False, "COOP"),
    ("TUTY", "Potraviny Tuty", "TEMPO, obchodní družstvo", False, "COOP"),
    ("Coop Konzum", "Konzum", "", False, "COOP"),
    ("", "Jednota, spotřební družstvo v Klatovech", "", False, "COOP"),
    ("Tempo", "Tempo", "", False, "COOP"),
    ("", "Terno", "", False, "COOP"),
    ("Žabka", "Žabka", "", False, "ZABKA"),
    ("", "Zabka", "", False, "ZABKA"),
    ("", "ŽABKA", "", False, "ZABKA"),
    ("Hruška", "Hruška", "", False, "HRUSKA"),
    ("", "Potraviny Hruška", "", False, "HRUSKA"),
    ("Můj obchod", "Můj obchod", "", False, "MUJOBCHOD"),
    ("", "Muj Obchod", "", False, "MUJOBCHOD"),
    ("Penny", "Penny Market", "", False, "PENNY"),
    ("PENNY", "PENNY", "", False, "PENNY"),
    ("Lidl", "Lidl", "", False, "LIDL"),
    ("", "Prodejna", "Lidl Česká republika v.o.s.", False, "LIDL"),
    ("Kaufland", "Kaufland", "", False, "KAUFLAND"),
    ("Tesco Expres", "Tesco Expres", "", False, "TESCO"),
    ("Tesco Extra", "Tesco Extra", "", False, "TESCO"),
    ("", "Tesco Hypermarket", "", False, "TESCO"),
    ("Globus", "Globus", "", False, "GLOBUS"),
    ("Makro", "Makro Cash & Carry", "", False, "MAKRO"),
    ("", "Mikro Makro", "", False, None),
    ("Enapo", "Enapo", "", False, "ENAPO"),
    ("CBA", "CBA Potraviny", "", False, "CBA"),
    ("Brněnka", "Brněnka", "", False, "BRNENKA"),
    ("Norma", "Norma", "", False, "NORMA"),
    ("OMV", "OMV", "", True, "OMV"),
    ("Viva", "VIVA", "", False, "OMV"),
    ("Shell", "Shell", "", True, "SHELL"),
    ("Shell Select", "Shell Select", "", False, "SHELL"),
    ("Orlen", "ORLEN", "", True, "ORLEN"),
    ("Benzina", "Benzina", "", True, "ORLEN"),
    ("", "Benzina Orlen", "", True, "ORLEN"),
    ("Orlen express", "", "", True, "ORLEN"),
    ("Stop Cafe", "Stop Cafe", "", False, "ORLEN"),
    ("MOL", "MOL", "", True, "MOL"),
    ("Fresh Corner", "MOL Fresh Corner", "", False, "MOL"),
    ("Slovnaft", "Slovnaft", "MOL Česká republika, s.r.o.", True, "MOL"),
    ("EuroOil", "EuroOil", "", True, "EUROOIL"),
    ("", "Euro Oil", "", True, "EUROOIL"),
    ("", "Čerpací stanice", "ČEPRO, a.s.", True, "EUROOIL"),
    ("Tank Ono", "Tank ONO", "", True, "TANKONO"),
    ("Tesco", "Tesco", "", True, None),  # a hypermarket's pumps
    ("Globus", "Čerpací stanice Globus", "", True, None),
    ("Makro", "Makro - čerpací stanice", "", True, None),
    ("Robin Oil", "Robin Oil", "ČEPRO, a.s.", True, None),  # a brand of its own: its operator does not decide
    ("Avia", "AVIA", "", True, None),
    ("", "Potraviny U Lípy", "", False, None),
    ("", "Večerka", "", False, None),
    ("", "Smíšené zboží", "", False, None),
    ("", "Albertov – potraviny", "", False, None),
    ("", "Pekárna Molík", "", False, None),
    ("", "Ovoce a zelenina", "", False, None),
]


def test_sixty_two_variants() -> None:
    assert len(GOLDEN) == 62


@pytest.mark.parametrize(("brand", "name", "operator", "fuel", "code"), GOLDEN)
def test_a_name_variant_finds_its_chain(brand: str, name: str, operator: str, fuel: bool, code: str | None) -> None:
    chain = match(brand=brand, name=name, operator=operator, fuel=fuel)
    assert (chain.code if chain else None) == code


def test_eleven_key_accounts_nine_retail_and_two_station_chains() -> None:
    key = [chain for chain in CHAINS if chain.is_key_account]
    assert sorted(chain.code for chain in key if not chain.is_station) == sorted(
        ["TESCO", "ALBERT", "LIDL", "KAUFLAND", "PENNY", "BILLA", "GLOBUS", "COOP", "MAKRO"]
    )
    assert sorted(chain.code for chain in key if chain.is_station) == ["OMV", "SHELL"]
    assert set(BY_CODE) == {chain.code for chain in CHAINS}


def test_fold() -> None:
    assert fold("  ŽABKA – Stop & Shop! ") == "zabka stop and shop"
