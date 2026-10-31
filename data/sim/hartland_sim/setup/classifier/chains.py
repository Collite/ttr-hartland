"""Chain matching (G0 S0.3 T3): an OSM object's `brand` / `name` / `operator` → its chain, or none (an independent,
which keeps its OSM name, ⚑NH-2).

The chains are `03` §3's: the nine KA retail chains of `01` §4 (COOP's co-operative groups are one chain), the
station chains (OMV and Shell are key accounts, NH-D5; Orlen/Benzina, MOL, EuroOil and Tank ONO are not), and the
convenience franchises and smaller chains the store set holds (Žabka, Hruška, Můj obchod, …). A station chain also
owns its shop and café brands (Shell Select, OMV's Viva, Orlen's Stop Cafe, MOL's Fresh Corner), which OSM often maps
as a separate object beside the pumps.

Matching folds case and diacritics and compares whole words, so `Albertov` is not Albert. The brand decides first,
then the name, then the operator. A fuel station is matched against the station chains only: a hypermarket's pumps
(Tesco, Globus, Makro) are an independent station.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Chain:
    code: str
    name: str
    kind: str  # modern_trade · cooperative · petrol · convenience_franchise
    is_key_account: bool
    patterns: tuple[str, ...]  # whole-word regexes over the folded text

    @property
    def is_station(self) -> bool:
        return self.kind == "petrol"


CHAINS: tuple[Chain, ...] = (
    Chain("TESCO", "Tesco", "modern_trade", True, (r"tesco",)),
    Chain("ALBERT", "Albert", "modern_trade", True, (r"albert", r"ahold")),
    Chain("LIDL", "Lidl", "modern_trade", True, (r"lidl",)),
    Chain("KAUFLAND", "Kaufland", "modern_trade", True, (r"kaufland",)),
    Chain("PENNY", "Penny", "modern_trade", True, (r"penny( market)?", r"pennymarket")),
    Chain("BILLA", "Billa", "modern_trade", True, (r"billa",)),
    Chain("GLOBUS", "Globus", "modern_trade", True, (r"globus",)),
    Chain("COOP", "COOP", "cooperative", True, (r"coop", r"jednota", r"konzum", r"tempo", r"terno")),
    Chain("MAKRO", "Makro", "modern_trade", True, (r"^makro",)),  # not "Mikro Makro", an independent
    Chain("OMV", "OMV", "petrol", True, (r"omv", r"viva")),
    Chain("SHELL", "Shell", "petrol", True, (r"shell( select)?",)),
    Chain("ORLEN", "Orlen", "petrol", False, (r"orlen", r"benzina", r"stop cafe")),
    Chain("MOL", "MOL", "petrol", False, (r"mol", r"fresh corner")),
    Chain("EUROOIL", "EuroOil", "petrol", False, (r"euro ?oil", r"cepro")),
    Chain("TANKONO", "Tank ONO", "petrol", False, (r"tank ono",)),
    Chain("ZABKA", "Žabka", "convenience_franchise", False, (r"zabka",)),
    Chain("HRUSKA", "Hruška", "convenience_franchise", False, (r"hruska",)),
    Chain("MUJOBCHOD", "Můj obchod", "convenience_franchise", False, (r"muj obchod",)),
    Chain("ENAPO", "Enapo", "convenience_franchise", False, (r"enapo",)),
    Chain("CBA", "CBA", "convenience_franchise", False, (r"cba",)),
    Chain("BRNENKA", "Brněnka", "convenience_franchise", False, (r"brnenka",)),
    Chain("NORMA", "Norma", "modern_trade", False, (r"norma",)),
)

BY_CODE = {chain.code: chain for chain in CHAINS}

_COMPILED = [(chain, re.compile(r"\b(?:" + "|".join(chain.patterns) + r")\b")) for chain in CHAINS]


def fold(text: str) -> str:
    """Lower case, no diacritics, `&` read as `and`, every other non-alphanumeric run as one space."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).replace("&", " and ")
    return " ".join(re.sub(r"[^0-9a-z]+", " ", plain).split())


def match(*, brand: str = "", name: str = "", operator: str = "", fuel: bool = False) -> Chain | None:
    """The chain of an OSM object, or None for an independent. `fuel`: the object is a station (`amenity=fuel`)."""
    for text in (brand, name, operator):
        folded = fold(text)
        if not folded:
            continue
        for chain, pattern in _COMPILED:
            if fuel and not chain.is_station:
                continue
            if pattern.search(folded):
                return chain
    return None
