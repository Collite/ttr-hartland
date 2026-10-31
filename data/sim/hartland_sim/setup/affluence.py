"""The affluence index of every okres (⚑NH-17): its average gross wage over the national one, clamped to 0.8–1.3.

The inputs are two reference files (`data/reference/SOURCES.md`): `wage_by_kraj.csv` — ČSÚ's average gross monthly
wage of each kraj and of Česko (the `CZ` row), which ČSÚ publishes no finer — and `wage_by_okres.csv`, the world's
own hand-kept spread of each okres around its kraj (`wage_vs_kraj`). A Prague district (`parent_okres` = CZ0100)
takes Prague's kraj figure. The index drives the premium share of base demand (S4) and the TV-promotion uplift (S2).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from hartland_sim.io.reference import REFERENCE_DIR, read

FLOOR = 0.8
CEILING = 1.3
NATIONAL = "CZ"


def affluence_index(
    kraj_of: Mapping[str, str],
    kraj_wage: Mapping[str, float],
    wage_vs_kraj: Mapping[str, float],
) -> dict[str, float]:
    """okres → clamp(kraj wage × the okres's spread / the national wage, 0.8, 1.3), to three decimals.
    `kraj_wage` carries the national figure under `"CZ"`."""
    national = kraj_wage[NATIONAL]
    return {
        okres: round(min(max(kraj_wage[kraj] * wage_vs_kraj[okres] / national, FLOOR), CEILING), 3)
        for okres, kraj in kraj_of.items()
    }


def load(root: Path = REFERENCE_DIR) -> dict[str, float]:
    """The index of every okres in `okres.csv`, from the committed reference files."""
    kraj_of = {row["code"]: row["kraj"] for row in read("okres", root)}
    kraj_wage = {row["code"]: float(row["wage_czk"]) for row in read("wage_by_kraj", root)}
    spread = {row["okres"]: float(row["wage_vs_kraj"]) for row in read("wage_by_okres", root)}
    return affluence_index(kraj_of, kraj_wage, spread)
