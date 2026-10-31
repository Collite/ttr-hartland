"""The simulation's state and its daily checkpoint (contracts §2, architecture §2).

`World` holds everything the day loop carries from one day to the next — customers with their condition, phase and
basket, stock per SKU × DC, open orders, shipments and invoices, tourplans, loans, running promotions, budgets, the
actors' hidden skills. `Checkpoint` is the pickled `World` after a day, hashed, stored as
`_sim_checkpoint(day, sha256, blob)`; `prolong` resumes from the last one, and the 2023-12-31 checkpoint (the end of
the warm-up year) lets a null-world or re-parameterised build start from the same opening. G1 S1.1 T3.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class World:
    """The state after the last simulated day (G1 S1.1 T3)."""


@dataclass(frozen=True)
class Checkpoint:
    """One row of `_sim_checkpoint`: the `World` after `day`, pickled, with the sha256 of the pickle."""

    day: date
    sha256: str
    blob: bytes

    @classmethod
    def of(cls, world: World, day: date) -> Checkpoint:
        raise NotImplementedError("G1 S1.1 T3")

    def restore(self) -> World:
        raise NotImplementedError("G1 S1.1 T3")
