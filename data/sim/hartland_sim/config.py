"""The three configs the generator reads — each a typed view over a committed, schema-checked file.

- `WorldConfig` ← `data/world.yaml`: every number of `01-company.md` (sites, channels, salesforce, pricing,
  promotions, policies, finance, calendar, dials, `master_seed`); validated by jsonschema. G0 S0.4 T2.
- `StoriesConfig` ← `data/stories/s1…s6.yaml` + `anomalies.yaml`: the answer key (contracts §3). G0 S0.4 T3.
- `InventoryConfig` ← `data/inventory.yaml`: one entry per table — the one source of the DDL, the load order and
  the db layer (contracts §1). G0 S0.4 T4.

A missing or mistyped key fails at load with its path, never later inside the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorldConfig:
    """`data/world.yaml`, validated (G0 S0.4 T2)."""

    @classmethod
    def load(cls, path: Path) -> WorldConfig:
        raise NotImplementedError("G0 S0.4 T2")


@dataclass(frozen=True)
class StoriesConfig:
    """`data/stories/`, validated: the six stories, the anomalies, the pins (contracts §3; G0 S0.4 T3)."""

    @classmethod
    def load(cls, directory: Path) -> StoriesConfig:
        raise NotImplementedError("G0 S0.4 T3")


@dataclass(frozen=True)
class InventoryConfig:
    """`data/inventory.yaml`, validated: tables, columns, keys, `load_order` (contracts §1; G0 S0.4 T4)."""

    @classmethod
    def load(cls, path: Path) -> InventoryConfig:
        raise NotImplementedError("G0 S0.4 T4")
