"""hartland_sim — the New Hartland generator (architecture §2).

One Czech FMCG company, Hartland CZ Group, simulated day by day from committed inputs — `data/world.yaml`,
`data/stories/`, `data/reference/` — into the tables of `data/inventory.yaml`, then COPY-loaded into Postgres.
Deterministic: the same inputs give the same rows and the same ids (contracts §2). Czech only (NH-D8).

Modules: `cli` (the verbs, contracts §7) · `config` (the three configs) · `rng` + `streams` (named random streams,
contracts §2) · `ids` (business-key ids, contracts §2) · `world` (the state and its checkpoint, contracts §2).
Packages: `setup` · `loop` · `period` · `documents` · `io` · `checks`.
"""

__version__ = "0.11.0"
