"""hartland_sim — the New Hartland generator (architecture §2).

One Czech FMCG company, Hartland CZ Group, simulated day by day from committed inputs — `data/world.yaml`,
`data/stories/`, `data/reference/` — into the tables of `data/inventory.yaml`, then COPY-loaded into Postgres.
Deterministic: the same inputs give the same rows and the same ids (contracts §2). Czech only (NH-D8).

Modules: `cli` (the verbs, contracts §7) · `config` (the three configs) · `rng` + `streams` (named random streams,
contracts §2) · `ids` (business-key ids, contracts §2) · `world` (the state and its checkpoint, contracts §2).
Packages: `setup` · `loop` · `period` · `documents` · `io` · `checks`.
"""

__version__ = "0.11.0"
# The generator's revision (⚑R221-C): what a build writes — its rows, its ids, its draws. Bump it in the PR that changes
# any of them, so a run is extended only by a generator that writes as the one that built it (`build_id` hashes it).
# The package version above is the release's, not this.
REVISION = 6  # 6: G2 S2.5, the PoS effective day; 5: G2 S2.3, the point of sale; 4: G2 S2.2, the rep prompts;
# 3: G2 S2.1, the field; 2: G1 S1.5
