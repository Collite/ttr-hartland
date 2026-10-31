# ttr-hartland — the New Hartland: one Czech FMCG company, its generated world, its model.
#
# The surface is contracts §7 (project/hartland/plan/contracts.md). The data verbs call the generator package
# (`hartland-sim`, data/sim/hartland_sim) through uv; a verb whose stage has not landed exits 1 naming that stage.
# The model verbs arrive at M0.

set positional-arguments

# List the recipes.
default:
    @just --list --unsorted

# ── the generator ─────────────────────────────────────────────────────────────

# ruff · mypy --strict · pytest — the unit tier; `just test -m component` runs the Postgres tier instead.
test *ARGS:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run pytest "$@"

# Format and auto-fix (ruff).
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Every data/reference/*.csv against its schema + row counts (G0 S0.2).
reference-check *ARGS:
    uv run hartland-sim reference-check "$@"

# The Overpass queries → data/reference/osm-*.csv + the extract's meta — Bora's Mac (G0 S0.3).
osm-extract *ARGS:
    uv run hartland-sim osm-extract "$@"

# data/inventory.yaml → data/ddl/schema.sql + indexes.sql; `--check` fails when the committed files differ (G0 S0.4).
ddl *ARGS:
    uv run hartland-sim ddl "$@"

# The simulation → data/build/<run>/ [--until DAY] [--csv] [--null] (G1 S1.4).
build-world *ARGS:
    uv run hartland-sim build-world "$@"

# COPY in load order · views · indexes · the _sim_meta guard [--db URL] [--csv-dir DIR] (G5 S5.1).
load-world *ARGS:
    uv run hartland-sim load-world "$@"

# From the last checkpoint up to DAY: the world reacts, appends (G5 S5.3).
prolong *ARGS:
    uv run hartland-sim prolong "$@"

# The answer key: every detector and probe [--null] [--csv RUN] [--dry] (G0 S0.4 --dry · G5 S5.2).
stories-check *ARGS:
    uv run hartland-sim stories-check "$@"

# The invariants + the R0/R1 diff (G5 S5.1).
recon *ARGS:
    uv run hartland-sim recon "$@"

# data/documents/ → the DocWH with the registry's metadata [--cluster NAME] (DOC.3).
load-documents *ARGS:
    uv run hartland-sim load-documents "$@"

# The DX slice validates against run-set/docwh/schemas/ (DOC.3).
docwh-slice-check *ARGS:
    uv run hartland-sim docwh-slice-check "$@"

# The twelve H-L questions as hand SQL, against the expected rows (G5 S5.4).
h-l-sql *ARGS:
    uv run hartland-sim h-l-sql "$@"

# The H-L corpus against CLUSTER's Golem (M5 S5.1).
h-l-run *ARGS:
    uv run hartland-sim h-l-run "$@"

# ── the model (contracts §7) — M0 fills these, in ttr-demo's shape ───────────
#
# verify-model · resolve-packages · check-model · ddl-drift · build-lexicon · check-lexicon · cn-hygiene ·
# check-cn-hygiene · verify-prompts · bi-one-number · load-model · regenerate
