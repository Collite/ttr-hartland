# ttr-hartland — the New Hartland: one Czech FMCG company, its generated world, its model.
#
# The surface is contracts §7 (project/hartland/plan/contracts.md). The data verbs call the generator package
# (`hartland-sim`, data/sim/hartland_sim) through uv; a verb whose stage has not landed exits 1 naming that stage.
# The model verbs (M0) borrow a built ttr-core checkout; see their section below.

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

# Auto-fix, then format (ruff).
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Every data/reference/*.csv against its schema + row counts (G0 S0.2).
reference-check *ARGS:
    uv run hartland-sim reference-check "$@"

# The Overpass queries → data/reference/osm-*.csv + the meta; scripts/osm_extract_pbf.sh is the fallback (G0 S0.3).
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

# ── the model (contracts §7; ttr-demo's verbs, M0 S0.1) ──────────────────────
#
# The model verbs borrow a BUILT ttr-core checkout at toolchain.lock's [ttr-core].ref — the TS CLIs (pnpm build) and
# the Kotlin installDists — rather than vendoring one: TTR_CORE names it, else the sibling ../tatrman (the side-by-side
# layout CI builds, .github/actions/toolchain). The strict loader is a second checkout at grammar/v<[veles].ttr-metadata>:
# TTR_LOADER, else ../ttr-loader. (Env vars, not `cli=` parameters: just takes positional arguments only.)

tatrman := env_var_or_default("TTR_CORE", "../tatrman")
ttr_loader := env_var_or_default("TTR_LOADER", "../ttr-loader")

# The TS model verbs run this first: exit 2 when the ttr-core checkout is not at toolchain.lock's [ttr-core].ref
# (scripts/toolchain-check.sh, which regenerate-local.sh shares).
_toolchain:
    TTR_CORE="{{tatrman}}" bash scripts/toolchain-check.sh

# Every model suite (node's test runner, no deps): every *.test.mjs under model/.
verify-model: _toolchain
    TATRMAN_PACKAGES="{{tatrman}}/packages" node --test $(find model -name '*.test.mjs' | sort)

# Through a symlink named `ttr-hartland`: the CLI writes `generatedFrom` = the project directory's basename, so a
# worktree's name would leak into the artifact.
# generated/resolved-packages.json (packages, entities, areas) via the ttr-core Modeler CLI.
resolve-packages: _toolchain
    #!/usr/bin/env bash
    set -euo pipefail
    link="$(mktemp -d)"; trap 'rm -rf "$link"' EXIT
    ln -s "$(pwd)" "$link/ttr-hartland"
    node "{{tatrman}}/packages/migrate/dist/cli.js" resolve-packages "$link/ttr-hartland" --out "$(pwd)/generated/resolved-packages.json" --verbose

# Drift check: fail when the committed generated/resolved-packages.json is stale.
check-model: _toolchain
    #!/usr/bin/env bash
    set -euo pipefail
    link="$(mktemp -d)"; trap 'rm -rf "$link"' EXIT
    ln -s "$(pwd)" "$link/ttr-hartland"
    node "{{tatrman}}/packages/migrate/dist/cli.js" resolve-packages "$link/ttr-hartland" --check --out "$(pwd)/generated/resolved-packages.json"

# Tables, columns (name · type · nullability), primary keys, foreign keys, views (name · columns · a plain view's SQL);
# prints the differences, exit 1 on any (contracts §1), and refuses a SQL form it does not read. It parses the .ttrm
# files themselves, so no regenerate is needed first.
# data/ddl/schema.sql ⇄ model/db — the model-gate fails on any difference.
ddl-drift *ARGS: _toolchain
    TATRMAN_PACKAGES="{{tatrman}}/packages" uv run python scripts/ddl_drift.py "$@"

# The lexicon archive generated/lexicon.tar.zst (M3 S3.1).
build-lexicon:
    @echo "build-lexicon: the Czech lexicon lands at M3 S3.1 — nothing to build yet" >&2; exit 1

# Drift check of generated/lexicon.tar.zst (M3 S3.1; model-gate skips it until then).
check-lexicon:
    @echo "check-lexicon: the Czech lexicon lands at M3 S3.1 — nothing to check yet" >&2; exit 1

# `ttr cn hygiene` over model/ (M4 S4.1).
cn-hygiene:
    @echo "cn-hygiene: the cn layer lands at M4 S4.1" >&2; exit 1

# Drift check of model/cn/tests/hygiene.md (M4 S4.1).
check-cn-hygiene:
    @echo "check-cn-hygiene: the cn layer lands at M4 S4.1" >&2; exit 1

# The Shem's intent prompts against the plan-composer placeholder contract (M5 S5.1).
verify-prompts:
    @echo "verify-prompts: the Shem lands at M5 S5.1" >&2; exit 1

# The one-number check: net revenue from the fast path, sales_pl, the md cubelet and the GL agree (M2 S2.1).
bi-one-number:
    @echo "bi-one-number: the md layer and its one-number check land at M2 S2.1" >&2; exit 1

# The Kotlin loader at toolchain.lock's [veles].ttr-metadata (`just build-loader`), strict; its diagnostics JSON on
# stdout. Non-zero on a dropped file, a reconcile error, or a CLI that is not that version.
# The model as the metadata service loads it.
load-model:
    #!/usr/bin/env bash
    set -euo pipefail
    pins="$(python3 scripts/toolchain-lock.py)"   # its own line, so a failed read stops the recipe
    while IFS='=' read -r key value; do export "$key=$value"; done <<<"$pins"
    "{{ttr_loader}}/packages/kotlin/ttr-metadata-cli/build/install/ttr-metadata/bin/ttr-metadata" load "$(pwd)" --strict --expect-version "$VELES_TTR_METADATA" --json

# ttr-metadata-cli from the TTR_LOADER checkout at grammar/v<[veles].ttr-metadata>, stamped with that version. Prints
# the command that makes the checkout right and exits 2 when it is not.
# Build the loader `load-model` runs.
build-loader:
    #!/usr/bin/env bash
    set -euo pipefail
    pins="$(python3 scripts/toolchain-lock.py)"
    while IFS='=' read -r key value; do export "$key=$value"; done <<<"$pins"
    tag="grammar/v$VELES_TTR_METADATA"
    have="$(git -C "{{ttr_loader}}" rev-parse HEAD 2>/dev/null || true)"
    want="$(git -C "{{ttr_loader}}" rev-parse --verify --quiet "$tag^{commit}" 2>/dev/null || true)"
    if [ -z "$have" ] || [ "$have" != "$want" ]; then
      echo "build-loader: {{ttr_loader}} is at '${have:0:12}', not $tag (toolchain.lock [veles].ttr-metadata):" >&2
      echo "  git -C {{tatrman}} fetch --tags origin && git -C {{tatrman}} worktree add --detach {{ttr_loader}} $tag" >&2
      exit 2
    fi
    cd "{{ttr_loader}}"
    ./gradlew --no-configuration-cache -Pversion="$VELES_TTR_METADATA" :packages:kotlin:ttr-metadata-cli:installDist

# Rebuild generated/ with the toolchain toolchain.lock pins (scripts/regenerate-local.sh; regenerate.yml runs the same).
regenerate:
    bash scripts/regenerate-local.sh

# Whoever moves a [veles] pin runs this (scripts/check-veles-pin.sh; not CI, olymp is private): the metadata release
# olymp deploys on bp-dsk and hartland is [veles].image, and ttr-server's server-libs/v<image> names
# [veles].ttr-metadata. `just check-veles-pin [<olymp-checkout> [<revision>]]` — default ../olymp, origin/master.
check-veles-pin *ARGS:
    bash scripts/check-veles-pin.sh "$@"
