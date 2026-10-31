# ttr-hartland — the New Hartland

**Hartland CZ Group** is a fictional Czech FMCG manufacturer: instant coffee under **Harcafé**, dried and instant
soups and noodles under **Harsoup**. Four legal entities — the Center (Hartland CZ s.r.o., Praha) and three factories
(Harcafé Kolín, Harcafé Hodonín, Harsoup Chrudim) — two distribution centres (Jirny, Modřice), twelve own Harcafé
shops with cafeterias, and six sales channels: key-account retail, small retail (with the station shops), the own
shops, web, vending and HoReCa. One world, one country, **Czech only** — every generated row, label and document is
Czech.

This repo holds all of it: the generator that simulates the company day by day into Postgres (a warm-up year 2023,
history from 2024-01-01), the answer key of the stories planted in that data, the documents rendered from it, the
TTR model (db · er · md · cn · lexicon cs) the agents read, the agents' configuration and the run-sets.

Hartland, its people, figures and documents are invented. The stores are real places from OpenStreetMap, appearing as
customers of a fictional supplier. Only their OpenStreetMap entries are used (name, kind, location), never any data
of their own.

## The reveal

The data hides six stories and a set of smaller anomalies — seasonality, TV promotions, the reps' hidden skills,
store clustering, the area managers' PoS efficiency, discount efficiency — and the answer key is public:
[`data/stories/`](data/stories/) (the stories' parameters, a detector SQL per story, a probe per anomaly, the pinned
reps and customers). `just stories-check` runs it against a build; `just stories-check --null` proves the null world
gives the opposite. *(The key lands at G0 S0.4.)*

## Quick start

```sh
uv sync                     # Python 3.13 + the locked dependencies
just                        # the recipes
just test                   # ruff · mypy --strict · pytest (the unit tier)
just test -m component      # the Postgres 16 tier (Docker for testcontainers, or HARTLAND_TEST_PG=<server DSN>)
```

## The map

```
ttr-hartland/
├── LICENSE · LICENSE-DATA.md                MIT · the ODbL for the OSM extract and the generated world
├── justfile · pyproject.toml · uv.lock      the verbs (data now, model from M0) · the generator package
├── .github/workflows/data-gate.yml          lint · types · tests · reference-check · ddl drift (model-gate, regenerate: M0)
├── data/
│   ├── inventory.yaml                       every table: the one source of the DDL, the load order, the db layer
│   ├── world.yaml                           every number of the company (sites, channels, salesforce, pricing, dials)
│   ├── stories/                             the answer key (public) · detectors/*.sql · probes/*.sql
│   ├── reference/                           calendar · kraj/okres + affluence · regions/areas · weather · names ·
│   │                                        catalogue · the OSM extract · promotion and letter templates
│   ├── ddl/                                 schema.sql · indexes.sql · views/*.sql (generated, committed, drift-checked)
│   ├── sim/hartland_sim/ · sim/tests/       the generator
│   ├── recon/                               invariants · R0 · R1
│   ├── documents/<yyyy>/<mm>/               the rendered letters and sheets
│   └── build/                               outputs (ignored)
├── model/          db · er · md · cn · lexicon/cs · binding · layouts      (M0–M4)
├── generated/      the model and lexicon archives the cluster loads       (M0)
├── agents/         hartland.yaml · the Golem's Shem                       (M5)
├── automation/ · dashboards/                                              (empty until their efforts re-base)
├── run-set/        h-l/ (the twelve questions) · one-number/ · docwh/      (G5, DOC, M5)
└── scripts/
```

Only the generator's skeleton exists today; each folder arrives with the phase named above.

## How it is built

Coding by terminal Claude Code from the plan in `project/hartland/plan/` (Collite's task-tracking repo; the design
corpus is `project/hartland/design/`), phase by phase: **G0–G5** the data, **DOC** the documents, **M0–M5** the
model, **SW** the hartland cluster switch. One branch per phase (`feat/nh-<phase>-<short>`), one PR per phase;
`master` takes changes through pull requests with the data gate green (and the model gate from M0).

The generator is deterministic: ids are hashes of business keys and every random draw comes from a named stream
seeded from `world.yaml`'s `master_seed`, so the same inputs rebuild the same world, row for row.

## Licence and attribution

Two licences, split by what the files are. [`LICENSE-DATA.md`](LICENSE-DATA.md) has the details and the notice to keep.

- **MIT** ([`LICENSE`](LICENSE)) covers everything in this repo except the OSM extract. That includes the generator,
  the model, the agents, the scripts, the configuration, the answer key and the rendered documents in
  `data/documents/`.
- **ODbL 1.0** ([Open Database License](https://opendatacommons.org/licenses/odbl/1-0/)) covers the OpenStreetMap
  extract and every database built from it:
  - **The extract:** `data/reference/osm-stores.csv`, `osm-horeca.csv` and `okres-boundaries.geojson`, © OpenStreetMap
    contributors, <https://www.openstreetmap.org/copyright>. Extract: *(date · endpoint · row counts —
    `data/reference/osm-extract.meta`, filled at G0 S0.3)*.
  - **The generated world:** every CSV build, the `hartland` Postgres database and every published dump
    (`hartland-<yyyymmdd>.dump`). It is a Derivative Database of OpenStreetMap, published under the ODbL 1.0 as a
    whole (its individual contents under the DbCL 1.0). If you share it, or a database you made from it, keep the
    notice and share yours under the ODbL too.

The rendered documents and the run-sets' expected rows are Produced Works of that database. They are MIT, with the
OpenStreetMap notice beside them, and the database they came from is the published dump. Facts from other public
sources (the ČSÚ wage table, and others) are attributed in `data/reference/SOURCES.md` by the stage that adds them.
