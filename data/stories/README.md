# The answer key

What the generated world hides, and how to find it: the six stories of the brief (S1–S6) and the fifteen planted cases
(A-1…A-15). The key is public on purpose (⚑NH-16): the demonstrations reveal it at the end. The design is
`project/hartland/design/04-stories.md`; the contract is `project/hartland/plan/contracts.md` §3.

## The files

| file | holds |
|---|---|
| `s1.yaml` … `s6.yaml` | a story: its title, `enabled` (off = the null world), the stream its draws use, the hidden parameters the generator reads, and its `detector` (plus `also`, the further checks of the same story) |
| `anomalies.yaml` | the planted cases, each with its parameters and a `probe`; the `pins`: the eight reps of S3 with their skills and traits, and the customers the cases need |
| `detectors/*.sql`, `probes/*.sql` | one `SELECT` each, over the tables of `data/inventory.yaml` in the schema `hartland`, returning one number |

The schemas are `data/schemas/story.schema.json` and `anomalies.schema.json`; `config.StoriesConfig` loads and checks
them (`just stories-check --dry` lists them).

## How a story is judged

A detector returns its `metric`. The story passes when `metric direction threshold` holds on the build, and the
**null world** — the same seed with every story and case switched off — must give the opposite (`null_world`). A
story that passes in the null world is an artefact of the mechanics, not a story. A probe returns a count; `expect`
is what it must reach (`>= n`, `= n`, or `rows`), and in the null world it must find nothing.

A metric is never a ratio whose denominator the mechanics can bring to zero or below: where one side can lose, the
detector takes a difference, or a ratio of positive quantities (a lift, gain ÷ cost). A metric that comes out NULL (an
empty base, a region whose displays destroy margin) fails. Every window ends on the world's last order day, read from
the data, so a prolonged world is judged to its new end; the fixed windows (S1's 2024–2025, S4's 2025, a case's month)
stay.

`just stories-check` runs them all against the database (the runner arrives at G5 S5.2) and writes
`build/<run>/stories.md`. `--csv RUN` runs each detector's pandas twin over a CSV build instead
(`hartland_sim/checks/stories.py`, G1 S1.2): S1, S1-weather, S2, S2-affluence and S4 so far, a twin whose tables the
build does not hold yet is "not run", and S4's k-means (04's adjusted Rand index ≥ 0.6, s4.yaml `kmeans`) is reported
beside S4, not judged. Until then the SQL is written from the inventory, not yet against data: the
component tests run every file on the empty schema, and G5 S5.2 tunes the parameters (never the thresholds) on the
first full build.

A threshold comes from the mechanics, never from a build: S2-affluence's 1.35 lies between the ≈ 1.68 that `s2.yaml`'s
mechanics give over the real affluence quarters and the null world's 1.0. S5's historical spread (`historical/s5_best_worst_ratio.sql`) was set
that way at G2 S2.3 T6, before the first build (`hartland_sim/checks/s5_expected.py`): over the opening market the
mechanics give PHA's gain ÷ cost over VCH's 1.54 in the story and 1.21 in the null world — Prague's stores sell more,
so its displays gain more whoever places them — and the threshold, 1.35, lies between them. G5 S5.2 does not retune it.
It is provisional — S2.4 corrects the gain and re-derives it (NH-D17); it is not certified.

## The pins

Where a detector compares actors, it names the pinned ones: the reps by the area they hold on the warm-up's first day,
2023-01-01 (`slot`, an area of `data/reference/area.csv` — a rep changes areas twice, the pin follows the person, and a
later hire into the area is not the pin), the regions by their code, the product managers by their line. The SQL
repeats those codes, and a test keeps them equal to the YAML.

The customers a case needs are pinned by a rule — the `rank`-th customer in OSM order matching a channel, an area, a
chain or a store class — never by name: every store and café of the world is a real one from OpenStreetMap (⚑NH-2),
and its behaviour here is invented.


## S5 placement quality

The canonical S5 detector is `detectors/s5_best_worst.sql`. Bind one of `story`, `skill-neutral` or `full-null`
as its mode. Its result holds the audited regions, monthly rounds and all common calendar omission cases.
The empty omission list is the full window. The case fields `q_high`, `q_low` and `difference` are fractions.
The story requires `Q_H ≥ 0.70` and `D ≥ 0.10` on every date set. Each control must have `D < 0.10` throughout.
The historical regional rank query (`detectors/s5.sql`) keeps its original meaning and plays no part in certification.

`just stories-check --csv RUN` runs the source and audit-manifest judge before calculating the scores.
The source checks authenticate the CSV build from the current checkout and reject incomplete exports.
The full-protocol gate also compares those results to PostgreSQL on the same audited regional frame.
The label **Kvalita výběru umístění stojanů podle skupin** describes pre-decision forecasts. Causal return is
`NOT_ESTABLISHED`.

Prepare a receipt at a clean final commit and publish its SHA-256 before dispatching the full protocol.

```sh
head=$(git rev-parse HEAD)
uv run --locked python -m hartland_sim.checks.s5_certify prepare data/build/s5-freeze --head "$head"
gh workflow run data-gate.yml --ref <frozen-branch> -f s5_freeze=<published-receipt-sha256>
```

Each seed job builds its three worlds sequentially and stops on the first failed validation. The final job requires
thirty successful world receipts from that workflow run. A three-month smoke checks the path and reports
`NOT_CERTIFIED`. The finite-suite result concerns forecast-based decision quality. The retained historical spread
fixtures document the original ratio outcomes. [Earlier development NO-GO](https://github.com/Collite/ttr-hartland/pull/22)
