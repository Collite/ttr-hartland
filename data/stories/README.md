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

`just stories-check` runs them all against the database (the runner arrives at G5 S5.2, `--csv` reads a CSV build)
and writes `build/<run>/stories.md`. Until then the SQL is written from the inventory, not yet against data: the
component tests run every file on the empty schema, and G5 S5.2 tunes the parameters (never the thresholds) on the
first full build.

A threshold comes from the mechanics, never from a build: S2-affluence's 1.35 lies between the ≈ 1.68 that `s2.yaml`'s
mechanics give over the real affluence quarters and the null world's 1.0. One is still to be set that way: S5's
spread (`s5_best_worst.sql`), whose 3 is `04`'s, set for the return rather than for gain ÷ cost. G2 S2.3 T1 builds
`display_uplift`'s street × size grid, derives the spread that grid gives over the market, and sets the threshold
between it and 1.0, before the first build.

## The pins

Where a detector compares actors, it names the pinned ones: the reps by the area they hold on the warm-up's first day,
2023-01-01 (`slot`, an area of `data/reference/area.csv` — a rep changes areas twice, the pin follows the person, and a
later hire into the area is not the pin), the regions by their code, the product managers by their line. The SQL
repeats those codes, and a test keeps them equal to the YAML.

The customers a case needs are pinned by a rule — the `rank`-th customer in OSM order matching a channel, an area, a
chain or a store class — never by name: every store and café of the world is a real one from OpenStreetMap (⚑NH-2),
and its behaviour here is invented.
