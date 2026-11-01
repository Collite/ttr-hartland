# model/db — the db layer

`model db schema hartland`: every table and view of `data/inventory.yaml` (= `data/ddl/schema.sql`), one file per
inventory `group` (design `07` §1) — `reference` · `org` · `customers` · `products` · `pricing` · `plans` · `orders`
· `finance` · `supply` · `delivery` · `field` · `marketing` · `documents` · `shops` — plus `views.ttrm` and
`fks.ttrm`. First written by `ttr import-schema` (ttr-core grammar/v0.20.3) against Postgres 16 with `schema.sql`
applied (M0 S0.1 T3), hand-kept since. `just ddl-drift` compares it with `schema.sql`; `model/db/tests/` checks its
shape.

What the import could not say, added by hand:

- **Numeric precision.** The importer writes a bare `decimal`; every `numeric(p,s)` column carries
  `{ type: decimal, length: p, precision: s }`. The rest is the importer's vocabulary: `bigint` · `int` (integer)
  · `text` (text, and `jsonb`, marked `// jsonb`) · `bool` · `date` · `datetime` (timestamptz) · `time`.
- **Deferrable foreign keys** — every FK is `DEFERRABLE INITIALLY DEFERRED` in the database; TTR-M has no flag for
  it, so `fks.ttrm` says it in its header.
- **The views** (the importer lists tables only): the four materialized views `sales_pl`, `trade_spend`,
  `route_current`, `receivable` as `def table` (⚑NH-P-10) and `return_request` as `def view` with its SQL.
  Nullability is the SELECT's: NOT NULL only where the SQL guarantees it.
- **Grains** — each table's `grain` from the inventory as its leading comment.

Known gaps (M0 S0.1 Found while doing): `document.language`, `competitor_sighting.description` and
`delivery_event.description` are not declarable — `language` and `description` are TTR-M keywords, not identifiers,
and the grammar has no quoting. `ddl-drift` reports the three as a known gap. The editor (TS semantics) also misreads
`db.hartland.area.<column>` (`area` is a kind keyword there); the metadata service's loader resolves it.
