# model — the New Hartland TTR model

One flat `package hartland` (modeler.toml `[packages]`), one schema handle `hartland` = the Postgres DB `hartland`,
schema `hartland` (contracts §8), Czech only (NH-D8). The design is `project/hartland/design/07-model.md`; the layers
arrive phase by phase:

| Folder | Kind | Phase |
|---|---|---|
| `db/` | `model db schema hartland` — every table and view of `data/inventory.yaml`, imported from `schema.sql` | M0 |
| `er/` | `model er` — entities, relations, the stacked queries, the row rules | M1 |
| `md/` | `model md` — the conformed dimensions and cubelets | M2 |
| `lexicon/cs/` | `model lexicon locale cs` — terms, value labels, the member word lists (seeded at M0) | M0 · M3 |
| `binding/` | `er2db` · `md2db` · `cn` | M1 · M2 · M4 |
| `cn/` | `model cn` — the conceptual package | M4 |
| `layouts/` | the Golem's layouts | M5 |
| `tests/` | the model suites (`just verify-model`) | M0 → |

`connections.toml` — the one connection descriptor `pg-hartland` (CZK, cs). Not a TTR-M construct: a repo-level
descriptor the cluster's readers consume.

**Verify:** `just verify-model` (the suites, no database) · `just check-model` (generated/resolved-packages.json is
current) · `just ddl-drift` (schema.sql ⇄ `db/`) · `just load-model` (the metadata service's loader, strict). The
verbs borrow a built ttr-core checkout at toolchain.lock's ref: `TTR_CORE` (default `../tatrman`).

**Changing a table** (contracts §1): edit `data/inventory.yaml` → `just ddl` → the `db/` file by hand → `just
ddl-drift` clean, in one PR.
