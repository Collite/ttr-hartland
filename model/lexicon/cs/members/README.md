# model/lexicon/cs/members — the member word lists

Seeded by `scripts/seed_members.py` (M0 S0.1 T5) from the committed reference data — **do not edit the `.ttrm`**: edit
the sources and re-seed (`uv run python scripts/seed_members.py`; the unit tests fail on a stale list).

| list | members | words |
|---|---|---|
| `sku` | the 123 SKUs of `catalogue.csv` | the name |
| `pack` | the catalogue's three pack words | `malá sklenice` · `velká sklenice` · `doplňovačka`, each the packs it means |
| `product` | the 48 products | the name + `nicknames_cs` (`klasik`, `zlatý kafe`, `trojka`, …) |
| `product_line` · `brand` | 11 lines · 2 brands | the name |
| `chain` | the 22 chains of the store classifier | the name + the sub-brands (`Jednota`, `Viva`, `Shell Select`, …) |
| `kraj` · `okres` | 14 kraje · 99 okresy (Prague's 22 districts among them) | nominative, genitive, locative (`data/reference/declension.csv`, hand-kept) |
| `site` | the 18 sites | the name; a DC's town (`Jirny`, `Modřice`) |

One term per member; a word names one member (the editor's `ttr/lexicon-duplicate-form` and
`ttr/lexicon-form-collides-with-name` hold the lists to it — so a factory's town, an okres's name too, is not a site's
word). Carriers have no list yet: their names are generated at G1 S1.1.

**The `for:` is a placeholder.** The pinned toolchain (ttr-core grammar/v0.21.1) reaches a member only as
`<er attribute>.<code>` with `valueLabels` on the attribute; there is no er layer before M1, and an EAN-13 SKU code
cannot be a reference segment. Until M3 S3.1 T5 re-targets each term to its member, it points at its member
vocabulary's column (`db.hartland.<table>.name`; the pack words `db.hartland.sku.pack_format`) and its `description`
names the member. Nothing compiles these before M3 (`build-lexicon`).

The vocabularies themselves are declared on the db columns (MV): the names `searchable method: TYPOS(1)` — `TOKENS`
for the multi-word SKU, product, kraj and site names — the codes `method: EXACT`.
