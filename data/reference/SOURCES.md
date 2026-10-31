# Where the reference data comes from

Every file in `data/reference/` is one of three kinds:

- **public facts**, named here with their source, year and terms;
- **rules**, computed by the generator and drift-tested;
- **the world's own**, invented for Hartland and kept by hand.

Each file is checked against its schema in `schemas/` by `just reference-check`. The OpenStreetMap extract (G0 S0.3)
is under the ODbL; see [`LICENSE-DATA.md`](../../LICENSE-DATA.md).

## Public facts

**Terms.** Data from ČSÚ (Český statistický úřad) is published under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with the attribution "Zdroj: ČSÚ"
([ČSÚ's terms](https://csu.gov.cz/podminky_pro_vyuzivani_a_dalsi_zverejnovani_statistickych_udaju_csu)). Neither ČSÚ
nor anyone else endorses Hartland. The figures below are copied as published, rounded to whole units, or derived as
stated.

| file | what | source |
|---|---|---|
| `kraj.csv` | the 14 kraje: CZ-NUTS 3 code, name | ČSÚ open data, dataset `OBY02A` (its territory list) |
| `okres.csv` | the 76 okresy + Praha: LAU 1 code, name, kraj, population on 1 January 2026 | ČSÚ `OBY02A` "Počet obyvatel podle pohlaví", indicator "Počet obyvatel k 1. 1.", both sexes, 2026 |
| `okres.csv` | the 22 Prague districts (správní obvody, `CZ0100-01` … `CZ0100-22` = ČSÚ codes `1101` … `1122`): population | derived, see below |
| `wage_by_kraj.csv` | the average gross monthly wage per full-time equivalent, all sections A–S, all enterprise sizes, workplace method, 2025: each kraj and Česko | ČSÚ open data, dataset `MZDR` |
| `calendar.csv` | the thirteen days off | zákon č. 245/2000 Sb., o státních svátcích, o ostatních svátcích, o významných dnech a o dnech pracovního klidu |

**The Prague districts' population.** ČSÚ publishes register populations for Prague's 57 municipal parts up to
1 January 2022 (`OBY02A`). Each part is assigned to its district by the 2021 census (`SLD0100`, level `PHA22`).
Each district's 2022 share of Prague is then applied to Prague's 2026 total, and the largest-remainder rounding makes
the 22 add up to it exactly.

## Rules

| file | rule |
|---|---|
| `calendar.csv` | `hartland_sim.setup.calendar`: ISO weeks, the Czech days off with Good Friday and Easter Monday from the Gregorian Easter, meteorological seasons. A test fails when the file and the rules differ. |

## The world's own

| file | what it is |
|---|---|
| `wage_by_okres.csv` | Each okres's wage relative to its kraj's (`wage_vs_kraj`). **Not a ČSÚ figure:** ČSÚ publishes the average wage down to the kraj only, and neither its open data nor the kraj yearbooks have an okres table. The spread is hand-kept and normalised so that each kraj's population-weighted mean is 1, which keeps ČSÚ's kraj figure (a test). A Prague district spreads around Prague's figure. `hartland_sim.setup.affluence` turns the kraj wage × the spread ÷ the Česko wage into the affluence index, clamped to 0.8–1.3 (⚑NH-17). |
| `region.csv`, `area.csv`, `area_okres.csv` | the six sales regions of `01` §4.3; the 50 rep areas (10 in Prague over the 22 districts); each okres in exactly one area |
| `weather_normals.csv` | monthly temperature and precipitation per region, shaped like the Czech climate (not ČHMÚ figures) |
| `severe_days.csv` | the planted A-5 days, which include a flood on 13–16 September 2024 in Moravia and Silesia, as happened, plus a seeded background of snow, ice, storm and wind days per region and year |
| `names_first.csv`, `names_last.csv` | common Czech names, declined by hand; every person in the world is synthetic |
| `legal_entity.csv` | the four companies of the group; their IČO are fictional, checksum-valid, and assigned to no one in ARES on 2026-10-31 |
| `site.csv` | the group's 18 sites in real towns, at approximate town coordinates; the shops and the addresses are invented |
| `brand.csv`, `product_line.csv`, `product.csv`, `catalogue.csv` | the portfolio of `01` §3: two brands, 11 lines, the products and their SKUs (117 in the range on 2026-10-31, 12 of them launched and 6 others delisted during the history). Prices and costs are invented, shaped like the Czech market of 2024; `sku_code` is an EAN-13 in GS1's 29x prefix, which is for restricted circulation, so no real product carries one |
| `competitor.csv`, `competitor_product.csv` | the four competitors of ⚑NH-18 and 40 of their products, each with the Hartland product it competes with; brands, products and prices are invented |
| `promotion_templates.yaml`, `visit_note_templates.yaml`, `templates/` | the promotion calendar's 25 yearly shapes, the sentences of the reps' visit notes, and the seeds of the letter and marketing templates; written for Hartland |
| `cost_centre.csv`, `gl_account.csv` | the Center's cost centres; accounts numbered on the Czech chart of accounts (classes 1–7), one or more per P/L line of the contracts, plus VAT, the balance accounts and `361100`, what the Center owes the three factories (intercompany) |
