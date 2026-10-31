# The data licences

The code and everything we wrote in this repo is MIT ([`LICENSE`](LICENSE)). The OpenStreetMap extract, and every
database built from it, is under the Open Database License 1.0 instead. This file says which is which and gives the
notice to keep.

## 1. The OpenStreetMap extract — ODbL 1.0

| file | what |
|---|---|
| `data/reference/osm-stores.csv` | shops: the key-account, small-retail and station stores |
| `data/reference/osm-horeca.csv` | cafés, restaurants, pubs, hotels |
| `data/reference/okres-boundaries.geojson` | the okres boundaries |
| `data/reference/*.overpassql` · `osm-extract.meta` | the queries, the date and endpoint of the extract, the row counts |

© OpenStreetMap contributors. Available under the
[Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/) —
<https://www.openstreetmap.org/copyright>. The files arrive at G0 S0.3.

## 2. The generated world — ODbL 1.0, as a whole

The generator turns the extract into the company's stores and customers. The orders, deliveries, visits, routes
and invoices all hang off them, so the world is a Derivative Database of OpenStreetMap:

- every CSV build (`data/build/<run>/`);
- the Postgres database `hartland` loaded from a build, and every dump of it (`hartland-<yyyymmdd>.dump`);
- any extract of its tables.

It is published under the ODbL 1.0 as a whole, with this notice:

> The New Hartland database is made available under the Open Database License 1.0:
> <https://opendatacommons.org/licenses/odbl/1-0/>. Any rights in individual contents of the database are licensed
> under the Database Contents License 1.0: <https://opendatacommons.org/licenses/dbcl/1-0/>. It contains information
> from OpenStreetMap, © OpenStreetMap contributors, available under the ODbL.

The notice travels with the data in two ways:
- **inside the database:** a comment on the schema `hartland`, in Czech, so every dump carries it (`schema.sql`,
  from G0 S0.4);
- **beside every published dump:** a copy of this file (G5 S5.4).

If you share the world, or a database you made from it:
- keep the notice;
- share your database under the ODbL as well (ODbL §4.4);
- if you publish something produced from it, say where the database is available (§4.6).

## 3. Documents and query results — MIT, with the notice

The rendered letters and sheets in `data/documents/` and the expected rows in `run-set/` are Produced Works of the
generated world (ODbL §4.3):
- **licence:** they are MIT, like the rest of the repo;
- **notice:** they carry the OpenStreetMap notice beside them, in `data/documents/NOTICE.md` from DOC.1. It is not
  inside each document, because a letter is part of the world and Czech only;
- **the database they came from:** the published dump (§2), and the generator in this repo rebuilds it.

## 4. Other public sources

Some facts come from other public sources: ČSÚ's population by okres and average wage by kraj (G0 S0.2), and others
as their stages add them. Each is named with its source and terms in `data/reference/SOURCES.md`, written by the
stage that adds it. The names of people are synthetic.
