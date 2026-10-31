"""In and out of the world (architecture §2): `reference` (the committed `data/reference/` files, schema-typed) ·
`osm_extract` (`just osm-extract`: Overpass → the OSM snapshot) ·
`csv_writer` (streaming, per table, in load order) · `pg_loader` (COPY, views, indexes, the `_sim_meta` guard) ·
`db_reader` (the world reacts: prolong reads the DB) · `ddl` (`inventory.yaml` → `data/ddl/`).
G0 S0.2 · G0 S0.3 · G0 S0.4 · G1 S1.1 · G5."""
