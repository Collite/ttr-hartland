"""In and out of the world (architecture §2): `csv_writer` (streaming, per table, in load order) · `pg_loader`
(COPY, views, indexes, the `_sim_meta` guard) · `db_reader` (the world reacts: prolong reads the DB) · `ddl`
(`inventory.yaml` → `data/ddl/`). G0 S0.4 · G1 S1.1 · G5."""
