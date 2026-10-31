"""The OSM extract's endpoint and output names: what `cli.py` builds its parser from, without `io.osm_extract`'s
imports (numpy), so `hartland-sim --version` stays light. `io.osm_extract.EXTRACTS` writes exactly these outputs
(tested)."""

ENDPOINT = "https://overpass-api.de/api/interpreter"
OUTPUTS = ("osm-stores.csv", "osm-horeca.csv", "osm-context.csv", "osm/okres-boundaries.geojson")
