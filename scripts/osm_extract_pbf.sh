#!/usr/bin/env bash
# The fallback of `just osm-extract` (05 §8) for when Overpass is too slow: Geofabrik's Czech extract, filtered by
# osmium with the tag sets of data/reference/osm/*.overpassql, then written as the same files by
# `hartland-sim osm-extract --from-pbf`, with the PBF's name and timestamp in the meta. Data © OpenStreetMap
# contributors, ODbL 1.0.
#
# Needs osmium-tool ≥ 1.14 and curl. Run from the repo root:
#   scripts/osm_extract_pbf.sh [czech-republic-latest.osm.pbf]     (without an argument it is downloaded)
set -euo pipefail

url=https://download.geofabrik.de/europe/czech-republic-latest.osm.pbf
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
pbf="${1:-}"
if [ -z "$pbf" ]; then
  pbf="$work/czech-republic-latest.osm.pbf"
  curl -fL --retry 3 -o "$pbf" "$url"
  source="$url"
else
  source="$(basename "$pbf")"
fi
osm_base="$(osmium fileinfo -g header.option.osmosis_replication_timestamp "$pbf")"

# $1 = the query's stem; the rest = osmium filter expressions (wider than the query; the converter applies the rest:
# a station's shop, Makro's brand, the boundaries' admin_level).
extract() {
  local stem="$1"
  shift
  osmium tags-filter --overwrite -o "$work/$stem.osm.pbf" "$pbf" "$@"
  osmium export --overwrite -f geojsonseq --add-unique-id=type_id -o "$work/$stem.geojsonseq" "$work/$stem.osm.pbf"
}

extract stores nwr/shop=supermarket,convenience,grocery,general,greengrocer,deli,wholesale nwr/amenity=fuel
extract horeca nwr/amenity=restaurant,cafe,bar,pub,fast_food,canteen
extract context nwr/railway=station,halt nwr/amenity=bus_station nwr/shop=mall n/place=city,town,village,suburb,quarter
extract okres-boundaries r/boundary=administrative

uv run hartland-sim osm-extract --from-pbf "$work" \
  --pbf-source "$source (osmium)" --pbf-osm-base "${osm_base:-unknown}"
