/* The leaves of okres.csv as polygons, for the okres of each store (05 §8): the 76 okresy outside Prague
   (admin_level 5 in OSM's Czech scheme; `ref` = the LAU 1 code) and Prague's 22 správní obvody (admin_level 6,
   "SO Praha 1" … "SO Praha 22"). `just osm-extract` → okres-boundaries.geojson, simplified.
   Data © OpenStreetMap contributors, ODbL 1.0. */
[out:json][timeout:600];
area["ISO3166-1"="CZ"]->.cz;
( rel["boundary"="administrative"]["admin_level"="5"]["ref"~"^CZ0"](area.cz);
  rel["boundary"="administrative"]["admin_level"="6"]["name"~"^SO Praha [0-9]+$"](area.cz); );
out geom;
