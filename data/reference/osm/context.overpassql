/* What a store's street type is read from (05 §8, 04 S4): railway stations and halts, bus stations, shopping malls,
   and the settlements' centre points with their kind and population.
   `just osm-extract` → data/reference/osm-context.csv. Data © OpenStreetMap contributors, ODbL 1.0. */
[out:csv(::type, ::id, ::lat, ::lon, name, railway, station, amenity, shop, place, population)][timeout:600];
area["ISO3166-1"="CZ"]->.cz;
( nwr["railway"~"^(station|halt)$"](area.cz);
  nwr["amenity"="bus_station"](area.cz);
  nwr["shop"="mall"](area.cz);
  node["place"~"^(city|town|village|suburb|quarter)$"](area.cz); );
out center;
