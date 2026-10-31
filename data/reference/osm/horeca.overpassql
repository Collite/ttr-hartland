/* The HoReCa places the sample is drawn from (05 §8): restaurants, cafés, bars, pubs, fast food, canteens.
   `just osm-extract` → data/reference/osm-horeca.csv. Data © OpenStreetMap contributors, ODbL 1.0. */
[out:csv(::type, ::id, ::lat, ::lon, name, brand, amenity, cuisine, "addr:street", "addr:place", "addr:housenumber",
         "addr:city", "addr:postcode", opening_hours)][timeout:600];
area["ISO3166-1"="CZ"]->.cz;
nwr["amenity"~"^(restaurant|cafe|bar|pub|fast_food|canteen)$"](area.cz);
out center;
