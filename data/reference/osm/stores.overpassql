/* The grocery store set (05 §8, ⚑NH-2): every food shop and every fuel station in Czechia, and Makro's cash-and-carry
   stores (shop=wholesale; Makro is one of the nine KA chains, 01 §4). shop=kiosk is not taken. A station tagged
   shop=no says it has no shop: it counts only through its brand.
   `just osm-extract` → data/reference/osm-stores.csv. Data © OpenStreetMap contributors, ODbL 1.0. */
[out:csv(::type, ::id, ::lat, ::lon, name, brand, operator, shop, amenity, "addr:street", "addr:place",
         "addr:housenumber", "addr:city", "addr:postcode", opening_hours)][timeout:600];
area["ISO3166-1"="CZ"]->.cz;
( nwr["shop"~"^(supermarket|convenience|grocery|general|greengrocer|deli)$"](area.cz);
  nwr["amenity"="fuel"]["shop"]["shop"!="no"](area.cz);
  nwr["amenity"="fuel"]["brand"](area.cz);
  nwr["shop"="wholesale"]["brand"~"^makro$",i](area.cz); );
out center;
