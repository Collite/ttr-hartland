"""The store classifier (G0 S0.3): an OSM object → its okres (`okres`), its chain (`chains`), its street type and
size class (`place`) and its weekly hours (`hours`). `setup.market` puts them together into `store` and `chain`.

Where the classifier has to choose (the HoReCa sample's order, the hours of an untagged store), it goes by
`osm_order`, a hash of the OSM id: the choice is fixed by the extract, not by the world's seed, so a re-seeded world
keeps the same store set."""

import hashlib


def osm_order(osm_id: str) -> int:
    """A fixed, well-mixed number for an OSM id (`node/123`): blake2b, 64 bits."""
    return int.from_bytes(hashlib.blake2b(osm_id.encode(), digest_size=8).digest(), "big")
