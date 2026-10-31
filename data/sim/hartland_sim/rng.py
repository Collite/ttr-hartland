"""Named random streams (contracts §2) — the only module allowed to create a random generator.

`rng(stream, *key)` returns a `numpy.random.Generator` seeded from `blake2b(master_seed ‖ stream ‖ key)`. Every draw
site names its stream from the catalogue in `streams.py`, so adding a stream or a draw never shifts another stream's
draws, and a prolong continues the same streams by (stream, key, day). A test asserts no `np.random`/`random` call
outside this module. G1 S1.1 T1.
"""

from __future__ import annotations

import numpy as np


def rng(stream: str, *key: object) -> np.random.Generator:
    """The generator for one draw site: `stream` from `streams.STREAMS`, `key` the site's business key (+ day)."""
    raise NotImplementedError("G1 S1.1 T1")
