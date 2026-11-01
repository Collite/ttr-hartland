"""Named random streams (contracts §2) — the only module allowed to create a random generator.

`rng(stream, *key)` returns a `numpy.random.Generator` seeded from `blake2b(master_seed ‖ stream ‖ key)`. Every draw
site names its stream from the catalogue in `streams.py`, so adding a stream or a draw never shifts another stream's
draws, and a prolong continues the same streams by (stream, key, day). A test asserts no `np.random`/`random` call
outside this module.

The key's parts are canonical before they are hashed (review-188): an integer is an `int` whatever numpy type it
came in, a date is its ISO text, text is `str`; anything else — a float, a set, `None`, an object — is refused, as
its text would not be a stable key. The parts are hashed as the `repr` of a tuple, so `(12, 3)` is not `(1, 23)` and
the text `"12"` is not the integer 12.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta

import numpy as np

from hartland_sim.streams import STREAMS

Generator = np.random.Generator  # for the draw sites' annotations: no module but this one names np.random

_master_seed: int | None = None


def master_seed() -> int:
    """The world's master seed: the one set by `using_master_seed`, else `world.yaml`'s."""
    global _master_seed
    if _master_seed is None:
        from hartland_sim.config import WorldConfig

        _master_seed = WorldConfig.load().master_seed
    return _master_seed


@contextmanager
def using_master_seed(seed: int) -> Iterator[None]:
    """Draw from another world's seed inside the block (a null world, a test), then go back."""
    global _master_seed
    previous = _master_seed
    _master_seed = seed
    try:
        yield
    finally:
        _master_seed = previous


def canonical(part: object) -> int | str | bool:
    """A key part as the value its seed is made of: `bool` · `int` (numpy's included) · ISO text for a date · `str`."""
    if isinstance(part, bool | np.bool_):
        return bool(part)
    if isinstance(part, int | np.integer):
        return int(part)
    if isinstance(part, datetime):
        return part.isoformat()
    if isinstance(part, date):
        return part.isoformat()
    if isinstance(part, str):
        return str(part)
    raise TypeError(f"key part {part!r} ({type(part).__name__}) is no text, integer or date")


def seed_of(stream: str, *key: object) -> int:
    """The 128-bit seed of (`stream`, `key`) under the master seed."""
    if stream not in STREAMS:
        raise ValueError(f"no stream {stream!r} in streams.STREAMS")
    parts = tuple(canonical(part) for part in key)
    text = f"{master_seed()}\x1f{stream}\x1f{parts!r}"
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=16).digest(), "big")


def rng(stream: str, *key: object) -> np.random.Generator:
    """The generator for one draw site: `stream` from `streams.STREAMS`, `key` the site's business key (+ day)."""
    return np.random.default_rng(seed_of(stream, *key))


def draw_day(draw: Generator, first: date, last: date) -> date:
    """A day drawn evenly from `first` to `last`, both included — one integer from `draw`."""
    return first + timedelta(days=int(draw.integers(0, (last - first).days + 1)))
