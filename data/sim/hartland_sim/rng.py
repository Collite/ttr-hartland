"""Named random streams (contracts §2) — the only module allowed to create a random generator.

`rng(stream, *key)` returns a `numpy.random.Generator` seeded from `blake2b(master_seed ‖ stream ‖ key)`. Every draw
site names its stream from the catalogue in `streams.py`, so adding a stream or a draw never shifts another stream's
draws, and a prolong continues the same streams by (stream, key, day). A test asserts no `np.random`/`random` call
outside this module.

`uniforms(stream, *key, ids=…)` is the vectorised twin for a draw per entity on a day: one uniform per id, each a
counter-based hash of the stream's seed for `key` and the id alone (splitmix64), so an id's draw never depends on
which other ids are drawn with it, and `pairs` makes one id of two (a point and a SKU, a customer and a line).

`frozen_diagnostic_rng(seed)` is the one exception (contracts §2, NH-D21-A): a fresh `numpy.random.default_rng(seed)`
for an offline diagnostic that froze its own seed (G2 S2.4's S5 prototype, `checks.s5_pooled`) — no master seed, no
stream, and drawing from it moves neither; a test keeps every other module of the package from it.

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
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

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


def frozen_diagnostic_rng(seed: int) -> Generator:
    """A fresh `numpy.random.default_rng(seed)` for an offline diagnostic (module docstring): `seed` a nonnegative
    integer the diagnostic froze; never the master seed's, never a stream's, and each call starts again."""
    if isinstance(seed, bool | np.bool_) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ValueError(f"a diagnostic seed is a nonnegative integer, not {seed!r}")
    return np.random.default_rng(int(seed))


def draw_day(draw: Generator, first: date, last: date) -> date:
    """A day drawn evenly from `first` to `last`, both included — one integer from `draw`."""
    return first + timedelta(days=int(draw.integers(0, (last - first).days + 1)))


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MULTIPLY = (np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB))
_MASK = (1 << 64) - 1


def _mix(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """splitmix64's finaliser, element-wise over uint64 (the arithmetic wraps)."""
    x = x + _GOLDEN
    x = (x ^ (x >> np.uint64(30))) * _MULTIPLY[0]
    x = (x ^ (x >> np.uint64(27))) * _MULTIPLY[1]
    return np.asarray(x ^ (x >> np.uint64(31)))


def _bits(ids: ArrayLike) -> np.ndarray[Any, Any]:
    return np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).view(np.uint64)


def uniforms(stream: str, *key: object, ids: ArrayLike) -> np.ndarray[Any, Any]:
    """A uniform in [0, 1) per element of `ids` (int64, any shape), from `stream` and `key` (module docstring)."""
    seed = np.uint64(seed_of(stream, *key) & _MASK)
    mixed = _mix(_mix(_bits(ids) ^ seed) ^ seed)
    return np.asarray((mixed >> np.uint64(11)).astype(np.float64) * 2.0**-53)


def pairs(first: ArrayLike, second: ArrayLike) -> np.ndarray[Any, Any]:
    """One int64 id per pair of ids, broadcast (`pairs(points[:, None], skus[None, :])`)."""
    return np.ascontiguousarray(_mix(_mix(_bits(first)) ^ _bits(second))).view(np.int64)
