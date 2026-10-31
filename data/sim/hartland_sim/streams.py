"""The committed catalogue of named random streams (contracts §2).

One name per kind of draw (`demand`, `basket`, `acquire`, `churn`, `visit`, … — contracts §2 lists them); `rng()`
refuses a name not listed here, and a test asserts every `rng(` call site in the package names one. Filled by
G1 S1.1 T1, together with `rng.py`.
"""

from __future__ import annotations

STREAMS: frozenset[str] = frozenset()
