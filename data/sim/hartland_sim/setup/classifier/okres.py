"""The okres of a point (G0 S0.3 T2): point-in-polygon against `data/reference/osm/okres-boundaries.geojson`.

The leaves are `okres.csv`'s: the 76 okresy outside Prague and Prague's 22 districts (`CZ0100-01` … `-22`). A ring is
read even–odd, so a hole is outside. The boundaries are simplified (`io.osm_extract.TOLERANCE`), which opens slivers
of a few metres between neighbours; a point in none of the polygons takes the nearest leaf within `SNAP_M`, and a
point farther than that is outside Czechia — dropped, and counted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hartland_sim.io.reference import REFERENCE_DIR

BOUNDARIES = REFERENCE_DIR / "osm" / "okres-boundaries.geojson"
SNAP_M = 300.0
METRES_PER_DEGREE = 111_320.0
COS_LAT = math.cos(math.radians(49.8))  # Czechia's middle latitude: a degree of longitude ≈ 0.65 of one of latitude

Points = NDArray[np.float64]  # (n, 2): lon, lat


@dataclass(frozen=True)
class Leaf:
    code: str
    rings: list[Points]
    low: Points = field(init=False)
    high: Points = field(init=False)

    def __post_init__(self) -> None:
        every = np.vstack(self.rings)
        object.__setattr__(self, "low", every.min(axis=0))
        object.__setattr__(self, "high", every.max(axis=0))


@dataclass(frozen=True)
class Assignment:
    codes: list[str | None]  # per point: its leaf, or None outside Czechia
    snapped: int  # in no polygon, but within SNAP_M of one
    outside: int  # dropped


def load(path: Path = BOUNDARIES) -> list[Leaf]:
    collection = json.loads(path.read_text(encoding="utf-8"))
    return [
        Leaf(
            feature["properties"]["code"],
            [np.asarray(ring, dtype=float) for polygon in feature["geometry"]["coordinates"] for ring in polygon],
        )
        for feature in collection["features"]
    ]


def contains(rings: Sequence[Points], points: Points) -> NDArray[np.bool_]:
    """Whether each point is inside the rings, even–odd (a ray cast to the east)."""
    inside = np.zeros(len(points), dtype=bool)
    x, y = points[:, :1], points[:, 1:]
    for ring in rings:
        x1, y1, x2, y2 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            crosses = ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        inside ^= (crosses.sum(axis=1) % 2).astype(bool)
    return inside


def distance_m(rings: Sequence[Points], points: Points) -> NDArray[np.float64]:
    """Each point's distance to the nearest edge of the rings, in metres (a local flat projection)."""
    scale = np.array([COS_LAT * METRES_PER_DEGREE, METRES_PER_DEGREE])
    p = points[:, None, :] * scale
    best = np.full(len(points), np.inf)
    for ring in rings:
        a, b = ring[:-1] * scale, ring[1:] * scale
        ab = b - a
        length2 = np.maximum((ab**2).sum(axis=1), 1e-12)
        t = np.clip(((p - a) * ab).sum(axis=2) / length2, 0.0, 1.0)
        nearest = a + t[..., None] * ab
        best = np.minimum(best, np.sqrt(((p - nearest) ** 2).sum(axis=2)).min(axis=1))
    return best


def assign(leaves: Sequence[Leaf], points: Points, snap_m: float = SNAP_M) -> Assignment:
    """The leaf of each point (lon, lat)."""
    codes: list[str | None] = [None] * len(points)
    for leaf in leaves:
        candidates = np.flatnonzero(np.all((points >= leaf.low) & (points <= leaf.high), axis=1))
        candidates = np.array([i for i in candidates if codes[i] is None], dtype=int)
        if len(candidates):
            for i in candidates[contains(leaf.rings, points[candidates])]:
                codes[i] = leaf.code
    missing = np.array([i for i, code in enumerate(codes) if code is None], dtype=int)
    snapped = 0
    if len(missing):
        margin = np.array([snap_m / (COS_LAT * METRES_PER_DEGREE), snap_m / METRES_PER_DEGREE])
        best = np.full(len(missing), np.inf)
        nearest: list[str | None] = [None] * len(missing)
        for leaf in leaves:
            near = np.flatnonzero(
                np.all((points[missing] >= leaf.low - margin) & (points[missing] <= leaf.high + margin), axis=1)
            )
            if not len(near):
                continue
            distance = distance_m(leaf.rings, points[missing[near]])
            for j, d in zip(near.tolist(), distance.tolist(), strict=True):
                if d < best[j]:
                    best[j], nearest[j] = d, leaf.code
        for k, i in enumerate(missing.tolist()):
            if best[k] <= snap_m:
                codes[i] = nearest[k]
                snapped += 1
    return Assignment(codes, snapped, sum(code is None for code in codes))
