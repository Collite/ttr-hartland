"""The year-end (`05` §3 year-end): the retro bonuses' credit notes (A-11) and the plan versions are G4 S4.3's. In G1
the hook exists so the loop calls it on 31 December, after the quarter-end."""

from __future__ import annotations

from datetime import date

from hartland_sim.loop.warmup import Output
from hartland_sim.period import Closing
from hartland_sim.world import World


def close(world: World, closing: Closing, day: date, output: Output) -> None:
    """The year-end of `day`'s year, on 31 December."""
    if (day.month, day.day) != (12, 31):
        raise ValueError(f"{day} is not a year's last day")
    del world, closing, output
    # S4.3: the retro credit notes (A-11), the plan versions.
