"""The warm-up switch (contracts §2, `05` §3 step 0): the loop runs from `history.warm_up_from` with its output
suppressed, so the first written day, `history.first_day`, starts with a year of history behind it — customers gained
and lost, baskets part-filled, phases moved — and nothing in a build is "the first day".

`Output` stands between the loop and the CSV writer: a day's rows reach the writer only from `first_day` on. Setup's
own rows are not the loop's and go through the writer directly (`setup.opening`); the opening's `stock_day`, dated
the day before the warm-up, is setup's (G1 S1.4 T4 splits `opening.write` for a full build).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from hartland_sim.config import WorldConfig
from hartland_sim.io.csv_writer import CsvWriter


@dataclass
class Output:
    """The loop's rows of a day, written from `first_day` on, dropped before."""

    writer: CsvWriter
    first_day: date

    @classmethod
    def of(cls, writer: CsvWriter, world: WorldConfig) -> Output:
        return cls(writer, world.history.first_day)

    def writes(self, day: date) -> bool:
        return day >= self.first_day

    def append(self, day: date, table: str, rows: Iterable[Mapping[str, Any]]) -> int:
        """`rows` of `day` into `table`'s file once the warm-up is over; the number written."""
        return self.writer.append(table, rows) if self.writes(day) else 0
