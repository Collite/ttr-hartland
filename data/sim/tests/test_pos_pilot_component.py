"""A pilot run loads (G2 S2.6 T4, component; NH-D22): a story pilot built to the assignment's month-end, prolonged over
its effective day with one selection's customer lost, COPYs whole into the `hartland` schema in `load_order` inside one
transaction; the commit's deferred foreign keys hold — the cancelled decision links no loan, every other selected one
its loan — and the outcome and arm checks take the rows. `lifecycle_phase` is left out: a build renders a customer
acquired on the warm-up's first day with a zero-length `potential` episode beside its first one, under the same id
(the default build's as much as a pilot's; S2.6 Found), and no table refers to it."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import psycopg
import pytest
from psycopg import sql

from hartland_sim.config import InventoryConfig, StoriesConfig, WorldConfig
from hartland_sim.loop.build import build, prolong
from hartland_sim.rng import uniforms
from hartland_sim.setup.market import Market
from hartland_sim.world import Checkpoint

pytestmark = pytest.mark.component

WORLD = WorldConfig.load()
STORIES = StoriesConfig.load()
END, FIRST = date(2023, 2, 28), date(2023, 3, 1)


def test_a_pilot_run_with_a_cancellation_loads_and_its_keys_hold(
    osm_market: Market, hartland_db: psycopg.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "pilot"
    build(run, END, WORLD, STORIES, market=osm_market, pilot="story")
    held = Checkpoint.load(run, END).restore()
    assert held.pilot is not None
    victim = next(
        d for d in held.pilot.decisions.values() if d.outcome == "selected" and not held.customers[d.customer_id].traits
    )
    lost = np.array([victim.customer_id], dtype=np.int64)

    def forced(stream: str, *key: object, ids: np.ndarray) -> np.ndarray:
        out = uniforms(stream, *key, ids=ids)
        return np.where(np.isin(ids, lost), 0.0, out) if (stream, key) == ("churn", (FIRST,)) else out

    monkeypatch.setattr("hartland_sim.loop.acquisition.uniforms", forced)
    prolong(run, FIRST, WORLD, STORIES)
    with hartland_db.transaction():
        for table in InventoryConfig.load().tables:
            path = run / f"{table.name}.csv"
            if path.exists() and table.name != "lifecycle_phase":
                statement = sql.SQL("COPY hartland.{} FROM STDIN (FORMAT csv, HEADER true)").format(
                    sql.Identifier(table.name)
                )
                with hartland_db.cursor() as cursor, cursor.copy(statement) as copy:
                    copy.write(path.read_bytes())
    found = hartland_db.execute("SELECT arm, count(*) FROM hartland.pos_pilot_assignment GROUP BY arm").fetchall()
    arms: dict[str, int] = dict(found)
    assert sum(arms.values()) == 8160 and arms["programme"] == arms["holdout"]
    found = hartland_db.execute(
        "SELECT outcome, loan_id IS NULL, count(*) FROM hartland.pos_placement_decision GROUP BY 1, 2"
    ).fetchall()
    outcomes = {(outcome, unlinked): n for outcome, unlinked, n in found}
    assert outcomes[("cancelled", True)] == 1 and ("selected", True) not in outcomes
    assert all(unlinked for (outcome, unlinked) in outcomes if outcome != "selected")
    (joined,) = hartland_db.execute(
        "SELECT count(*) FROM hartland.pos_placement_decision d JOIN hartland.pos_loan l ON l.pos_loan_id = d.loan_id"
        " JOIN hartland.pos_pilot_assignment a ON a.customer_id = d.customer_id"
        " WHERE a.arm = 'programme' AND l.customer_id = d.customer_id AND l.lent_on = d.decision_day + 1"
    ).fetchone() or (0,)
    (loans,) = hartland_db.execute("SELECT count(*) FROM hartland.pos_loan").fetchone() or (0,)
    assert joined == outcomes[("selected", False)] == loans
