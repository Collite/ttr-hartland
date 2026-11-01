"""The `World` and its checkpoint (contracts §2; G1 S1.1 T3). Setup's own world round-trips in `test_setup.py`."""

from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from hartland_sim.world import (
    CHECKPOINT_FILE,
    Actor,
    Budget,
    Checkpoint,
    Condition,
    Customer,
    Duty,
    Episode,
    Meta,
    Skills,
    Stock,
    World,
    load_all,
    unordered_containers,
)

WARM_UP_END = date(2023, 12, 31)


def _world(day: date = date(2022, 12, 31)) -> World:
    world = World(day)
    world.customers[11] = Customer(
        11, "RT-000001", "Potraviny U Lípy", "small_retail", None, "active", date(2021, 5, 3), None, "CZ0201",
        "SJC-06", "node/1", None, [Episode("occasional", date(2021, 5, 3), "První objednávka")],
        traits=("chronic_late_payer",), condition=Condition(date(2023, 1, 1), "VOL-T1", 2.0), basket={7: 1.5, 3: 0.25},
    )  # fmt: skip
    world.customers[12] = Customer(
        12, "RT-000002", "Čerpací stanice Lhota", "small_retail", "special_retail", "potential", None, None, "CZ0202",
        "SJC-06", "node/2", "MOL", [Episode("potential", date(2023, 1, 1), "Neodebírá")],
    )  # fmt: skip
    world.actors[5] = Actor(
        5, "novak01", "Jan", "Novák", "rep", "TP-SJC", date(2019, 3, 1), manager_id=4, skills=Skills(conversion=1.35)
    )
    world.duties.append(Duty(5, "rep_area", "SJC-06", date(2023, 1, 1), date(2024, 8, 31)))
    world.stock[(100, 200)] = Stock(120.0)
    world.budgets[(5, date(2023, 1, 1))] = Budget(25_000.0)
    return world


def test_a_world_pickles_hashes_and_reloads_equal() -> None:
    world = _world()
    checkpoint = Checkpoint.of(world)
    assert checkpoint.day == world.day
    assert checkpoint.sha256 == hashlib.sha256(checkpoint.blob).hexdigest()
    assert checkpoint.restore() == world
    assert Checkpoint.of(_world()).sha256 == checkpoint.sha256  # built again, the same bytes


def test_a_changed_world_hashes_otherwise() -> None:
    world = _world()
    before = Checkpoint.of(world).sha256
    world.stock[(100, 200)].on_hand -= 1
    assert Checkpoint.of(world).sha256 != before


def test_the_pickle_does_not_follow_the_hash_seed() -> None:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from test_world import _world; from hartland_sim.world import Checkpoint; "
        "print(Checkpoint.of(_world()).sha256)"
    )
    here = str(Path(__file__).parent)
    hashes = {
        subprocess.run(
            [sys.executable, "-c", code, here],
            env=os.environ | {"PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("1", "2", "3")
    }
    assert hashes == {Checkpoint.of(_world()).sha256}


def test_a_set_in_the_world_is_refused() -> None:
    world = _world()
    world.customers[11].traits = {"a", "b"}  # type: ignore[assignment]
    assert unordered_containers(world) == ["world.customers[11].traits"]
    with pytest.raises(TypeError, match="unordered"):
        Checkpoint.of(world)


def test_a_blob_that_is_not_its_sha256_or_not_a_world_is_refused() -> None:
    checkpoint = Checkpoint.of(_world())
    with pytest.raises(ValueError, match="does not match"):
        Checkpoint(checkpoint.day, checkpoint.sha256, checkpoint.blob + b"\0").restore()
    blob = pickle.dumps({"not": "a world"})
    with pytest.raises(ValueError, match="holds a dict"):
        Checkpoint(checkpoint.day, hashlib.sha256(blob).hexdigest(), blob).restore()


def test_save_keeps_the_warm_up_checkpoint_and_the_last(tmp_path: Path) -> None:
    for day in (date(2023, 12, 30), WARM_UP_END, date(2024, 1, 1), date(2024, 1, 2)):
        Checkpoint.of(_world(day)).save(tmp_path, keep=[WARM_UP_END])
    assert [row.day for row in load_all(tmp_path)] == [WARM_UP_END, date(2024, 1, 2)]
    assert Checkpoint.load(tmp_path).restore() == _world(date(2024, 1, 2))
    assert Checkpoint.load(tmp_path, WARM_UP_END).restore() == _world(WARM_UP_END)
    header, first = (tmp_path / CHECKPOINT_FILE).read_text(encoding="utf-8").splitlines()[:2]
    assert header == "day,sha256,blob"
    assert first.split(",")[2].startswith("\\x")  # bytea's hex text, as COPY reads it


def test_load_without_a_checkpoint_says_so(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        Checkpoint.load(tmp_path)


def test_meta_carries_the_five_keys_and_round_trips(tmp_path: Path) -> None:
    meta = Meta(31102026, "a" * 64, "b" * 64, WARM_UP_END, "c" * 16)
    assert [key for key, _ in meta.rows()] == [
        "master_seed",
        "stories_sha256",
        "world_sha256",
        "last_day_done",
        "build_id",
    ]
    meta.save(tmp_path)
    assert Meta.load(tmp_path) == meta


def test_a_customer_is_active_from_since_until_until() -> None:
    customer = _world().customers[11]
    assert not customer.active_on(date(2021, 5, 2))
    assert customer.active_on(date(2021, 5, 3))
    customer.until = date(2024, 3, 31)
    assert customer.active_on(date(2024, 3, 31))
    assert not customer.active_on(date(2024, 4, 1))
    assert not _world().customers[12].active_on(date(2024, 1, 1))


def test_a_customer_s_phase_on_a_day_is_its_episode_s() -> None:
    customer = _world().customers[11]
    customer.episodes.append(Episode("regular", date(2023, 8, 15), "Pravidelné objednávky"))
    assert customer.phase_on(date(2021, 5, 2)) == "potential"  # before the first episode
    assert customer.phase_on(date(2021, 5, 3)) == "occasional"
    assert customer.phase_on(date(2023, 8, 14)) == "occasional"
    assert customer.phase_on(date(2023, 8, 15)) == "regular"
    assert _world().customers[12].phase_on(date(2030, 1, 1)) == "potential"


def test_episodes_and_names_survive_the_checkpoint() -> None:
    world = _world()
    world.customers[11].episodes.append(Episode("regular", date(2023, 8, 15), "Pravidelné objednávky"))
    restored = Checkpoint.of(world).restore()
    assert restored == world
    assert restored.customers[11].episodes[-1] == Episode("regular", date(2023, 8, 15), "Pravidelné objednávky")
    assert (restored.customers[12].name, restored.actors[5].last_name, restored.actors[5].manager_id) == (
        "Čerpací stanice Lhota",
        "Novák",
        4,
    )


def test_the_duty_on_a_day() -> None:
    world = _world()
    world.duties.append(Duty(6, "rep_area", "SJC-06", date(2024, 9, 1)))
    assert world.duty("rep_area", "SJC-06", date(2024, 8, 31)) == world.duties[0]
    assert world.duty("rep_area", "SJC-06", date(2024, 9, 1)) == world.duties[1]
    assert world.duty("rep_area", "SJC-06", date(2022, 12, 31)) is None
