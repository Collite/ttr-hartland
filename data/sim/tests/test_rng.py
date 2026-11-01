"""The named random streams (contracts §2; G1 S1.1 T1): one seed per (stream, key), the catalogue, and the guard that
no draw is made outside `rng.py`."""

from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

import hartland_sim
from hartland_sim import rng as rng_module
from hartland_sim.config import SCHEMA_DIR, WorldConfig
from hartland_sim.rng import master_seed, rng, using_master_seed
from hartland_sim.streams import STREAMS

PACKAGE = Path(hartland_sim.__file__).resolve().parent
# Contracts §2's catalogue, with the two review-196 ⚑C added (`skills` S1.1 T4, `promo` S5.3 T4) and `org` (S1.1 T4).
CONTRACT_STREAMS = {
    "demand",
    "basket",
    "acquire",
    "churn",
    "visit",
    "note",
    "observe",
    "sighting",
    "prompt_outcome",
    "pos",
    "pricing_agree",
    "payment",
    "return",
    "complaint",
    "supply",
    "route",
    "event",
    "shop",
    "weather",
    "letter",
    "scan",
    "skills",
    "promo",
    "org",
}


def _first(stream: str, *key: object) -> float:
    return float(rng(stream, *key).random())


def test_the_same_stream_and_key_give_the_same_draw_and_another_key_another() -> None:
    first = _first("demand", 42, date(2024, 1, 1))
    assert _first("demand", 42, date(2024, 1, 1)) == first
    assert _first("demand", 43, date(2024, 1, 1)) != first
    assert _first("demand", 42, date(2024, 1, 2)) != first
    assert _first("basket", 42, date(2024, 1, 1)) != first


def test_a_draw_added_to_one_stream_leaves_every_other_stream_alone() -> None:
    before = rng("basket", 42, date(2024, 1, 1)).random(5)
    demand = rng("demand", 42, date(2024, 1, 1))
    demand.random(1000)  # a new draw site on `demand`
    after = rng("basket", 42, date(2024, 1, 1)).random(5)
    assert np.array_equal(before, after)


def test_key_parts_are_canonical_before_they_seed() -> None:
    """review-188: under numpy 2 `repr((np.int64(12),))` ≠ `repr((12,))` — a key read from an array is the same key."""
    assert _first("demand", np.int64(42)) == _first("demand", 42)
    assert _first("demand", np.int32(42), date(2024, 1, 1)) == _first("demand", 42, date(2024, 1, 1))
    assert _first("demand", np.str_("CZ0201")) == _first("demand", "CZ0201")
    # a tuple's parts stay apart: (12, 3) is not (1, 23), nor "12" the integer 12
    assert _first("demand", 12, 3) != _first("demand", 1, 23)
    assert _first("demand", "12") != _first("demand", 12)
    assert _first("demand", True) != _first("demand", 1)


@pytest.mark.parametrize("part", [1.5, {1, 2}, object(), [1, 2], None])
def test_a_key_part_that_is_no_text_integer_or_date_is_refused(part: object) -> None:
    with pytest.raises(TypeError, match="key part"):
        rng("demand", part)


def test_a_stream_off_the_catalogue_is_refused() -> None:
    with pytest.raises(ValueError, match="no stream 'demmand'"):
        rng("demmand", 1)


def test_the_master_seed_is_world_yaml_s_and_another_seed_draws_another_world() -> None:
    assert master_seed() == WorldConfig.load().master_seed
    first = _first("demand", 42)
    with using_master_seed(master_seed() + 1):
        assert _first("demand", 42) != first
    assert _first("demand", 42) == first


def test_the_catalogue_is_the_contract_s() -> None:
    assert set(STREAMS) == CONTRACT_STREAMS


def test_a_story_s_seed_stream_is_in_the_catalogue() -> None:
    schema = json.loads((SCHEMA_DIR / "story.schema.json").read_text(encoding="utf-8"))
    assert set(schema["properties"]["seed_stream"]["enum"]) <= set(STREAMS)


def _modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _random_reaches(source: str) -> list[str]:
    """Where `source` reaches a random generator but through `rng`: `random`, `numpy.random` however imported, the
    attribute `np.random`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "random" or alias.name.startswith("numpy.random"):
                    found.append(f"{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] == "random" or module.startswith("numpy.random"):
                found.append(f"{node.lineno} from {module}")
            elif module == "numpy" and any(alias.name == "random" for alias in node.names):
                found.append(f"{node.lineno} from numpy import random")
        elif isinstance(node, ast.Attribute) and node.attr == "random" and _is_numpy(node.value):
            found.append(f"{node.lineno} np.random")
    return found


def test_no_module_but_rng_py_makes_a_random_generator() -> None:
    offenders = [
        f"{path.name}:{where}"
        for path in _modules()
        if path != Path(rng_module.__file__).resolve()
        for where in _random_reaches(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


@pytest.mark.parametrize(
    "source",
    [
        "from numpy import random\nrandom.default_rng()",
        "import numpy.random as r\nr.default_rng()",
        "from numpy import array, random as npr\nnpr.default_rng()",
        "import numpy.random\n",
        "from numpy.random import default_rng\n",
        "import random\n",
        "import numpy as np\nnp.random.default_rng()",
    ],
)
def test_the_guard_sees_every_way_to_numpy_s_random(source: str) -> None:
    assert _random_reaches(source)


def _is_numpy(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id in ("np", "numpy")


def test_every_rng_call_in_the_package_names_a_catalogued_stream_by_its_literal() -> None:
    problems = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name != "rng":
                continue
            first = node.args[0] if node.args else None
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                problems.append(f"{path.name}:{node.lineno}: the stream is not a literal")
            elif first.value not in STREAMS:
                problems.append(f"{path.name}:{node.lineno}: {first.value!r} is not catalogued")
    assert problems == []
