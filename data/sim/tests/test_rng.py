"""The named random streams (contracts §2; G1 S1.1 T1): one seed per (stream, key), the catalogue, and the guard that
no draw is made outside `rng.py`; the offline diagnostics' frozen generator (NH-D21-A) and the guard that keeps the
rest of the package from it."""

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
from hartland_sim.rng import frozen_diagnostic_rng, master_seed, pairs, rng, uniforms, using_master_seed
from hartland_sim.streams import STREAMS

PACKAGE = Path(hartland_sim.__file__).resolve().parent
# Contracts §2's catalogue, with the two review-196 ⚑C added (`skills` S1.1 T4, `promo` S5.3 T4), `org` (S1.1 T4), and
# the two G1 S1.2's list names (`elasticity` T4, `breadth` T6), and G1 S1.4's (`order` T1, `plan` T3).
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
    "elasticity",
    "breadth",
    "order",
    "plan",
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


def test_a_vectorised_draw_is_each_id_s_own() -> None:
    ids = np.array([-7, 3, 42, 2**62, -(2**63)], dtype=np.int64)
    found = uniforms("demand", date(2024, 1, 1), ids=ids)
    assert ((found >= 0) & (found < 1)).all()
    assert np.array_equal(uniforms("demand", date(2024, 1, 1), ids=ids), found)
    alone = [float(uniforms("demand", date(2024, 1, 1), ids=[i])[0]) for i in ids]
    assert alone == found.tolist()  # an id's draw does not depend on who else is drawn
    assert np.array_equal(uniforms("demand", date(2024, 1, 1), ids=ids[::-1]), found[::-1])
    assert not np.array_equal(uniforms("demand", date(2024, 1, 2), ids=ids), found)
    assert not np.array_equal(uniforms("basket", date(2024, 1, 1), ids=ids), found)
    with using_master_seed(master_seed() + 1):
        assert not np.array_equal(uniforms("demand", date(2024, 1, 1), ids=ids), found)


def test_vectorised_draws_are_uniform_and_pairs_are_distinct() -> None:
    draws = uniforms("demand", 1, ids=np.arange(200_000))
    assert abs(draws.mean() - 0.5) < 0.005 and abs(np.median(draws) - 0.5) < 0.005
    assert np.histogram(draws, bins=10, range=(0, 1))[0].min() > 19_000
    grid = pairs(np.arange(300)[:, None], np.arange(300)[None, :])
    assert grid.shape == (300, 300) and len(np.unique(grid)) == 300 * 300
    assert grid[3, 5] != grid[5, 3]


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


def test_every_draw_call_in_the_package_names_a_catalogued_stream_by_its_literal() -> None:
    problems = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name not in ("rng", "uniforms"):
                continue
            first = node.args[0] if node.args else None
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                problems.append(f"{path.name}:{node.lineno}: the stream is not a literal")
            elif first.value not in STREAMS:
                problems.append(f"{path.name}:{node.lineno}: {first.value!r} is not catalogued")
    assert problems == []


# ── the offline diagnostics' frozen generator (contracts §2, NH-D21-A) ──────────────────────────────────────────────
FROZEN = 20261113  # G2 S2.4's S5 prototype seed (`checks.s5_pooled.SEED`)
DIAGNOSTICS = {"checks/s5_pooled.py"}  # the frozen generator's callers, relative to the package


def test_a_frozen_diagnostic_draws_numpy_s_own_from_its_seed() -> None:
    """`default_rng(seed)`'s uniforms then exponentials, in that order, under the locked NumPy — the first pinned."""
    found, expected = frozen_diagnostic_rng(FROZEN), np.random.default_rng(FROZEN)
    uniform, exponential = found.random(1000), found.exponential(size=(200, 37))
    assert np.array_equal(uniform, expected.random(1000)) and uniform.dtype == np.float64
    assert np.array_equal(exponential, expected.exponential(size=(200, 37))) and exponential.shape == (200, 37)
    assert uniform[:3].tolist() == [0.18400199779628634, 0.8376162964490108, 0.21073844371423278]
    first = frozen_diagnostic_rng(FROZEN)
    first.random(3)
    assert first.exponential(size=3).tolist() == [0.17598387883556, 0.8335597343932627, 0.9120051180479031]
    for seed in (-1, 1.0, True, "1"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            frozen_diagnostic_rng(seed)  # type: ignore[arg-type]


def test_each_frozen_diagnostic_call_starts_afresh() -> None:
    used = frozen_diagnostic_rng(FROZEN)
    head = used.random(5)
    used.exponential(size=100)
    assert np.array_equal(frozen_diagnostic_rng(FROZEN).random(5), head)
    assert frozen_diagnostic_rng(FROZEN) is not frozen_diagnostic_rng(FROZEN)


def test_a_frozen_diagnostic_and_the_world_s_streams_leave_each_other_alone() -> None:
    """Another master seed moves the streams, never the diagnostic; drawing from the diagnostic moves no stream."""
    with using_master_seed(1):
        one = frozen_diagnostic_rng(FROZEN).random(5), rng("pos", "term", 7).random(3)
    with using_master_seed(2):
        two = frozen_diagnostic_rng(FROZEN).random(5), rng("pos", "term", 7).random(3)
    assert np.array_equal(one[0], two[0]) and not np.array_equal(one[1], two[1])
    ids = np.arange(50, dtype=np.int64)
    before = rng("pos", "term", 7).random(3), uniforms("pos", "place", "PHA", date(2025, 3, 1), ids=ids)
    frozen_diagnostic_rng(FROZEN).exponential(size=10_000)
    after = rng("pos", "term", 7).random(3), uniforms("pos", "place", "PHA", date(2025, 3, 1), ids=ids)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


def _frozen_reaches(source: str) -> list[str]:
    """Where `source` names the frozen generator: imported, called or read as an attribute."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and any(a.name == "frozen_diagnostic_rng" for a in node.names):
            found.append(f"{node.lineno} import")
        elif isinstance(node, ast.Name) and node.id == "frozen_diagnostic_rng":
            found.append(f"{node.lineno} name")
        elif isinstance(node, ast.Attribute) and node.attr == "frozen_diagnostic_rng":
            found.append(f"{node.lineno} attribute")
    return found


def test_no_generator_code_reaches_the_frozen_diagnostic_rng() -> None:
    """Only the offline diagnostics (`DIAGNOSTICS`) call it: the generator, setup, loop and world code never."""
    own = Path(rng_module.__file__).resolve()
    offenders = [
        f"{path.relative_to(PACKAGE)}:{where}"
        for path in _modules()
        if path != own and path.relative_to(PACKAGE).as_posix() not in DIAGNOSTICS
        for where in _frozen_reaches(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
    for diagnostic in DIAGNOSTICS:
        assert _frozen_reaches((PACKAGE / diagnostic).read_text(encoding="utf-8"))
    for source in (
        "from hartland_sim.rng import frozen_diagnostic_rng\nfrozen_diagnostic_rng(1)",
        "from hartland_sim import rng\nrng.frozen_diagnostic_rng(1)",
        "import hartland_sim.rng as r\nf = r.frozen_diagnostic_rng",
        "from hartland_sim.rng import frozen_diagnostic_rng as quiet\n",
    ):
        assert _frozen_reaches(source), source
