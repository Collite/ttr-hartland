"""The `just` surface (contracts §7): every generator verb has a recipe, and `just test` exists."""

import shutil
import subprocess
from pathlib import Path

import pytest

from hartland_sim.cli import VERBS

pytestmark = pytest.mark.skipif(shutil.which("just") is None, reason="`just` is not installed")


def recipes(repo_root: Path) -> set[str]:
    result = subprocess.run(["just", "--summary"], cwd=repo_root, capture_output=True, text=True, check=True)
    return set(result.stdout.split())


def test_every_generator_verb_has_a_recipe(repo_root: Path) -> None:
    assert set(VERBS) <= recipes(repo_root)


def test_the_test_recipe_exists(repo_root: Path) -> None:
    assert "test" in recipes(repo_root)


@pytest.mark.parametrize("verb", sorted(VERBS))
def test_each_data_recipe_calls_its_own_verb(verb: str, repo_root: Path) -> None:
    result = subprocess.run(["just", "--show", verb], cwd=repo_root, capture_output=True, text=True, check=True)
    assert f'uv run hartland-sim {verb} "$@"' in result.stdout
