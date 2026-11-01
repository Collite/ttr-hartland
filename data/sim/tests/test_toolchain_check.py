"""scripts/toolchain-check.sh (review-202 N2): the model verbs refuse a ttr-core checkout that is not at
toolchain.lock's [ttr-core].ref. A fixture repository stands in for ttr-core; no network."""

from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REF = tomllib.loads((REPO / "toolchain.lock").read_text(encoding="utf-8"))["ttr-core"]["ref"]
ENV = {
    **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.test",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.test",
}


def _check(toolchain: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPO / "scripts/toolchain-check.sh")],
        env={**ENV, "TTR_CORE": str(toolchain)},
        capture_output=True,
        text=True,
    )


def _commit(repo: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", message], env=ENV, check=True)


def test_the_pinned_ref_passes_and_any_other_commit_is_refused(tmp_path: Path) -> None:
    core = tmp_path / "ttr-core"
    subprocess.run(["git", "init", "-q", str(core)], env=ENV, check=True)
    _commit(core, "the release")
    subprocess.run(["git", "-C", str(core), "tag", REF], env=ENV, check=True)
    assert _check(core).returncode == 0

    _commit(core, "after the release")
    result = _check(core)
    assert result.returncode == 2
    assert f"toolchain: {core} is at '" in result.stderr
    assert f"toolchain.lock pins {REF} (" in result.stderr


def test_a_directory_that_is_no_checkout_is_refused(tmp_path: Path) -> None:
    result = _check(tmp_path)
    assert result.returncode == 2
    assert f"toolchain: {tmp_path} is at ''" in result.stderr
