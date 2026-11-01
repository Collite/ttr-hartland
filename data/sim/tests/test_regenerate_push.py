"""scripts/regenerate-push.sh (M0 S0.1 T2; review-202 R1): a push that loses a race rebuilds on the new head without
the write token in its environment, and a branch deleted meanwhile is not re-created. A bare repository is `origin`;
no network."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# The rebuild regenerate-push.sh runs on a retry: records the GH_TOKEN it sees, rebuilds generated/ for the head.
STUB = """#!/usr/bin/env bash
echo "${GH_TOKEN-unset}" >> "$SEEN"
git rev-parse HEAD > generated/stamp
exit 10
"""


def _env(tmp: Path) -> dict[str, str]:
    return {
        **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.test",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.test",
        "SEEN": str(tmp / "seen"),
    }


def _git(cwd: Path, *args: str, env: dict[str, str]) -> str:
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def _setup(tmp: Path) -> tuple[Path, Path, dict[str, str]]:
    """origin (bare) with `feat`; `runner` (the job's checkout) and `other` (someone else) cloned from it."""
    env = _env(tmp)
    origin = tmp / "origin.git"
    _git(tmp, "init", "-q", "--bare", str(origin), env=env)
    seed = tmp / "seed"
    (seed / "scripts").mkdir(parents=True)
    (seed / "generated").mkdir()
    shutil.copy(REPO / "scripts/regenerate-push.sh", seed / "scripts/regenerate-push.sh")
    (seed / "scripts/regenerate-local.sh").write_text(STUB, encoding="utf-8")
    (seed / "generated/stamp").write_text("old\n", encoding="utf-8")
    _git(seed, "init", "-q", env=env)
    _git(seed, "add", ".", env=env)
    _git(seed, "commit", "-q", "-m", "seed", env=env)
    _git(seed, "push", "-q", str(origin), "HEAD:refs/heads/feat", env=env)
    runner, other = tmp / "runner", tmp / "other"
    for clone in (runner, other):
        _git(tmp, "clone", "-q", "--branch", "feat", str(origin), str(clone), env=env)
    return runner, other, env


def _push(runner: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    (runner / "generated/stamp").write_text("rebuilt\n", encoding="utf-8")  # the first build, before the token
    return subprocess.run(
        ["bash", "scripts/regenerate-push.sh", "feat"],
        cwd=runner,
        env={**env, "GH_TOKEN": "x", "BOT_NAME": "bot", "BOT_EMAIL": "bot@example.test"},
        capture_output=True,
        text=True,
    )


def test_the_retry_rebuilds_on_the_new_head_without_the_token(tmp_path: Path) -> None:
    runner, other, env = _setup(tmp_path)
    (other / "model.ttrm").write_text("package hartland\n", encoding="utf-8")
    _git(other, "add", "model.ttrm", env=env)
    _git(other, "commit", "-q", "-m", "someone else's change", env=env)
    _git(other, "push", "-q", "origin", "HEAD:refs/heads/feat", env=env)
    moved = _git(other, "rev-parse", "HEAD", env=env)

    result = _push(runner, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "regenerating on its new head" in result.stdout
    assert (tmp_path / "seen").read_text(encoding="utf-8") == "unset\n"
    origin = tmp_path / "origin.git"
    assert _git(origin, "rev-parse", "feat^", env=env) == moved
    assert _git(origin, "show", "feat:generated/stamp", env=env) == moved


def test_a_deleted_branch_is_not_re_created(tmp_path: Path) -> None:
    runner, other, env = _setup(tmp_path)
    _git(other, "push", "-q", "origin", "--delete", "feat", env=env)

    result = _push(runner, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "the branch is gone — nothing pushed" in result.stdout
    assert _git(tmp_path / "origin.git", "branch", "--list", "feat", env=env) == ""
    assert not (tmp_path / "seen").exists()
