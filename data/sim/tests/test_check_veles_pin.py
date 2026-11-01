"""scripts/check-veles-pin.sh (review-202 N4; ttr-demo's `just check-veles-pin`): the [veles] pins against the
metadata release olymp deploys on bp-dsk and hartland, and against ttr-server's catalogue at server-libs/v<image>.
A fixture olymp repository and a stub `gh`; no network."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PINS = tomllib.loads((REPO / "toolchain.lock").read_text(encoding="utf-8"))["veles"]
GIT = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}


def _run(
    tmp: Path, *, olymp: dict[str, str] | None, loader: str = PINS["ttr-metadata"]
) -> subprocess.CompletedProcess[str]:
    """`olymp`: cluster → the metadata app's config.json `version` (None: no olymp checkout)."""
    env = {**os.environ, **GIT}
    checkout = tmp / "olymp"
    if olymp is not None:
        for cluster, version in olymp.items():
            app = checkout / "clusters" / cluster / "apps/metadata"
            app.mkdir(parents=True)
            (app / "config.json").write_text(
                json.dumps({"chartRevision": "server-libs/v{version}", "version": version})
            )
        for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "f"]):
            subprocess.run(["git", "-C", str(checkout), *args], env=env, check=True)
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\nimport os, sys\n"
        f'assert "ref=server-libs/v{PINS["image"]}" in sys.argv[2], sys.argv\n'
        'print(os.environ["PIN_TEST_CATALOGUE"])\n',
        encoding="utf-8",
    )
    gh.chmod(0o755)
    catalogue = base64.b64encode(f'tatrman-ttr-metadata = "{loader}"\n'.encode()).decode()
    return subprocess.run(
        ["bash", str(REPO / "scripts/check-veles-pin.sh"), str(checkout), "HEAD"],
        env={**env, "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}", "PIN_TEST_CATALOGUE": catalogue},
        capture_output=True,
        text=True,
    )


def test_the_pins_match_both_clusters_and_the_release(tmp_path: Path) -> None:
    result = _run(tmp_path, olymp={"bp-dsk": PINS["image"], "hartland": PINS["image"]})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok   olymp HEAD clusters/bp-dsk/apps/metadata/config.json" in result.stdout
    assert "ok   olymp HEAD clusters/hartland/apps/metadata/config.json" in result.stdout
    assert f"ok   ttr-server server-libs/v{PINS['image']}" in result.stdout


def test_a_cluster_on_another_release_fails_by_name(tmp_path: Path) -> None:
    result = _run(tmp_path, olymp={"bp-dsk": PINS["image"], "hartland": "0.0.1"})
    assert result.returncode == 1
    assert "FAIL olymp HEAD clusters/hartland/apps/metadata/config.json: metadata '0.0.1'" in result.stderr
    assert "bp-dsk" not in result.stderr


def test_a_release_built_on_another_ttr_metadata_fails(tmp_path: Path) -> None:
    result = _run(tmp_path, olymp={"bp-dsk": PINS["image"], "hartland": PINS["image"]}, loader="0.0.1")
    assert result.returncode == 1
    assert "FAIL ttr-server" in result.stderr


def test_without_olymp_the_release_is_still_checked(tmp_path: Path) -> None:
    result = _run(tmp_path, olymp=None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped: no olymp checkout" in result.stdout
    assert "ok   ttr-server" in result.stdout
