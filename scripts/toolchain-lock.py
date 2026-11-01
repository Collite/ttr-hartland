#!/usr/bin/env python3
"""Print toolchain.lock's pins as env lines (KEY=value) — the one reader of the lock that the workflows' toolchain
action and the justfile's lock-reading recipes share (ttr-demo's scripts/toolchain-lock.py, trimmed: this repo has no
[ttrp] recipes and no [automation-preview]).

    python3 scripts/toolchain-lock.py [<lock>]      # default: toolchain.lock beside scripts/

All or nothing, and every value shape-checked: the lines are appended to $GITHUB_ENV and `export`ed by shell callers,
so a value that is not a plain tag or version — a `$(…)`, a space, an escaped newline — is refused with the key named,
never printed. An empty TTR_CORE_REF in particular would check out ttr-core's default branch: the unpinned toolchain
the lock exists to prevent.
"""

import re
import sys

# Run by the system python3 (the workflows' toolchain action, the justfile), not uv's 3.13 — so the check stays.
if sys.version_info < (3, 11):  # noqa: UP036
    sys.exit(f"toolchain-lock.py needs Python >= 3.11 (tomllib); python3 is {sys.version.split()[0]}")

import tomllib  # noqa: E402 — after the version check, which is the point
from pathlib import Path  # noqa: E402

VERSION = r"[0-9]+\.[0-9]+\.[0-9]+"

# env name → (table, key, the shape the value must have — a regex, or bool for a TOML boolean — and how to say it)
PINS: dict[str, tuple[str, str, str | type, str]] = {
    "TTR_CORE_REF": ("ttr-core", "ref", rf"grammar/v{VERSION}", "grammar/vX.Y.Z"),
    "VELES_IMAGE": ("veles", "image", VERSION, "X.Y.Z"),
    "VELES_TTR_METADATA": ("veles", "ttr-metadata", VERSION, "X.Y.Z"),
    "STRICT_LOAD": ("veles", "strict-load", bool, "true or false (a TOML boolean, unquoted)"),
}


def pins(path: Path) -> list[str]:
    """The env lines, or SystemExit naming the first bad key."""
    try:
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        sys.exit(f"{path}: {e}")
    lines = []
    for env, (table, key, shape, said) in PINS.items():
        section = lock.get(table)
        value = section.get(key) if isinstance(section, dict) else None
        if value is None:
            sys.exit(f"{path}: [{table}].{key} is missing")
        if shape is bool:
            if not isinstance(value, bool):
                sys.exit(f"{path}: [{table}].{key} must be {said}, not {value!r}")
            value = "true" if value else "false"
        elif not isinstance(value, str) or not isinstance(shape, str) or not re.fullmatch(shape, value):
            sys.exit(f"{path}: [{table}].{key} must be {said}, not {value!r}")
        lines.append(f"{env}={value}")
    return lines


def main(argv: list[str]) -> None:
    path = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "toolchain.lock"
    print("\n".join(pins(path)))


if __name__ == "__main__":
    main(sys.argv)
