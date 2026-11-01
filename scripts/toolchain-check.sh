#!/usr/bin/env bash
# The model verbs' toolchain guard: the ttr-core checkout they borrow — $TTR_CORE, else the sibling ../tatrman — must be
# at toolchain.lock's [ttr-core].ref, or a suite, the drift check or a rebuilt generated/ reports that toolchain rather
# than the model. Silent, exit 0, when it is; exit 2 naming the checkout, its HEAD and the lock's ref when it is not (or
# is no git checkout). Every TS model verb runs it first (the justfile's `_toolchain`), and so does regenerate-local.sh.
#
#   scripts/toolchain-check.sh [<checkout>]          # default: the checkout this script is in
set -euo pipefail

checkout="$(cd "${1:-$(dirname "$0")/..}" && pwd)"
pins="$(python3 "$checkout/scripts/toolchain-lock.py" "$checkout/toolchain.lock")"   # its own line: a failed read stops
while IFS='=' read -r key value; do export "$key=$value"; done <<<"$pins"

toolchain="${TTR_CORE:-$checkout/../tatrman}"
have="$(git -C "$toolchain" rev-parse HEAD 2>/dev/null || true)"
want="$(git -C "$toolchain" rev-parse --verify --quiet "$TTR_CORE_REF^{commit}" 2>/dev/null || true)"
if [ -z "$have" ] || [ "$have" != "$want" ]; then
  echo "toolchain: $toolchain is at '${have:0:12}', toolchain.lock pins $TTR_CORE_REF${want:+ (${want:0:12})} — check that out and build it first (or point TTR_CORE at one that is)" >&2
  exit 2
fi
