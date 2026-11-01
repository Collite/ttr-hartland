#!/usr/bin/env bash
# The regenerate job on any checkout (ttr-demo's scripts/regenerate-local.sh, trimmed for NH M0). regenerate.yml runs
# this script as-is, so a local run IS the workflow's dry run, minus the commit and the push.
#
#   scripts/regenerate-local.sh [<checkout>]          # default: the checkout this script is in
#
# Rebuilds generated/ with the toolchain toolchain.lock pins — resolved-packages.json now; the lexicon archive joins at
# M3 S3.1 (`just build-lexicon`) — then:
#   exit 0   nothing stale: generated/ already matched; nothing changed. The loop guard: the workflow pushes nothing.
#   exit 10  stale: rebuilt and left in the tree for the caller to commit
#   exit 1   the build failed, or it changed something under model/ (the job never commits authored model/**)
#   exit 2   the toolchain at $TTR_CORE (default <checkout>/../tatrman) is not at toolchain.lock's [ttr-core].ref
set -euo pipefail

checkout="$(cd "${1:-$(dirname "$0")/..}" && pwd)"
cd "$checkout"

pins="$(python3 scripts/toolchain-lock.py)"   # its own line, so a failed read stops the script
while IFS='=' read -r key value; do export "$key=$value"; done <<<"$pins"
echo "toolchain.lock: ttr-core $TTR_CORE_REF · metadata ttr-metadata $VELES_TTR_METADATA"

toolchain="${TTR_CORE:-$checkout/../tatrman}"
have="$(git -C "$toolchain" rev-parse HEAD 2>/dev/null || true)"
want="$(git -C "$toolchain" rev-parse --verify --quiet "$TTR_CORE_REF^{commit}" 2>/dev/null || true)"
if [ -z "$have" ] || [ "$have" != "$want" ]; then
  echo "regenerate: $toolchain is at '${have:0:12}', toolchain.lock pins $TTR_CORE_REF${want:+ (${want:0:12})} — check that out and build it first" >&2
  exit 2
fi

# Content, not `git status`: a file already modified before the build and modified again by it reads the same there.
model_hash() { find model -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum; }
model_before="$(model_hash)"

just resolve-packages || { echo "regenerate: resolve-packages failed" >&2; exit 1; }

if [ "$(model_hash)" != "$model_before" ]; then
  echo "regenerate: the build changed files under model/ — refusing (the job never commits authored model/**):" >&2
  git status --porcelain -- model >&2
  exit 1
fi

stale="$(git status --porcelain -- generated)"
if [ -z "$stale" ]; then
  echo "regenerate: nothing stale — generated/ matches the model"
  exit 0
fi
echo "$stale"
echo "regenerate: stale — rebuilt, left in the tree to commit"
exit 10
