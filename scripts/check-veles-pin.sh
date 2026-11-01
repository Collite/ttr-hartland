#!/usr/bin/env bash
# The bump-time check of toolchain.lock's [veles] pins (ttr-demo's `just check-veles-pin`, carried over) — for whoever
# moves them, not CI: olymp is private. Two promises:
#   - the metadata release olymp deploys on the clusters that serve this model — bp-dsk (M5's staging) and hartland
#     (SW) — is [veles].image: each app's config.json `version` (olymp D43);
#   - ttr-server's gradle/libs.versions.toml at server-libs/v<image> names [veles].ttr-metadata.
# Exit 1 on a mismatch, both values printed. Needs an authenticated `gh`; the olymp half is skipped when there is no
# olymp checkout.
#
#   scripts/check-veles-pin.sh                           # olymp beside this checkout, its origin/master (fetched)
#   scripts/check-veles-pin.sh <olymp-checkout> [<rev>]  # elsewhere; HEAD for a proposed wave before its merge
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
olymp="${1:-$root/../olymp}"
revision="${2:-origin/master}"
pins="$(python3 "$root/scripts/toolchain-lock.py" "$root/toolchain.lock")"   # its own line: a failed read stops here
while IFS='=' read -r key value; do export "$key=$value"; done <<<"$pins"

rc=0
if git -C "$olymp" rev-parse --git-dir >/dev/null 2>&1; then
  if [ "$revision" = origin/master ]; then
    git -C "$olymp" fetch -q origin 2>/dev/null \
      || echo "check-veles-pin: could not fetch $olymp — reading its last-fetched origin/master" >&2
  fi
  for cluster in bp-dsk hartland; do
    app="clusters/$cluster/apps/metadata/config.json"
    deployed="$( (git -C "$olymp" show "$revision:$app" 2>/dev/null || true) \
      | python3 -c 'import json,sys; t=sys.stdin.read(); print(json.loads(t).get("version", "") if t.strip() else "")')"
    if [ "$deployed" = "$VELES_IMAGE" ]; then
      echo "ok   olymp $revision $app: metadata $deployed = [veles].image"
    else
      echo "FAIL olymp $revision $app: metadata '$deployed' ≠ [veles].image '$VELES_IMAGE'" >&2
      rc=1
    fi
  done
else
  echo "skipped: no olymp checkout at $olymp"
fi

toml="$(gh api "repos/Collite/ttr-server/contents/gradle/libs.versions.toml?ref=server-libs/v$VELES_IMAGE" --jq .content | base64 -d)"
built="$(printf '%s\n' "$toml" | sed -n 's/^tatrman-ttr-metadata *= *"\([^"]*\)".*/\1/p' | head -1)"
if [ "$built" = "$VELES_TTR_METADATA" ]; then
  echo "ok   ttr-server server-libs/v$VELES_IMAGE: tatrman-ttr-metadata $built = [veles].ttr-metadata"
else
  echo "FAIL ttr-server server-libs/v$VELES_IMAGE: tatrman-ttr-metadata '$built' ≠ [veles].ttr-metadata '$VELES_TTR_METADATA'" >&2
  rc=1
fi
exit "$rc"
