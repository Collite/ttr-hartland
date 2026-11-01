#!/usr/bin/env bash
# Commit the rebuilt generated/ and push it to the branch it was built for — regenerate.yml's push step, run after
# scripts/regenerate-local.sh exited 10 (ttr-demo's scripts/regenerate-push.sh, trimmed for NH M0).
#
#   scripts/regenerate-push.sh <branch>          # from inside the checkout
#
# A push can race another commit on the branch; it is then rejected as non-fast-forward. A rebase would be wrong — an
# artifact built for the old head is stale on the new one — so the script fetches the new head, resets to it and
# regenerates there: nothing stale → exit 0; rebuilt → commit and push again; anything else → that exit code. At most
# REGENERATE_PUSH_ATTEMPTS pushes (default 3), then exit 1. Any other push failure (auth, a hook, the network) exits 1
# at once, not retried.
#
# Env: BOT_NAME, BOT_EMAIL — the commit's identity. GH_TOKEN (optional) — sent as the HTTP auth header of the push and
# the fetch for that one command, never written to .git/config (the checkout persists no credentials).
#
# Refuses (exit 1, nothing committed, nothing reset) when anything outside generated/ is modified: a retry resets the
# tree, so it must never run over someone's edits.
set -euo pipefail

branch="${1:?usage: regenerate-push.sh <branch>}"
: "${BOT_NAME:?BOT_NAME is not set}" "${BOT_EMAIL:?BOT_EMAIL is not set}"
attempts="${REGENERATE_PUSH_ATTEMPTS:-3}"
cd "$(git rev-parse --show-toplevel)"

dirty="$(git status --porcelain -- . ':!generated')"
if [ -n "$dirty" ]; then
  echo "regenerate-push: refusing — files outside generated/ are modified, and a retry would reset them:" >&2
  echo "$dirty" >&2
  exit 1
fi
if [ -z "$(git status --porcelain -- generated)" ]; then
  echo "regenerate-push: nothing to commit in generated/ — run it after regenerate-local.sh exits 10" >&2
  exit 1
fi

# git, with the token as this one command's auth header when there is one.
authed() {
  if [ -n "${GH_TOKEN:-}" ]; then
    local basic
    basic="$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 | tr -d '\n')"
    git -c http.https://github.com/.extraheader="AUTHORIZATION: basic $basic" "$@"
  else
    git "$@"
  fi
}

commit() {
  git add -A -- generated
  git -c user.name="$BOT_NAME" -c user.email="$BOT_EMAIL" commit -q -m "regenerate: generated/ for $(git rev-parse --short HEAD)"
}

commit
attempt=1
while :; do
  echo "regenerate-push: push attempt $attempt of $attempts → $branch"
  set +e
  out="$(authed push origin "HEAD:refs/heads/$branch" 2>&1)"
  code=$?
  set -e
  printf '%s\n' "$out"
  if [ "$code" -eq 0 ]; then
    echo "regenerate-push: pushed $(git rev-parse --short HEAD) to $branch"
    exit 0
  fi
  if ! grep -qE 'non-fast-forward|fetch first' <<<"$out"; then
    echo "regenerate-push: the push failed for another reason than a moved branch — not retried" >&2
    exit 1
  fi
  if [ "$attempt" -ge "$attempts" ]; then
    echo "regenerate-push: the branch kept moving — gave up after $attempts push attempts" >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  echo "regenerate-push: $branch moved during the build — regenerating on its new head"
  authed fetch -q --depth=1 origin "refs/heads/$branch"
  git reset -q --hard FETCH_HEAD
  set +e
  bash scripts/regenerate-local.sh
  code=$?
  set -e
  case "$code" in
    0)  echo "regenerate-push: the new head is current — nothing to push"; exit 0 ;;
    10) commit ;;
    *)  exit "$code" ;;
  esac
done
