#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to package a dirty worktree." >&2
  exit 2
fi

version=$(<VERSION)
tag="v$version"
if ! git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
  echo "Missing release tag $tag." >&2
  exit 2
fi
if [[ "$(git rev-list -n1 "$tag")" != "$(git rev-parse HEAD)" ]]; then
  echo "$tag does not point to HEAD." >&2
  exit 2
fi

mkdir -p dist
archive="dist/BEM-CPP-$version.tar.gz"
git archive --format=tar.gz --prefix="BEM-CPP-$version/" \
  --output="$archive" "$tag"
sha256sum "$archive" > "$archive.sha256"
printf 'Created %s\n' "$archive"
cat "$archive.sha256"
