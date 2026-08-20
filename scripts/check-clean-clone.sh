#!/usr/bin/env bash
# Fail if the qsr-zig path-dep is missing. See CLEAN_CLONE.md.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -e deps/qsr-zig/src/root.zig ]]; then
  echo "error: deps/qsr-zig missing; zig build test needs it." >&2
  echo "fix: ln -sfn ../../qsr-zig deps/qsr-zig   # or clone into deps/qsr-zig" >&2
  echo "see CLEAN_CLONE.md" >&2
  exit 1
fi
