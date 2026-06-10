#!/usr/bin/env bash
# scripts/checkpoint.sh — delegate to the canonical stax-checkpoint CLI.
# Installed by stax-checkpoint install on 2026-06-03.
set -e
if command -v stax-checkpoint >/dev/null 2>&1; then
    exec stax-checkpoint generate --lane "faiss-zig" "$@"
else
    echo "stax-checkpoint not on PATH — install at ~/.local/bin/stax-checkpoint" >&2
    exit 0  # advisory; never block a commit
fi
