# faiss-zig Agent Entry Point

Pure-Zig vector similarity search (Flat, HNSW, IVFFlat, IVFPQ) on Zig 0.16.

## Before you change anything

1. Read `README.md` and `STATUS.md`.
2. Prefer the smallest proof gate below before claiming pass/fail.
3. Do not treat README bench numbers as re-verified unless you re-run them.

## Proof gate

```sh
zig build test
```

Optional (not default agent work):

```sh
zig build bench
# recall / head-to-head benches only with explicit operator intent
```

## Allowed work

- docs/status reconciliation against local evidence
- unit-test and index-algorithm fixes
- narrow public API improvements with tests
- proof-gate / CI maintenance

## Forbidden without explicit operator intent

- publishing releases or changing package identity
- claiming production FAISS parity without fresh matched benches
- committing large private corpora or model dumps
- rewriting recall thresholds without documenting the measurement

## Escalation

Ask before release claims, changing recall thresholds used as gates, or
publishing external benchmark comparisons.

## Hygiene

Org templates: `SMC17/stax-hygiene` (topic target: `stax-hygiene-managed`).
