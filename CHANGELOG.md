# Changelog

## 0.0.1 — 2026-05-20

Initial commit. Pure-Zig vector similarity search. Flat (brute-force)
index with L2-squared, inner-product, cosine metrics. Scalar,
single-thread. Pre-1.0 substrate.

- `FlatIndex` with `add`, `addWithId`, `search`, `len`, `deinit`
- Three metrics: `l2_squared`, `inner_product`, `cosine`
- `Hit` struct (id, score) for kNN results
- 14 unit tests covering API + every documented error path
- `examples/search.zig` synthetic 1000-vector kNN demo
- `bench/bench_flat.zig` throughput bench (self-bench only at v0.0.1)
- AGPL-3.0
