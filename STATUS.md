# STATUS

**Version:** v0.6
**Last update:** 2026-05-22

## What landed in v0.6 (on top of v0.5)

- `src/ivf.zig`: complete `IndexIVFFlat` implementation (~430 LOC).
  Inverted-file approximate index. Lloyd k-means clusters the training
  set into `nlist` centroids; at query time we rank all `nlist`
  centroids by distance, then scan only the top `nprobe` lists.
- Public API mirrors FlatIndex / IndexHNSW: `init`, `deinit`, `train`,
  `add`, `addWithId`, `search`, `searchBatch` (multi-threaded chunk
  over queries).
- Defaults match FAISS: `nlist=100`, `nprobe=10`, `n_iter=10`,
  `seed=42`. Deterministic given the seed.
- L2-squared metric only at v0.6. Cosine + inner-product follow in
  v0.7 (matches the HNSW v0.3 → v0.4 trajectory).
- 9 unit tests including a recall test against FlatIndex at
  N=200 D=32 top-10 (nlist=16 nprobe=8): observed 0.83, threshold 0.7.
- `bench/bench_ivf.zig` at N=10000 D=128 K=10 Q=100 nlist=100 nprobe=10
  on this Ryzen-class consumer CPU, ReleaseFast:
  - IVF: 244 μs/query, 4,089 queries/sec
  - Flat: 1,822 μs/query, 548 queries/sec
  - **7.45x speedup**
  - Recall 0.31 on uniformly random data (no cluster structure to find).
    On clustered real-world embeddings the FAISS rule-of-thumb ~95%
    recall is expected; the unit test on small N + half-probe-rate
    measures the algorithm itself works (0.83 observed).
- Re-exported as `faiss.IndexIVFFlat` + `faiss.IVFConfig`.
- Test discovery: added a `test { _ = hnsw; _ = ivf; }` block in
  `src/root.zig` so HNSW + IVF tests are picked up by `zig build test`.
  Pre-v0.6 only the 18 root-file tests were running; v0.6 surfaces the
  full 33 (18 root + 5 HNSW + 9 IVF + 1 transitive-ref block).

## Honest non-claims for v0.6

- **L2-squared only** for IVF + HNSW. Cosine / inner-product follow.
- **No PQ compression.** v0.7 path is IVF + PQ → IVF-PQ.
- **K-means init is uniform-random sampling**, not k-means++.
- **Single-thread training.** Parallel k-means assignment in v0.7.
- **Recall on uniformly random data is lower** than the FAISS rule-
  of-thumb 95%. That rule assumes data with cluster structure that
  k-means can find. Uniformly random data spreads the top-10 neighbors
  across many cells.

## What landed in v0.3 (on top of v0.2)

- `src/hnsw.zig`: complete IndexHNSW implementation (~310 LOC).
  Hierarchical navigable small-world approximate index. Per-node
  level drawn from geometric distribution, M-nearest neighbor
  connections at each level, reciprocal edges with M-prune,
  ef_construction-bounded layer search during insert, ef_search-bounded
  query, multi-level greedy descent.
- Default config: M=16, ef_construction=200, ef_search=100. Standard
  HNSW parameters (matches faiss-cpu defaults).
- L2-squared metric only at v0.3. Cosine + inner-product paths can
  reuse the framework in v0.4.
- 5 unit tests including a recall test against FlatIndex at N=200
  D=32 top-10. Threshold set to 0.7; observed recall is typically
  0.9+ on this size.
- Re-exported as `faiss.IndexHNSW` + `faiss.HNSWConfig`.

## What landed in v0.2 (on top of v0.1)

- `FlatIndex.searchBatch(alloc, queries, k, n_threads)` — multi-threaded
  batch search via std.Thread. Each worker thread processes a
  contiguous chunk of queries through the single-thread search path
  with no shared mutable state during distance computation.
- `bench/bench_multithread.zig` — scaling measurement at T ∈ {1,2,4,8}
- Measured ~2.98x scaling at T=4 on 4-core/8-thread Ryzen consumer CPU
- Gap to faiss-cpu closed from ~4.5x (v0.1 single-thread) to ~1.5x
  (v0.2 4-thread)
- T=1 falls through to single-thread to avoid spawn overhead

## What landed in v0.1 (on top of v0.0.1)

- SIMD distance loops via `@Vector(8, f32)` for L2-squared,
  inner-product, and vectorNorm with scalar tail loops for
  non-multiple-of-8 dimensions
- `faiss.scalar.*` namespace exporting the scalar reference
  implementations (kept for testing + benches)
- `faiss.distance.*` namespace exporting the SIMD path for callers
  bypassing the Index wrapper
- 3 new property tests: SIMD ↔ scalar agreement at
  D ∈ {8, 16, 64, 100, 128, 129, 256, 1024, 1025} for each metric
  (l2_squared / inner_product / vectorNorm)
- `bench/bench_simd_vs_scalar.zig` — raw inner-loop throughput
  at D ∈ {16, 32, 64, 96, 128, 192, 256, 384, 512, 1024} with
  median-of-3 trials per path
- `bench/bench_faiss_cpu.py` — Python wrapper bench using `faiss-cpu`
  for fair head-to-head with `facebookresearch/faiss`
- Measured SIMD vs scalar speedup: 2.3x at D=16, 4.9x at D=128,
  6.1x at D=512
- Measured faiss-cpu head-to-head: faiss-cpu **~4.5x faster**
  end-to-end on N=10000, D=128, Q=100 L2 (honest published loss)
- Total test count: 14 → 17 (the 3 new SIMD-agreement tests)

## What landed in v0.0.1 (kept)

- FlatIndex with three metrics (L2-squared, inner-product, cosine)
- Hit struct (id, score), Error set, public API
- 14 unit tests covering API + every documented error path
- Search CLI demo
- Throughput bench
- AGPL-3.0

## What is open for v0.2

1. **Multi-threading** (highest-leverage; expected 4-8x close of
   the faiss-cpu gap on consumer 8-core)
2. **Cache-aware tiling** (~1.5x at D=128, modest)
3. **`IndexHNSW`** — first approximate index
4. **`IndexIVFFlat`** — second approximate index
5. **Vector serialisation format** (binary .bin + .fvecs reader)
6. **Multi-platform CI** (macOS, Apple Silicon NEON, Windows)

## Honest non-claims (still active for v0.1)

- **No multi-threading** at v0.1. The ~4.5x faiss-cpu gap is
  partially attributable.
- **No cache tiling**. Working set blasts through L2/L3 at large N.
- **No approximate index**. Flat only.
- **No GPU**. Not near-term.
- **Single-platform tested** (x86_64-linux Ryzen-class). NEON +
  Apple Silicon untested.
- The SIMD speedup ratios are CPU + compiler + -O specific. Numbers
  may shift on different hardware.
- The Python bench wraps the same C++ binary that
  `facebookresearch/faiss` ships; it is a fair head-to-head, but
  pure-C++ benches may show slightly different numbers due to
  Python overhead per call (small but nonzero).
