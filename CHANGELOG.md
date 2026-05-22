# Changelog

## 0.3.0 — 2026-05-21

IndexHNSW: hierarchical navigable small-world approximate index.

- `src/hnsw.zig` (~310 LOC): full HNSW with insert + search, level
  assignment from geometric distribution, M-nearest neighbor connections
  with reciprocal pruning, ef_construction-bounded layer search,
  ef_search-bounded query, multi-level greedy descent.
- L2-squared metric only at v0.3 (other metrics in v0.4).
- Default config: M=16, ef_construction=200, ef_search=100, seed=42.
- 5 unit tests including recall-vs-FlatIndex on N=200 D=32 top-10
  (recall threshold 0.7 — typically observed at ~0.9-1.0).
- Re-exported as `faiss.IndexHNSW` + `faiss.HNSWConfig`.

Approximate-index path unblocks production-scale-out (N >> 10K, where
FlatIndex's O(N*D) per-query cost dominates). HNSW search complexity is
roughly O(M * ef_search * log N) so query latency at large N is
dramatically lower than Flat — concrete head-to-head bench is v0.4.

## 0.2.0 — 2026-05-21

Multi-threaded `searchBatch`. Closes most of the remaining gap to
`facebookresearch/faiss` C++ upstream.

- `FlatIndex.searchBatch(alloc, queries, k, n_threads)`: spawns N
  worker threads (clamped to queries.len), each processes a contiguous
  chunk of queries via the single-thread search path. T=1 falls
  through to a single-thread loop to avoid spawn overhead.
- `bench/bench_multithread.zig`: scaling bench across T ∈ {1, 2, 4, 8}.
- Measured scaling on 4-core/8-thread consumer Ryzen Linux,
  ReleaseFast, N=10000 D=128 Q=100:
    1 thread: 2.30 ms/query, 434 q/s   (1.00x)
    2 threads: 1.29 ms/query, 775 q/s   (1.79x)
    4 threads: 772 us/query, 1295 q/s   (2.98x)
    8 threads: 840 us/query, 1189 q/s   (2.74x, HT contention)
- New gap to faiss-cpu: ~1.5x (was ~4.5x at v0.1, ~45x at v0.0.1)
- Honest non-claim: 4→8 thread regression is the HT + memory-
  bandwidth contention on this 4-core CPU. Real 8-core hardware
  should hold 5-6x scaling.

## 0.1.0 — 2026-05-21

SIMD distance loops + audited head-to-head against `facebookresearch/faiss`
via faiss-cpu Python wrapper.

- `@Vector(8, f32)` paths for l2_squared / inner_product / vectorNorm
  with scalar tail loops for non-multiple-of-8 dimensions
- `faiss.scalar.*` and `faiss.distance.*` public namespaces for
  testing + bypass-wrapper callers
- 3 new property tests asserting SIMD ↔ scalar agreement at varied D
- `bench/bench_simd_vs_scalar.zig` — raw inner-loop throughput
  comparison with median-of-3 trials per path
- `bench/bench_faiss_cpu.py` — faiss-cpu head-to-head bench
- Measured numbers (Ryzen-class consumer Linux, ReleaseFast):
  - SIMD speedup over scalar reduction loop: 4.9x at D=128, 6.1x
    at D=512
  - faiss-cpu (C++, multi-thread, AVX-2): **~4.5x faster than
    faiss-zig v0.1 end-to-end** on N=10000, D=128
- Honest non-claim published as headline ratio in README.md (not
  hidden): the C++ wins by ~4.5x because of multi-thread + cache
  tiling. v0.2 path to close this is named explicitly.

## 0.0.1 — 2026-05-20

Initial commit. Pure-Zig vector similarity search. Flat (brute-force)
index with L2-squared, inner-product, cosine metrics. Scalar,
single-thread. Pre-1.0 substrate.

- FlatIndex with add, addWithId, search, len, deinit
- Three metrics: l2_squared, inner_product, cosine
- Hit struct for kNN results
- 14 unit tests
- AGPL-3.0
