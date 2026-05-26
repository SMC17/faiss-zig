# Changelog

## 0.7.0 — 2026-05-22

IndexIVFPQ: inverted-file index with Product Quantization residual
encoding. The production FAISS workhorse — combines v0.6 IVFFlat
coarse quantization with PQ fine quantization for memory-efficient
ANN search. With this, faiss-zig now covers the four FAISS index
families that matter: Flat + HNSW + IVFFlat + IVFPQ.

- `src/ivf_pq.zig` (~560 LOC): full IVFPQ per Jégou/Douze/Schmid 2011.
  Coarse k-means → nlist centroids; per-subspace k-means over residuals
  → M codebooks of K_s=256 (1 byte each); Asymmetric Distance
  Computation (ADC) via per-query M×K_s LUT at search time.
- Public API: `IndexIVFPQ`, `IVFPQConfig` (nlist, nprobe, pq_m, n_iter,
  seed), `init`, `deinit`, `train`, `add`, `addWithId`, `search`,
  `searchBatch` (multi-threaded), `memoryBytes` (footprint accessor).
- Defaults match FAISS: `nlist=100`, `nprobe=10`, `pq_m=8`, `n_iter=10`.
  Deterministic given `seed: u64`. `pq_m` must evenly divide `dim`.
- L2-squared metric only at v0.7 (cosine + inner-product in v0.7.1+).
- 11 unit tests including: config-validation paths (nprobe>nlist,
  dim%pq_m!=0, pq_m==0), train-required errors, dim-mismatch errors,
  end-to-end train+add+search, EmptyIndex, **recall vs FlatIndex at
  N=2000 D=64 M=8 nlist=16 nprobe=8 top-10 (measured 0.360, threshold
  0.30 with seed-noise margin)**, searchBatch ↔ search agreement,
  memory-footprint sanity bound.
- `bench/bench_ivf_pq.zig` (N=10000 D=128 K=10 Q=100 nlist=100
  nprobe=10 pq_m=8) on Ryzen-class consumer CPU, ReleaseFast:
  - **IVFPQ: 522 μs/query, 1,915 queries/sec, 3.43x faster than Flat**
  - IVFFlat (v0.6): 208 μs/query, 4,801 q/s, 8.61x faster than Flat
  - Flat: 1,792 μs/query, 557 q/s
  - **Memory footprint: 295 KB total (codes + IDs + codebooks +
    coarse centroids) vs 5,000 KB raw f32 storage = 16.94x
    compression**. Per-vector: 8 bytes PQ code + 4 bytes ID = 12
    bytes/vec vs 512 bytes/vec raw f32 = **42.67x per-vector
    compression** before metadata.
- IVFPQ is slower per query than IVFFlat at N=10K because the per-
  probe ADC LUT build (8 × 256 × 16 = 32K sub-distance computes per
  query) is a fixed cost regardless of list size. IVFPQ's win shows
  at larger N (per-vector scan is byte-LUT vs full L2-squared), which
  is exactly the regime where Flat collapses.
- Re-exported as `faiss.IndexIVFPQ` + `faiss.IVFPQConfig`.
- Test count: 33 → 44 (11 IVFPQ).

Honest non-claims for v0.7:
- **PQ trades recall for memory.** On synthetic uniform [0,1]^D random
  data the recall drop vs Flat is much larger (0.36 in our test
  fixture, 0.10 in the N=10K D=128 bench) than the FAISS-published
  numbers on real (clustered) embeddings like SIFT1M (~0.6-0.8). We
  do not over-promise; the README cites the measured-on-uniform value.
- **L2-squared only.** Cosine + inner-product follow in v0.7.1.
- **No Optimal Product Quantization (OPQ rotation matrix).** v0.8+.
  OPQ adds a learned rotation before PQ to maximise codebook quality;
  for axis-correlated data (real embeddings) this matters substantially.
- **Single-thread training.** Both coarse and per-subspace k-means
  are sequential; M=8 subspaces × n_iter=10 × N points × K_s=256
  centroids is the dominant cost. Parallel k-means assignment in v0.8.
- **K-means init is uniform-random sampling**, not k-means++.
- **Concrete numbers for memory math** at D=128, M=8:
  - Per-vector code: 8 bytes
  - Codebook size: 8 × 256 × 16 × 4 = 128 KB (fixed)
  - Coarse centroids: nlist × 128 × 4 (e.g. 100 × 512 = 51 KB)

## 0.6.0 — 2026-05-22

IndexIVFFlat: inverted-file approximate index. Second approximate index
after HNSW; together with FlatIndex this gives faiss-zig the
Flat + HNSW + IVFFlat trio.

- `src/ivf.zig` (~430 LOC): full IVFFlat with Lloyd k-means training,
  inverted-list build + add, top-`nprobe` centroid scan at query time.
- Public API: `IndexIVFFlat`, `IVFConfig`, `init`, `deinit`, `train`,
  `add`, `addWithId`, `search`, `searchBatch` (multi-threaded chunk
  over queries, mirrors `FlatIndex.searchBatch`).
- Defaults match FAISS: `nlist=100`, `nprobe=10`. K-means: 10 Lloyd
  iterations (FAISS uses 25; recall delta is <1% on the typical N/D
  ranges tested). Deterministic given `seed: u64`.
- L2-squared metric only at v0.6 (cosine + inner-product follow in
  v0.7, matching the IndexHNSW v0.3 → v0.4 trajectory).
- 9 unit tests including a recall test against FlatIndex at N=200
  D=32 top-10 with nlist=16 nprobe=8. Threshold 0.7; observed 0.83.
- `bench/bench_ivf.zig` (N=10000 D=128 K=10 Q=100, nlist=100, nprobe=10):
  - IVF: ~244 μs/query, ~4,089 queries/sec
  - Flat: ~1,822 μs/query, ~548 queries/sec
  - **7.45x speedup vs Flat** on uniform [0,1]^128 random data
  - Recall on uniform random data is 0.31 — this is expected for
    structureless data where the top-10 nearest are scattered across
    many cells. Clustered real-world data (embeddings, etc.) recovers
    the ~95% recall that the FAISS rule-of-thumb describes.
- Re-exported as `faiss.IndexIVFFlat` + `faiss.IVFConfig`.
- Test count: 18 → 33 (also picks up HNSW tests via transitive
  test-discovery reference added to `src/root.zig`).

Honest non-claims for v0.6:
- **L2-squared only** — cosine + inner-product follow in v0.7.
- **No PQ compression** — v0.7 path is to combine IVF + PQ into the
  production FAISS workhorse "IVF-PQ".
- **K-means init is uniform-random sampling**, not k-means++. v0.7
  quality bump.
- **Single-thread training**. Parallel k-means assignment in v0.7
  (the assignment step is embarrassingly parallel).
- **Recall on uniformly random data is lower than the FAISS rule-of-
  thumb 95%** because uniformly random data has no cluster structure
  for k-means to find.

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
