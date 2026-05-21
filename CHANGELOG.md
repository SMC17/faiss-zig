# Changelog

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
