# STATUS

**Version:** v0.1
**Last update:** 2026-05-21

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
