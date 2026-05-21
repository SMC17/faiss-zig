# STATUS

**Version:** v0.0.1
**Last update:** 2026-05-20

## What landed

- `src/root.zig` — public API: `FlatIndex`, `Hit`, `Metric`, `Error`
- 14 unit tests covering:
  - Empty index returns `EmptyIndex`
  - Single-vector self-find (L2)
  - kNN ordering ascending for L2
  - Dimension mismatch on add + search
  - Inner-product ordering descending
  - Cosine ordering descending + normalisation
  - Cosine zero-norm rejection (both add + query)
  - `k > len()` returns `len()` hits
  - `addWithId` preserves caller IDs
  - `k=0` returns `InvalidK`
  - L2-squared, inner-product, vectorNorm primitives
- `examples/search.zig` — synthetic 1000-vector kNN demo CLI
- `bench/bench_flat.zig` — throughput bench
  - Default config N=10000, D=128, K=10, Q=100
  - Self-bench only (no faiss-cpp comparison yet)
- AGPL-3.0 LICENSE
- Zig 0.16 Io-compatible (`init: std.process.Init` for binaries,
  `std.testing.allocator` for tests)

## What is open for v0.1

1. **SIMD acceleration** of the inner distance loops via `std.simd`
2. **Vector serialisation format** — load / save vectors from disk
3. **Head-to-head bench against `facebookresearch/faiss`** in the same
   style as `safetensors-zig/bench/bench_rust/`
4. **Multi-thread search** — partition the index across cores
5. **`IndexHNSW`** — first approximate index
6. **`IndexIVFFlat`** — second approximate index, more memory-efficient

## What is open for v0.2+

- Product quantization (PQ, OPQ)
- GPU support (CUDA + Metal via mlx-nonapple)
- Disk-backed indexes (mmap'd vectors)
- Range search (radius-bounded instead of top-K)
- Cross-platform CI

## Honest non-claims

- **No faiss-cpp comparison at v0.0.1.** v0.1 stages the C++ faiss
  head-to-head bench. Until those numbers land, we make no perf
  claim against the canonical implementation.
- **Scalar only.** No SIMD intrinsics, no auto-vectorisation
  attempts. The compiler may auto-vectorise some loops at
  ReleaseFast but we don't measure that.
- **Single-platform tested.** x86_64-linux only.
- **No production deployment claim.** v0.0.1 is the API + correctness
  + tests gate. Production needs SIMD + at least one approximate
  index + multi-thread search at minimum.
