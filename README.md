# faiss-zig

**[Status](STATUS.md) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md) · [Issues](https://github.com/SMC17/faiss-zig/issues) · [All SMC17 work](https://github.com/users/SMC17/projects/16)**

[![CI](https://github.com/SMC17/faiss-zig/actions/workflows/ci.yml/badge.svg)](https://github.com/SMC17/faiss-zig/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/SMC17/faiss-zig?sort=semver)](https://github.com/SMC17/faiss-zig/releases)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)
[![Zig](https://img.shields.io/badge/Zig-0.16-orange.svg)](https://ziglang.org/)

Pure-Zig vector similarity search. Zig 0.16, AGPL-3.0,
single-binary, libc-only.

## Status

**v0.7 — four index families: Flat + HNSW + IVFFlat + IVFPQ.**


| Index family | Module          | Default config             | Headline @ N=10K D=128                                    |
|--------------|-----------------|----------------------------|-----------------------------------------------------------|
| Flat         | `FlatIndex`     | L2 / IP / cosine, SIMD     | 378 µs/q, 2,640 q/s, ground truth                          |
| HNSW         | `IndexHNSW`     | M=16, efC=200, efS=100     | 70%+ top-10 recall vs Flat                                |
| IVFFlat      | `IndexIVFFlat`  | nlist=100, nprobe=10       | 208 µs/q, 4,801 q/s, **12.6x vs Flat**                    |
| IVFPQ        | `IndexIVFPQ`    | nlist=100, nprobe=10, M=8  | 522 µs/q, 1,915 q/s, **5.06x vs Flat, 16.94x mem-compression** |

Historical, not remeasured on the current commit. Head-to-head against `faiss-cpu` (Python wrapper of
`facebookresearch/faiss`) on N=10000, D=128, K=10, Q=100, L2:

| Implementation                                    | ns/query | queries/sec | vs faiss-cpu  |
|---------------------------------------------------|---------:|------------:|--------------:|
| `faiss-cpu` 1.13.2 (C++ + AVX-2 + multi-thread)   | ~510,000 | ~1,960      | 1.00x (ref)   |
| **faiss-zig v0.7** (Zig + `@Vector` + 4 threads)  | **~378,000** | **~2,640** | **~1.34x faster** |
| faiss-zig v0.2 (Zig + `@Vector` + 4 threads)      | ~772,000 | ~1,295      | ~1.5x slower  |
| faiss-zig v0.0.1 (Zig scalar + 1 thread)          | ~22,900,000 | ~44      | ~45x slower   |

Versions are the FlatIndex baselines; HNSW/IVFFlat/IVFPQ since
v0.2 add approximate-index speedups *on top* of the v0.2 Flat number.

## SIMD-vs-scalar distance loop

Inner-loop distance throughput (no Index overhead, median of 3
trials, N=100K vectors per dim, single thread):

| Dim  | SIMD `@Vector(8, f32)` | Scalar reduction | Ratio |
|------|------------------------|------------------|-------|
| 16   | 7 ns/dist              | 16 ns/dist       | 2.3x  |
| 64   | 23 ns/dist             | 103 ns/dist      | 4.5x  |
| 128  | 49 ns/dist             | 239 ns/dist      | 4.9x  |
| 256  | 89 ns/dist             | 489 ns/dist      | 5.5x  |
| 512  | 169 ns/dist            | 1,024 ns/dist    | 6.1x  |
| 1024 | 362 ns/dist            | 2,148 ns/dist    | 5.9x  |

The scalar loop is also `ReleaseFast` Zig — the win comes from
`@Vector(8, f32)` letting LLVM emit 8 parallel f32 accumulators
that the strict single-accumulator scalar reduction can't.

## What each index does

- **FlatIndex** — brute-force kNN with L2-squared / inner-product /
  cosine. SIMD `@Vector(8, f32)` distance loops, multi-threaded
  `searchBatch`.
- **IndexHNSW** — hierarchical navigable small-world (Malkov 2016).
  L2-squared only at v0.3+. Default M=16, ef_construction=200,
  ef_search=100. Greedy multi-level descent + ef_search-bounded
  layer search.
- **IndexIVFFlat** — inverted-file approximate index (v0.6).
  Lloyd k-means clusters the training set into `nlist` centroids;
  at query time we rank centroids and scan the `nprobe` nearest
  lists. Default nlist=100, nprobe=10, n_iter=10, L2-squared.
- **IndexIVFPQ** — inverted-file + Product Quantization (v0.7,
  the production FAISS workhorse). Coarse k-means → nlist
  centroids; each vector stored as an M-byte PQ code of its
  residual against the assigned coarse centroid. Search uses
  Asymmetric Distance Computation (ADC) via a per-query M × 256
  LUT — pure byte-fetch + sum, no per-vector multiply. Default
  nlist=100, nprobe=10, pq_m=8. **16.94x memory compression** vs
  raw f32 storage at N=10K D=128 (295 KB total vs 5,000 KB),
  **42.67x per-vector compression** (12 bytes/vec vs 512). Recall
  trades against memory: 0.10 on uniform [0,1]^128 random data
  at this config; real clustered embeddings recover toward the
  FAISS-published 0.6-0.8 R@10. Do not cite the compression ratio
  without the recall qualifier.

- **Public `faiss.scalar.*` / `faiss.distance.*`** for callers
  that want to bypass the Index wrapper or compare paths.

## What it does NOT do (yet)

- **Cosine + inner-product on HNSW / IVFFlat / IVFPQ** — L2-squared
  only at v0.7; cosine + IP land in v0.7.1.
- **Optimal Product Quantization (OPQ rotation matrix)** — v0.8.
- **k-means++ init** — v0.8 quality bump. Currently uniform-random
  sampling.
- **Parallel k-means training** — v0.8. Currently single-thread.
- **GPU** — not a near-term target.
- **Disk-backed indexes** — future.
- **Vector serialisation format** — future.

## Quickstart

```zig
const std = @import("std");
const faiss = @import("faiss");

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var idx = faiss.FlatIndex.init(alloc, 128, .l2_squared);
    defer idx.deinit();

    var v: [128]f32 = undefined;
    // ... fill v ...
    _ = try idx.add(&v);

    var query: [128]f32 = undefined;
    const hits = try idx.search(alloc, &query, 10);
    defer alloc.free(hits);

    for (hits) |h| std.debug.print("id={d} score={d:.4}\n", .{ h.id, h.score });
}
```

## Build + bench

```sh
zig build test                  # unit tests (incl. SIMD vs scalar agreement)
zig build search                # synthetic kNN demo CLI
zig build bench                 # flat-index throughput
zig build bench-simd            # SIMD vs scalar speedup at varied D

# faiss-cpu head-to-head (requires venv with faiss-cpu + numpy)
python3 -m venv .venv && source .venv/bin/activate
pip install faiss-cpu numpy
LD_LIBRARY_PATH=/usr/lib python3 bench/bench_faiss_cpu.py
```

## Comparable upstream

- [`facebookresearch/faiss`](https://github.com/facebookresearch/faiss)
  — C++ canonical. faiss-zig at v0.7 covers the same four index
  families that matter most in production: Flat, HNSW, IVFFlat,
  IVFPQ. The C++ side has hand-tuned AVX-2 + cache-aware tiling +
  GPU; faiss-zig is multi-threaded SIMD on CPU only. v0.2 closed
  the Flat gap to ~1.5x. HNSW / IVFFlat / IVFPQ approximate-index
  numbers compete in their own regimes.

## License

AGPL-3.0-or-later. See `LICENSE`.

## Honest non-claims

- **No GPU.** Not a near-term target.
- **No cache-aware tiling on FlatIndex.** faiss-cpu blocks; we
  don't. Closing remaining gap below 1.5x needs this.
- **L2-squared only** on the three approximate indexes at v0.7.
  Cosine + IP for HNSW / IVF / IVFPQ are v0.7.1.
- **Single-platform tested** (x86_64-linux, Zig 0.16 ReleaseFast,
  Ryzen-class consumer Linux).
- The SIMD speedup table reflects this specific CPU's
  auto-vectorisation behavior at ReleaseFast. Different
  compilers / `-O` levels / CPUs may produce different ratios.
- **Recall on uniform random data is misleading.** PQ shines on
  clustered real embeddings; the 0.10 R@10 number at default
  config on uniform [0,1]^128 is an artifact of the worst-case
  distribution, not the expected real-world recall.

## Reproducible setup

```
nix develop        # Zig 0.16.0 + jq + curl + gawk + python3(websockets)
zig build test     # 76/76 expected
```

The repo's `flake.nix` pins exact dependency versions for `nix develop`.
