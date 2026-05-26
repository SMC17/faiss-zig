# faiss-zig

Pure-Zig vector similarity search. Portfolio artifact #2 of the
8-artifact jaw-drop portfolio (see `the broader sovereign-stack portfolio`).

## Status

**v0.2 — multi-threaded search batch. faiss-cpu gap narrows from ~4.5x → ~1.5x.**

| Implementation | ns/query | queries/sec | vs faiss-cpu |
|---|---|---|---|
| faiss-cpu 1.13.2 (C++ + AVX-2 + multi-thread) | ~510,000 | ~1,960 | 1.00x (ref) |
| **faiss-zig v0.2 (Zig + @Vector + 4 threads)** | **~772,000** | **~1,295** | **~1.5x slower** |
| faiss-zig v0.1 (Zig + @Vector + 1 thread) | ~2,290,000 | ~440 | ~4.5x slower |
| faiss-zig v0.0.1 (Zig scalar + 1 thread) | ~22,900,000 | ~44 | ~45x slower |

Scaling at N=10000, D=128, Q=100 across 1/2/4/8 threads:

| Threads | total ms | ns/query | queries/sec | scaling |
|---|---|---|---|---|
| 1 | 230 | 2,303,033 | 434 | 1.00x |
| 2 | 128 | 1,289,802 | 775 | 1.79x |
| 4 | 77 | 772,177 | 1,295 | **2.98x** |
| 8 | 84 | 840,507 | 1,189 | 2.74x |

The 4→8 thread regression is hyperthreading + memory-bandwidth
contention on the 4-core / 8-thread consumer CPU. Real 8-core
hardware (without HT contention) should hold 5-6x scaling.

**v0.1 — SIMD distance loops + audited head-to-head against `faiss-cpu`.**

The v0.0.1 scalar single-thread baseline was ~50x off faiss-cpu's
SIMD+multithread tuned C++. v0.1 closes most of the gap.

### v0.1 benchmark — single-thread, ReleaseFast, Ryzen-class consumer Linux

Inner-loop distance throughput (raw, no Index overhead, median of 3
trials, N=100K vectors per dim):

| Dim | SIMD `@Vector(8, f32)` | Scalar reduction loop | Ratio |
|---|---|---|---|
| 16 | 7 ns/dist | 16 ns/dist | 2.3x |
| 64 | 23 ns/dist | 103 ns/dist | 4.5x |
| 128 | 49 ns/dist | 239 ns/dist | 4.9x |
| 256 | 89 ns/dist | 489 ns/dist | 5.5x |
| 512 | 169 ns/dist | 1024 ns/dist | 6.1x |
| 1024 | 362 ns/dist | 2148 ns/dist | 5.9x |

The scalar loop is also `ReleaseFast` Zig — the win comes from
`@Vector(8, f32)` letting LLVM emit 8 parallel f32 accumulators
that the strict single-accumulator scalar reduction can't.

End-to-end head-to-head against `faiss-cpu` (Python wrapper of
`facebookresearch/faiss` C++) on N=10000, D=128, K=10, Q=100, L2:

| Implementation | ns/query | queries/sec | vectors-scored/sec |
|---|---|---|---|
| `faiss-cpu` 1.13.2 (C++ + AVX-2 + multi-thread) | ~510,000 | ~1,960 | ~19.6M |
| **faiss-zig v0.1** (Zig + @Vector + single-thread) | ~2,290,000 | ~440 | ~4.4M |

**Honest result: faiss-cpu is ~4.5x faster end-to-end.** The C++ side
uses hand-tuned AVX-2 + cache-aware tiling + multi-threading;
faiss-zig at v0.1 is single-threaded with one @Vector(8, f32)
accumulator and no cache tiling. The gap is closeable with v0.2
multi-threading (expected 4-8x on this 8-core CPU) but we don't
claim it until measured.

### Path to closing the faiss-cpu gap (v0.2 work)

1. **Multi-threading**: trivially parallel over queries; std.Thread
   pool gives 4-8x on consumer 8-core hardware. Expected to close
   most of the remaining gap.
2. **Cache-aware tiling**: blocks vectors so the working set fits
   L2/L3. Modest gains, ~1.5x at D=128.
3. **f16 / bf16 storage**: 2x memory bandwidth win, modest
   compute cost. Optional.

## What's covered

| Index family | Module | Default config | Headline perf @ N=10K D=128 |
|---|---|---|---|
| Flat (brute-force) | `FlatIndex` | L2 / IP / cosine, SIMD | 1,792 μs/q, 557 q/s, ground truth |
| HNSW | `IndexHNSW` | M=16, efC=200, efS=100 | (see bench-hnsw) |
| IVFFlat | `IndexIVFFlat` | nlist=100, nprobe=10 | 208 μs/q, 4,801 q/s, **8.61x vs Flat** |
| **IVFPQ** | **`IndexIVFPQ`** | **nlist=100, nprobe=10, M=8** | **522 μs/q, 1,915 q/s, 3.43x vs Flat, 16.94x mem compression** |

- **FlatIndex** — brute-force kNN with L2-squared / inner-product /
  cosine. SIMD `@Vector(8, f32)` distance loops, multi-threaded
  `searchBatch`.
- **IndexHNSW** — hierarchical navigable small-world approximate
  index. L2-squared only at v0.3+; default M=16, ef_construction=200,
  ef_search=100. Greedy multi-level descent + ef_search-bounded layer
  search.
- **IndexIVFFlat** — inverted-file approximate index (v0.6).
  Lloyd k-means clusters the training set into `nlist` centroids; at
  query time we rank centroids and scan the `nprobe` nearest lists.
  Default nlist=100, nprobe=10, n_iter=10, L2-squared. Observed
  **8.61x speedup vs FlatIndex** at N=10000 D=128 on this CPU.
- **IndexIVFPQ** — inverted-file + Product Quantization (new in v0.7).
  Production FAISS workhorse. Coarse k-means → nlist centroids; each
  vector stored as an M-byte PQ code of its residual against the
  assigned coarse centroid. Search uses Asymmetric Distance Computation
  (ADC) via a per-query M × 256 LUT — pure byte-fetch + sum, no
  per-vector multiply. Default nlist=100, nprobe=10, pq_m=8.
  **16.94x memory compression** vs raw f32 storage at N=10K D=128
  (295 KB total vs 5,000 KB), **42.67x per-vector compression**
  (12 bytes/vec vs 512). Recall trades against memory: 0.10 on
  uniform [0,1]^128 random data at this config (real clustered
  embeddings recover toward FAISS-published 0.6-0.8 R@10).
- **Public `faiss.scalar.*` and `faiss.distance.*`** for callers that
  want to bypass the Index wrapper or compare paths.

## What it does NOT do (yet)

- **Cosine + inner-product on HNSW / IVFFlat / IVFPQ** — v0.7.1
  (L2-squared only at v0.7).
- **Optimal Product Quantization (OPQ rotation matrix)** — v0.8+.
  OPQ rotates inputs to maximise PQ codebook quality on axis-
  correlated data.
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
zig build test                         # 17 unit tests (incl. SIMD vs scalar agreement)
zig build search                       # synthetic kNN demo CLI
zig build bench                        # flat-index throughput
zig build bench-simd                   # SIMD vs scalar speedup at varied D

# faiss-cpu head-to-head (requires venv with faiss-cpu + numpy)
python3 -m venv .venv && source .venv/bin/activate
pip install faiss-cpu numpy
LD_LIBRARY_PATH=/usr/lib python3 bench/bench_faiss_cpu.py
```

## Comparable upstream

- [`facebookresearch/faiss`](https://github.com/facebookresearch/faiss)
  — C++ canonical. faiss-zig v0.1 is single-threaded with single-
  accumulator @Vector lowering; faiss-cpu uses multi-thread +
  AVX-2-tuned kernels. We cite, we bench honestly, we publish the
  ~4.5x loss at v0.1.

## License

AGPL-3.0-or-later. See `LICENSE`.

## Honest non-claims

- **No multi-threading at v0.1.** Single-threaded inner loops.
  faiss-cpu uses OMP / pthread internally. The ~4.5x gap is
  partially attributable to this.
- **No cache-aware tiling.** Working set blasts through L2/L3 on
  large N; faiss-cpu blocks. Closeable in v0.2.
- **No approximate index.** Flat only. For N > 1M or D > 1024,
  production needs IVF / HNSW / PQ. Roadmap.
- **No GPU.** Not a near-term target.
- **Single-platform tested** (x86_64-linux, Zig 0.16 ReleaseFast,
  Ryzen-class consumer CPU).
- The SIMD speedup table reflects this specific CPU's auto-vectorisation
  behavior at ReleaseFast. Different compilers / -O levels / CPUs
  may produce different ratios.
