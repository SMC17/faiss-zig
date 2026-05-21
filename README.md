# faiss-zig

Pure-Zig vector similarity search. Portfolio artifact #2 of the
8-artifact jaw-drop portfolio (see `~/portfolio/README.md`).

## Status

**v0.0.1 — Flat (brute-force) index with L2-squared, inner-product,
cosine metrics. Scalar, single-thread.**

Comparable upstream: [facebookresearch/faiss](https://github.com/facebookresearch/faiss)
— C++ canonical, SIMD-tuned, GPU support. faiss-zig at v0.0.1 is a
correctness-first build; no head-to-head against faiss-cpp yet
(staged for v0.1, same posture as `safetensors-zig`'s Rust bench).

Pre-1.0 substrate per the Zig 0.16 ecosystem convention.

## What it does

- `FlatIndex.init(allocator, dim, metric)` — construct an in-memory
  brute-force index
- `addWithId(id, vector)` — insert a vector with caller-provided ID
- `add(vector)` — insert with auto-assigned ID
- `search(alloc, query, k)` — kNN search, returns `[]Hit` sorted by
  score (ascending for `l2_squared`, descending for `inner_product`
  and `cosine`)
- Three metrics:
  - `l2_squared` — squared Euclidean (sqrt skipped for speed; rank is
    monotonic so order is unchanged)
  - `inner_product` — sum of products
  - `cosine` — dot(a,b) / (|a| * |b|); rejects zero-norm vectors

## What it does NOT do (yet)

- **SIMD acceleration** — scalar loops at v0.0.1; AVX-2 / NEON
  acceleration is v0.1 work
- **Approximate indexes** — no IVF, no HNSW, no PQ, no OPQ. Flat
  only. v0.1+ work.
- **GPU support** — CPU only. v0.x.
- **Multi-thread search** — single-threaded at v0.0.1
- **Disk-backed indexes** — in-memory only
- **Vector serialisation format** — no binary file format yet; v0.1
  task. The example CLI generates synthetic vectors inline.
- **Head-to-head bench vs `facebookresearch/faiss`** — v0.1 work.
  v0.0.1 ships a Zig self-bench only.

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

    // Add vectors
    var v: [128]f32 = undefined;
    // ... fill v ...
    _ = try idx.add(&v);

    // Search
    var query: [128]f32 = undefined;
    // ... fill query ...
    const hits = try idx.search(alloc, &query, 10);
    defer alloc.free(hits);

    for (hits) |h| {
        std.debug.print("id={d} score={d:.4}\n", .{ h.id, h.score });
    }
}
```

## Build

```sh
zig build test            # 14 unit tests
zig build search          # synthetic 1000-vector kNN demo
zig build bench           # throughput bench (N=10000, D=128 default)
zig build install -Doptimize=ReleaseFast
```

## Position in the portfolio

faiss-zig is artifact #2 of the 8-artifact jaw-drop portfolio. Every
RAG / retrieval / embedding-search shop is a user of faiss-cpp; a
Zig-native sibling with audited bench numbers is a high-aura
artifact at low engineering cost. Audience: HuggingFace, vector-DB
vendors (Pinecone, Weaviate, Qdrant, Milvus), every shop running
RAG at production.

## License

AGPL-3.0-or-later. See `LICENSE`.

## Honest non-claims

- **No SIMD yet.** Scalar Zig loops will not match faiss-cpp's
  hand-tuned AVX-512 / NEON. The architecture admits SIMD
  acceleration cleanly via `std.simd`; v0.1 work.
- **No approximate indexes.** Flat at v0.0.1 only. For N > 1M or
  D > 1024, an approximate index is what production needs; we don't
  have one yet.
- **Single-thread.** Search latency scales linearly with N. v0.0.1
  is not the throughput tier.
- **No GPU.** Not a near-term target. faiss-cpp's GPU path is the
  reference; if a Zig GPU path materialises it would compose with
  `mlx-nonapple` from portfolio artifact #9.
- **No production deployment story.** v0.0.1 is the correctness-and-
  API gate; the bench / SIMD / IVF / HNSW work that makes the
  library production-grade is v0.1+.
