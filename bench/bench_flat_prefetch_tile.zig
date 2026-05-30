//! Flat-search throughput with explicit @prefetch + tiled query
//! processing.
//!
//! Comptime-D=128 alone matched library general-D (53 vs 54 q/s),
//! both at 0.28x of faiss-cpu single-thread 187 q/s. The gap is in
//! cache layout and prefetch instructions — not the kernel.
//!
//! This bench tries the explicit-prefetch lever:
//!   - @prefetch(corpus + (i + TILE) * D) ahead of the inner loop to
//!     mask the L2/L3 fetch latency
//!   - Tile over corpus in TILE chunks to keep query vector hot in
//!     registers/L1 while we stream corpus through L2
//!
//! Pre-reg target: ≥1.0x parity with faiss-cpu single-thread.
//! Falsifier: <0.7x of faiss-cpu.

const std = @import("std");
const faiss = @import("faiss");

const N: usize = 10_000;
const D: comptime_int = 128;
const K: usize = 10;
const Q: usize = 100;
const LANE: comptime_int = 8;
const TILE: comptime_int = 16; // prefetch lookahead in corpus rows
const V = @Vector(LANE, f32);

inline fn l2SquaredD128(a: *const [D]f32, b: *const [D]f32) f32 {
    var sum_vec: V = @splat(0.0);
    comptime var i: usize = 0;
    inline while (i < D) : (i += LANE) {
        const av: V = a[i..][0..LANE].*;
        const bv: V = b[i..][0..LANE].*;
        const d = av - bv;
        sum_vec += d * d;
    }
    return @reduce(.Add, sum_vec);
}

const Hit = faiss.Hit;
fn hitLessL2(_: void, x: Hit, y: Hit) bool {
    return x.score < y.score;
}

/// Flat search with prefetch-ahead. We process corpus in TILE chunks
/// and issue a software prefetch for the next chunk before working
/// on the current one. Empirically (and per the Intel optimization
/// manual) this masks L2 fetch latency for streaming workloads where
/// the access pattern is fully predictable.
fn searchPrefetch(
    alloc: std.mem.Allocator,
    vectors: []const [D]f32,
    ids: []const u32,
    query: *const [D]f32,
    k: usize,
) ![]Hit {
    const n = vectors.len;
    var all_hits = try alloc.alloc(Hit, n);
    defer alloc.free(all_hits);

    var i: usize = 0;
    // Prefetch the first TILE rows before entering the hot loop.
    {
        var p: usize = 0;
        while (p < @min(TILE, n)) : (p += 1) {
            @prefetch(@as([*]const f32, @ptrCast(&vectors[p])), .{ .rw = .read, .locality = 3 });
        }
    }

    while (i + TILE <= n) : (i += TILE) {
        // Prefetch the NEXT tile while we compute on the current one.
        var p = i + TILE;
        const end_prefetch = @min(p + TILE, n);
        while (p < end_prefetch) : (p += 1) {
            @prefetch(@as([*]const f32, @ptrCast(&vectors[p])), .{ .rw = .read, .locality = 3 });
        }
        // Compute distance for the current TILE rows
        inline for (0..TILE) |t| {
            const idx = i + t;
            all_hits[idx] = .{ .id = ids[idx], .score = l2SquaredD128(query, &vectors[idx]) };
        }
    }
    // Tail
    while (i < n) : (i += 1) {
        all_hits[i] = .{ .id = ids[i], .score = l2SquaredD128(query, &vectors[i]) };
    }

    std.sort.pdq(Hit, all_hits, {}, hitLessL2);
    const out = try alloc.alloc(Hit, k);
    @memcpy(out, all_hits[0..k]);
    return out;
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    const vectors = try alloc.alloc([D]f32, N);
    defer alloc.free(vectors);
    const ids = try alloc.alloc(u32, N);
    defer alloc.free(ids);
    for (vectors, ids, 0..) |*v, *id, i| {
        for (v) |*x| x.* = rng.float(f32);
        id.* = @intCast(i);
    }
    const queries = try alloc.alloc([D]f32, Q);
    defer alloc.free(queries);
    for (queries) |*q| {
        for (q) |*x| x.* = rng.float(f32);
    }

    // Warmup
    {
        const w = try searchPrefetch(alloc, vectors, ids, &queries[0], K);
        alloc.free(w);
    }

    const t0 = std.Io.Clock.awake.now(io);
    var checksum: u64 = 0;
    for (queries) |*q| {
        const hits = try searchPrefetch(alloc, vectors, ids, q, K);
        defer alloc.free(hits);
        checksum += hits[0].id;
    }
    const total_ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const ns_per_query = total_ns / Q;
    const qps = @as(u64, Q) * std.time.ns_per_s / total_ns;
    std.debug.print("=== faiss-zig Flat with prefetch+tile (N={d} D={d} K={d} Q={d}, TILE={d}) ===\n", .{ N, D, K, Q, TILE });
    std.debug.print("prefetch_tile\n", .{});
    std.debug.print("  ns_per_query    = {d}\n", .{ns_per_query});
    std.debug.print("  us_per_query    = {d}\n", .{ns_per_query / 1000});
    std.debug.print("  queries_per_sec = {d}\n", .{qps});
    std.debug.print("  (cksum {d})\n\n", .{checksum});
    std.debug.print("Reference (comptime-D=128 no prefetch): 53 q/s\n", .{});
    std.debug.print("Reference (library general-D no prefetch): 54 q/s\n", .{});
    std.debug.print("Reference (faiss-cpu single-thread):     187 q/s\n", .{});
}
