//! HNSW CSR A/B bench: same index, measure query latency
//! WITHOUT CSR (nested ArrayList path) vs WITH CSR (flat offsets+edges).
//!
//! Pre-reg: exp-1780055335-755192162. Target: >=1.3x speedup with CSR.

const std = @import("std");
const faiss = @import("faiss");

const N: usize = 10_000;
const D: u32 = 128;
const K: usize = 10;
const Q: usize = 100;
const WARMUP: usize = 20;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    var idx = faiss.IndexHNSW.init(alloc, D, .{
        .M = 16,
        .ef_construction = 200,
        .ef_search = 100,
        .seed = 42,
    });
    defer idx.deinit();

    const vec_buf = try alloc.alloc(f32, D);
    defer alloc.free(vec_buf);

    var i: usize = 0;
    while (i < N) : (i += 1) {
        for (vec_buf) |*v| v.* = rng.float(f32);
        _ = try idx.add(vec_buf);
    }

    const queries = try alloc.alloc(f32, Q * D);
    defer alloc.free(queries);
    for (queries) |*v| v.* = rng.float(f32);

    // Warmup BEFORE-CSR
    for (0..WARMUP) |qi| {
        const q = queries[(qi % Q) * D ..][0..D];
        const hits = try idx.search(alloc, q, K);
        defer alloc.free(hits);
    }

    // Timed BEFORE-CSR (nested ArrayList path)
    var before_sum: u64 = 0;
    const t0 = std.Io.Clock.awake.now(io);
    for (0..Q) |qi| {
        const q = queries[qi * D ..][0..D];
        const hits = try idx.search(alloc, q, K);
        defer alloc.free(hits);
        before_sum += hits[0].id;
    }
    const t1 = std.Io.Clock.awake.now(io);
    const ns_before: u64 = @intCast(t0.durationTo(t1).nanoseconds);

    // Build CSR
    try idx.finalize();

    // Warmup AFTER-CSR
    for (0..WARMUP) |qi| {
        const q = queries[(qi % Q) * D ..][0..D];
        const hits = try idx.search(alloc, q, K);
        defer alloc.free(hits);
    }

    // Timed AFTER-CSR
    var after_sum: u64 = 0;
    const t2 = std.Io.Clock.awake.now(io);
    for (0..Q) |qi| {
        const q = queries[qi * D ..][0..D];
        const hits = try idx.search(alloc, q, K);
        defer alloc.free(hits);
        after_sum += hits[0].id;
    }
    const t3 = std.Io.Clock.awake.now(io);
    const ns_after: u64 = @intCast(t2.durationTo(t3).nanoseconds);

    const us_before = ns_before / Q / 1000;
    const us_after = ns_after / Q / 1000;
    const qps_before = @as(u64, Q) * std.time.ns_per_s / ns_before;
    const qps_after = @as(u64, Q) * std.time.ns_per_s / ns_after;
    const speedup_x100 = (ns_before * 100) / ns_after;

    std.debug.print("=== HNSW CSR A/B (N={d}, D={d}, K={d}, Q={d}) ===\n", .{ N, D, K, Q });
    std.debug.print("\nBefore CSR (nested ArrayList):\n", .{});
    std.debug.print("  us_per_query    {d}\n", .{us_before});
    std.debug.print("  queries_per_sec {d}\n", .{qps_before});
    std.debug.print("\nAfter CSR (flat offsets+edges):\n", .{});
    std.debug.print("  us_per_query    {d}\n", .{us_after});
    std.debug.print("  queries_per_sec {d}\n", .{qps_after});
    std.debug.print("\nSpeedup_x100     {d}  (>=130 = >=1.30x = OoM-confirming)\n", .{speedup_x100});
    std.debug.print("\nCorrectness check: before_sum={d} after_sum={d} ({s})\n", .{
        before_sum,
        after_sum,
        if (before_sum == after_sum) "MATCH" else "DIVERGE",
    });
}
