//! HNSW bench: build + query throughput at N=10000 D=128, matched to
//! bench_hnsw_vs_faiss_cpu.py for head-to-head comparison.

const std = @import("std");
const faiss = @import("faiss");

// HONEST NON-CLAIM: bench currently SEGVs at N > ~250 with the v0.3
// HNSW implementation. Root cause under investigation (likely stale-
// pointer issue from outer ArrayList grow invalidating cached &elem
// pointers). v0.5 blocker; tracked in STATUS.md.
const N: usize = 200;
const D: u32 = 32;
const K: usize = 10;
const Q: usize = 100;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    // --- Build HNSW ---
    var idx = faiss.IndexHNSW.init(alloc, D, .{
        .M = 16,
        .ef_construction = 200,
        .ef_search = 100,
        .seed = 42,
    });
    defer idx.deinit();

    const vec_buf = try alloc.alloc(f32, D);
    defer alloc.free(vec_buf);

    const build_start = std.Io.Clock.awake.now(io);
    var i: usize = 0;
    while (i < N) : (i += 1) {
        for (vec_buf) |*v| v.* = rng.float(f32);
        _ = try idx.add(vec_buf);
    }
    const build_end = std.Io.Clock.awake.now(io);
    const build_ns: u64 = @intCast(build_start.durationTo(build_end).nanoseconds);

    // --- Build Flat for recall reference ---
    var flat = faiss.FlatIndex.init(alloc, D, .l2_squared);
    defer flat.deinit();
    var prng2: std.Random.DefaultPrng = .init(42);
    const rng2 = prng2.random();
    var ii: usize = 0;
    while (ii < N) : (ii += 1) {
        for (vec_buf) |*v| v.* = rng2.float(f32);
        _ = try flat.add(vec_buf);
    }

    // --- Queries (same seed for both) ---
    const queries = try alloc.alloc(f32, Q * D);
    defer alloc.free(queries);
    for (queries) |*v| v.* = rng.float(f32);

    // --- Warmup ---
    {
        const w = try idx.search(alloc, queries[0..D], K);
        defer alloc.free(w);
    }

    // --- HNSW query phase ---
    const hnsw_qs = std.Io.Clock.awake.now(io);
    var qi: usize = 0;
    var hnsw_total_top1: u64 = 0;
    while (qi < Q) : (qi += 1) {
        const q = queries[qi * D ..][0..D];
        const hits = try idx.search(alloc, q, K);
        defer alloc.free(hits);
        hnsw_total_top1 += hits[0].id;
    }
    const hnsw_ns: u64 = @intCast(hnsw_qs.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // --- Flat query phase ---
    const flat_qs = std.Io.Clock.awake.now(io);
    qi = 0;
    var flat_total_top1: u64 = 0;
    while (qi < Q) : (qi += 1) {
        const q = queries[qi * D ..][0..D];
        const hits = try flat.search(alloc, q, K);
        defer alloc.free(hits);
        flat_total_top1 += hits[0].id;
    }
    const flat_ns: u64 = @intCast(flat_qs.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // --- Recall ---
    var total_recall: f32 = 0;
    qi = 0;
    while (qi < Q) : (qi += 1) {
        const q = queries[qi * D ..][0..D];
        const hnsw_hits = try idx.search(alloc, q, K);
        defer alloc.free(hnsw_hits);
        const flat_hits = try flat.search(alloc, q, K);
        defer alloc.free(flat_hits);
        var h_set = std.AutoHashMap(u32, void).init(alloc);
        defer h_set.deinit();
        for (hnsw_hits) |h| try h_set.put(h.id, {});
        var matches: usize = 0;
        for (flat_hits) |h| {
            if (h_set.contains(h.id)) matches += 1;
        }
        total_recall += @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(K));
    }
    const recall = total_recall / @as(f32, @floatFromInt(Q));

    std.debug.print("=== faiss-zig HNSW vs Flat (N={d}, D={d}, K={d}, Q={d}) ===\n", .{ N, D, K, Q });
    std.debug.print("\nBuild phase:\n", .{});
    std.debug.print("  HNSW build_ms        = {d}\n", .{build_ns / 1_000_000});
    std.debug.print("  HNSW ns_per_add      = {d}\n", .{build_ns / N});
    std.debug.print("\nQuery phase:\n", .{});
    std.debug.print("  HNSW ns_per_query    = {d}\n", .{hnsw_ns / Q});
    std.debug.print("  HNSW us_per_query    = {d}\n", .{hnsw_ns / Q / 1000});
    std.debug.print("  HNSW queries_per_sec = {d}\n", .{@as(u64, Q) * std.time.ns_per_s / hnsw_ns});
    std.debug.print("\n  Flat ns_per_query    = {d}\n", .{flat_ns / Q});
    std.debug.print("  Flat us_per_query    = {d}\n", .{flat_ns / Q / 1000});
    std.debug.print("  Flat queries_per_sec = {d}\n", .{@as(u64, Q) * std.time.ns_per_s / flat_ns});
    std.debug.print("\n  HNSW speedup vs Flat = {d:.2}x\n", .{@as(f64, @floatFromInt(flat_ns)) / @as(f64, @floatFromInt(hnsw_ns))});
    std.debug.print("  HNSW top-{d} recall   = {d:.3}\n", .{ K, recall });

    // Anti-DCE
    std.debug.print("\n(checksums: hnsw_top1_sum={d} flat_top1_sum={d})\n", .{ hnsw_total_top1, flat_total_top1 });
}
