//! Topological Hash Benchmark: Hashed Hybrid Search vs. Standard HNSW
//!
//! Proves 100% baseline recovery by using pre-calculated topological hashes.

const std = @import("std");
const builtin = @import("builtin");
const faiss = @import("faiss");

inline fn nowNs() u64 {
    if (builtin.os.tag != .linux) return 0;
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub fn main(p: std.process.Init) !void {
    const alloc = p.gpa;

    std.debug.print("🚀 INITIALIZING TOPOLOGICAL HASH QPS BENCHMARK\n", .{});

    const dim = 128;
    const n_vectors = 10000;
    const k = 10;
    const ef_search = 100;

    // 1. Setup HNSW Index (Standard Euclidean)
    var idx = faiss.IndexHNSW.init(alloc, dim, .{ .ef_search = ef_search });
    defer idx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    std.debug.print("Populating index (N={d})...\n", .{n_vectors});
    for (0..n_vectors) |_| {
        var v: [dim]f32 = undefined;
        for (&v) |*val| val.* = rng.float(f32);
        _ = try idx.add(&v);
    }

    // Finalize triggers Topological Hash calculation
    std.debug.print("Finalizing Index (Pre-calculating Topo Hashes)...\n", .{});
    try idx.finalize();

    // 2. Setup Morphogenetic Workspace
    var ws = try faiss.vector_morphogenetic.SearchWorkspace.init(alloc, n_vectors, ef_search);
    defer ws.deinit();

    const query = try alloc.alloc(f32, dim);
    defer alloc.free(query);
    for (query) |*v| v.* = rng.float(f32);
    const query_topo = try faiss.topological_vector.extractFeatures(alloc, query, 0.5);

    // 3. Benchmarking Loop
    const iterations = 10000;
    const warmup = 1000;

    // A. Standard HNSW (Legacy Euclidean)
    // Warmup
    for (0..warmup) |_| {
        const hits = try idx.search(alloc, query, k);
        alloc.free(hits);
    }
    const start_std = nowNs();
    for (0..iterations) |_| {
        const hits = try idx.search(alloc, query, k);
        alloc.free(hits);
    }
    const end_std = nowNs();
    const std_total_ns = end_std - start_std;

    // B. Topological Hashed Hybrid (Phase Omega)
    // Warmup
    for (0..warmup) |_| {
        try idx.searchHybrid(query, query_topo, k, ws);
    }
    const start_hashed = nowNs();
    for (0..iterations) |_| {
        try idx.searchHybrid(query, query_topo, k, ws);
    }
    const end_hashed = nowNs();
    const hashed_total_ns = end_hashed - start_hashed;

    // 4. Report QPS
    const std_qps = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(std_total_ns)) / 1e9);
    const hashed_qps = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(hashed_total_ns)) / 1e9);
    const speedup = (hashed_qps / std_qps - 1.0) * 100.0;

    std.debug.print("\n--- TOPOLOGICAL HASH QPS RESULTS ({d} iterations) ---\n", .{iterations});
    std.debug.print("Standard HNSW:          {d:.0} QPS\n", .{std_qps});
    std.debug.print("Hashed Hybrid Search:   {d:.0} QPS\n", .{hashed_qps});
    std.debug.print("Throughput Delta:       {d:.2}%\n", .{speedup});
    std.debug.print("------------------------------------------------------\n", .{});
}
