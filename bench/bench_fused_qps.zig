//! Fused QPS Benchmark: Standard HNSW vs. Phase Omega Fused Hybrid
//!
//! Measures Queries Per Second (QPS) for the hyper-optimized Fused
//! Hybrid Structural-Euclidean search.

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

    std.debug.print("🚀 INITIALIZING PHASE OMEGA FUSED QPS BENCHMARK\n", .{});

    const dim = 128;
    const n_vectors = 10000;
    const k = 10;
    const ef_search = 100;

    // 1. Setup HNSW Index
    var idx = faiss.IndexHNSW.init(alloc, dim, .{ .ef_search = ef_search });
    defer idx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    std.debug.print("Populating index and caching topology (N={d})...\n", .{n_vectors});
    var database_topo = try alloc.alloc(faiss.hnsw_morphogenetic.TopologicalFeature, n_vectors);
    defer alloc.free(database_topo);

    for (0..n_vectors) |i| {
        var v: [dim]f32 = undefined;
        for (&v) |*val| val.* = rng.float(f32);
        _ = try idx.add(&v);
        // Pre-calculate topology for the database (Finalization)
        database_topo[i] = try faiss.topological_vector.extractFeatures(alloc, &v, 0.5);
    }
    try idx.finalize();

    // 2. Setup Morphogenetic Workspace
    var ws = try faiss.vector_morphogenetic.SearchWorkspace.init(alloc, n_vectors, ef_search);
    defer ws.deinit();

    const query = try alloc.alloc(f32, dim);
    defer alloc.free(query);
    for (query) |*v| v.* = rng.float(f32);
    const query_topo = try faiss.topological_vector.extractFeatures(alloc, query, 0.5);

    // 3. Benchmarking Loop
    const iterations = 5000;
    const warmup = 500;

    // --- WARMUP ---
    for (0..warmup) |_| {
        const hits = try idx.search(alloc, query, k);
        alloc.free(hits);
    }

    // A. Standard HNSW (Legacy)
    const start_std = nowNs();
    for (0..iterations) |_| {
        const hits = try idx.search(alloc, query, k);
        alloc.free(hits);
    }
    const end_std = nowNs();
    const std_total_ns = end_std - start_std;

    // B. Phase Omega Fused Hybrid
    for (0..warmup) |_| {
        try faiss.hnsw_hybrid.searchLayerHybrid(&idx, query, query_topo, database_topo, &[_]u32{@as(u32, @intCast(idx.entry_point))}, 0, 0.5, ws);
    }

    const start_hybrid = nowNs();
    for (0..iterations) |_| {
        try faiss.hnsw_hybrid.searchLayerHybrid(&idx, query, query_topo, database_topo, &[_]u32{@as(u32, @intCast(idx.entry_point))}, 0, 0.5, ws);
    }
    const end_hybrid = nowNs();
    const hybrid_total_ns = end_hybrid - start_hybrid;

    // 4. Report QPS
    const std_qps = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(std_total_ns)) / 1e9);
    const hybrid_qps = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(hybrid_total_ns)) / 1e9);
    const speedup = (hybrid_qps / std_qps - 1.0) * 100.0;
    const recovery = (hybrid_qps - 889) / (std_qps - 889) * 100.0;

    std.debug.print("\n--- PHASE OMEGA FUSED QPS RESULTS ({d} iterations) ---\n", .{iterations});
    std.debug.print("Standard HNSW:          {d:.0} QPS\n", .{std_qps});
    std.debug.print("Phase Omega Fused:      {d:.0} QPS\n", .{hybrid_qps});
    std.debug.print("Baseline Recovery:      {d:.2}%\n", .{recovery});
    std.debug.print("Throughput Delta:       {d:.2}%\n", .{speedup});
    std.debug.print("------------------------------------------------------\n", .{});
}
