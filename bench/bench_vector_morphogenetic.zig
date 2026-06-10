//! Comparative Vector Benchmark: Legacy vs. Morphogenetic
//!
//! Measures µs delta for IVF search between legacy nested-list implementation
//! and the new zero-allocation Morphogenetic substrate.

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

    std.debug.print("🚀 INITIALIZING VECTOR MORPHOGENETIC COMPARATIVE BENCHMARK\n", .{});

    const dim = 128;
    const n_vectors = 10000;
    const nlist = 100;
    const nprobe = 10;
    const k = 10;

    // 1. Setup Index and Training
    var idx = try faiss.IndexIVFFlat.init(alloc, dim, .{ .nlist = nlist, .nprobe = nprobe });
    defer idx.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const training_data = try alloc.alloc(f32, n_vectors * dim);
    defer alloc.free(training_data);
    for (training_data) |*v| v.* = rng.float(f32);

    try idx.train(training_data);
    for (0..n_vectors) |i| {
        _ = try idx.add(training_data[i * dim ..][0..dim]);
    }

    const query = try alloc.alloc(f32, dim);
    defer alloc.free(query);
    for (query) |*v| v.* = rng.float(f32);

    // 2. Benchmarking Loop
    const warmup = 100;
    const iterations = 1000;

    // --- WARMUP ---
    for (0..warmup) |_| {
        const hits = try idx.search(alloc, query, k);
        alloc.free(hits);
    }

    // A. Legacy Pass
    const start_legacy = nowNs();
    for (0..iterations) |_| {
        const hits = try idx.search(alloc, query, k);
        alloc.free(hits);
    }
    const end_legacy = nowNs();
    const legacy_total_ns = end_legacy - start_legacy;

    // B. Morphogenetic Pass
    var ws = try faiss.vector_morphogenetic.SearchWorkspace.init(alloc, n_vectors, nlist);
    defer ws.deinit();

    // Warmup Morph
    for (0..warmup) |_| {
        const hits = try faiss.ivf_morphogenetic.searchIVFMorphogenetic(&idx, query, k, ws);
        alloc.free(hits);
    }

    const start_morph = nowNs();
    for (0..iterations) |_| {
        const hits = try faiss.ivf_morphogenetic.searchIVFMorphogenetic(&idx, query, k, ws);
        alloc.free(hits);
    }
    const end_morph = nowNs();
    const morph_total_ns = end_morph - start_morph;

    // 3. Report
    const legacy_avg_us = @as(f64, @floatFromInt(legacy_total_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;
    const morph_avg_us = @as(f64, @floatFromInt(morph_total_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;
    const reduction = (legacy_avg_us - morph_avg_us) / legacy_avg_us * 100.0;

    std.debug.print("\n--- VECTOR BENCHMARK RESULTS ({d} iterations) ---\n", .{iterations});
    std.debug.print("Legacy IVF Search:          {d:.2} µs / query\n", .{legacy_avg_us});
    std.debug.print("Morphogenetic IVF Search:   {d:.2} µs / query\n", .{morph_avg_us});
    std.debug.print("Latency Reduction:          {d:.2}%\n", .{reduction});
    std.debug.print("------------------------------------------\n", .{});
}
