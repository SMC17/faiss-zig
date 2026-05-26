//! IVFPQ bench: train + add + query throughput at N=10000 D=128 M=8
//! nlist=100 nprobe=10, head-to-head with IVFFlat AND FlatIndex.
//! Reports recall + memory footprint.

const std = @import("std");
const faiss = @import("faiss");

const N: usize = 10_000;
const D: u32 = 128;
const K: usize = 10;
const Q: usize = 100;
const NLIST: u32 = 100;
const NPROBE: u32 = 10;
const PQ_M: u32 = 8;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    // --- shared dataset ---
    const dataset = try alloc.alloc(f32, N * D);
    defer alloc.free(dataset);
    for (dataset) |*v| v.* = rng.float(f32);

    // --- IVFPQ ---
    var ivfpq = try faiss.IndexIVFPQ.init(alloc, D, .{
        .nlist = NLIST,
        .nprobe = NPROBE,
        .pq_m = PQ_M,
        .n_iter = 10,
        .seed = 42,
    });
    defer ivfpq.deinit();

    const pq_train_start = std.Io.Clock.awake.now(io);
    try ivfpq.train(dataset);
    const pq_train_ns: u64 = @intCast(pq_train_start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const pq_add_start = std.Io.Clock.awake.now(io);
    var i: usize = 0;
    while (i < N) : (i += 1) {
        _ = try ivfpq.add(dataset[i * D ..][0..D]);
    }
    const pq_add_ns: u64 = @intCast(pq_add_start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // --- IVFFlat for comparison ---
    var ivf = try faiss.IndexIVFFlat.init(alloc, D, .{
        .nlist = NLIST,
        .nprobe = NPROBE,
        .n_iter = 10,
        .seed = 42,
    });
    defer ivf.deinit();
    try ivf.train(dataset);
    i = 0;
    while (i < N) : (i += 1) {
        _ = try ivf.add(dataset[i * D ..][0..D]);
    }

    // --- Flat reference ---
    var flat = faiss.FlatIndex.init(alloc, D, .l2_squared);
    defer flat.deinit();
    i = 0;
    while (i < N) : (i += 1) {
        _ = try flat.add(dataset[i * D ..][0..D]);
    }

    // --- queries ---
    const queries = try alloc.alloc(f32, Q * D);
    defer alloc.free(queries);
    for (queries) |*v| v.* = rng.float(f32);

    // warmup
    {
        const w = try ivfpq.search(alloc, queries[0..D], K);
        defer alloc.free(w);
    }

    // --- IVFPQ query phase ---
    const pq_qs = std.Io.Clock.awake.now(io);
    var pq_chk: u64 = 0;
    var qi: usize = 0;
    while (qi < Q) : (qi += 1) {
        const q = queries[qi * D ..][0..D];
        const hits = try ivfpq.search(alloc, q, K);
        defer alloc.free(hits);
        if (hits.len > 0) pq_chk += hits[0].id;
    }
    const pq_ns: u64 = @intCast(pq_qs.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // --- IVF query phase ---
    const ivf_qs = std.Io.Clock.awake.now(io);
    var ivf_chk: u64 = 0;
    qi = 0;
    while (qi < Q) : (qi += 1) {
        const q = queries[qi * D ..][0..D];
        const hits = try ivf.search(alloc, q, K);
        defer alloc.free(hits);
        if (hits.len > 0) ivf_chk += hits[0].id;
    }
    const ivf_ns: u64 = @intCast(ivf_qs.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // --- Flat query phase ---
    const flat_qs = std.Io.Clock.awake.now(io);
    var flat_chk: u64 = 0;
    qi = 0;
    while (qi < Q) : (qi += 1) {
        const q = queries[qi * D ..][0..D];
        const hits = try flat.search(alloc, q, K);
        defer alloc.free(hits);
        flat_chk += hits[0].id;
    }
    const flat_ns: u64 = @intCast(flat_qs.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // --- recall: PQ vs Flat (Flat is ground truth) ---
    var pq_recall_sum: f32 = 0;
    var ivf_recall_sum: f32 = 0;
    qi = 0;
    while (qi < Q) : (qi += 1) {
        const q = queries[qi * D ..][0..D];
        const pq_hits = try ivfpq.search(alloc, q, K);
        defer alloc.free(pq_hits);
        const ivf_hits = try ivf.search(alloc, q, K);
        defer alloc.free(ivf_hits);
        const flat_hits = try flat.search(alloc, q, K);
        defer alloc.free(flat_hits);

        var pq_set = std.AutoHashMap(u32, void).init(alloc);
        defer pq_set.deinit();
        for (pq_hits) |h| try pq_set.put(h.id, {});
        var ivf_set = std.AutoHashMap(u32, void).init(alloc);
        defer ivf_set.deinit();
        for (ivf_hits) |h| try ivf_set.put(h.id, {});

        var pq_matches: usize = 0;
        var ivf_matches: usize = 0;
        for (flat_hits) |h| {
            if (pq_set.contains(h.id)) pq_matches += 1;
            if (ivf_set.contains(h.id)) ivf_matches += 1;
        }
        pq_recall_sum += @as(f32, @floatFromInt(pq_matches)) / @as(f32, @floatFromInt(K));
        ivf_recall_sum += @as(f32, @floatFromInt(ivf_matches)) / @as(f32, @floatFromInt(K));
    }
    const pq_recall = pq_recall_sum / @as(f32, @floatFromInt(Q));
    const ivf_recall = ivf_recall_sum / @as(f32, @floatFromInt(Q));

    // --- memory footprint ---
    const pq_bytes = ivfpq.memoryBytes();
    const flat_storage_bytes = N * D * @sizeOf(f32);

    std.debug.print("=== faiss-zig IVFPQ vs IVFFlat vs Flat (N={d}, D={d}, K={d}, Q={d}, nlist={d}, nprobe={d}, pq_m={d}) ===\n", .{ N, D, K, Q, NLIST, NPROBE, PQ_M });

    std.debug.print("\nBuild phase (IVFPQ):\n", .{});
    std.debug.print("  train_ms             = {d}\n", .{pq_train_ns / 1_000_000});
    std.debug.print("  add_ms               = {d}\n", .{pq_add_ns / 1_000_000});
    std.debug.print("  ns_per_add           = {d}\n", .{pq_add_ns / N});

    std.debug.print("\nQuery phase:\n", .{});
    std.debug.print("  IVFPQ ns_per_query   = {d}\n", .{pq_ns / Q});
    std.debug.print("  IVFPQ us_per_query   = {d}\n", .{pq_ns / Q / 1000});
    std.debug.print("  IVFPQ q/sec          = {d}\n", .{@as(u64, Q) * std.time.ns_per_s / pq_ns});
    std.debug.print("\n  IVFFlat ns_per_query = {d}\n", .{ivf_ns / Q});
    std.debug.print("  IVFFlat us_per_query = {d}\n", .{ivf_ns / Q / 1000});
    std.debug.print("  IVFFlat q/sec        = {d}\n", .{@as(u64, Q) * std.time.ns_per_s / ivf_ns});
    std.debug.print("\n  Flat ns_per_query    = {d}\n", .{flat_ns / Q});
    std.debug.print("  Flat us_per_query    = {d}\n", .{flat_ns / Q / 1000});
    std.debug.print("  Flat q/sec           = {d}\n", .{@as(u64, Q) * std.time.ns_per_s / flat_ns});

    std.debug.print("\nSpeedup ratios:\n", .{});
    std.debug.print("  IVFPQ vs Flat        = {d:.2}x\n", .{@as(f64, @floatFromInt(flat_ns)) / @as(f64, @floatFromInt(pq_ns))});
    std.debug.print("  IVFPQ vs IVFFlat     = {d:.2}x\n", .{@as(f64, @floatFromInt(ivf_ns)) / @as(f64, @floatFromInt(pq_ns))});
    std.debug.print("  IVFFlat vs Flat      = {d:.2}x\n", .{@as(f64, @floatFromInt(flat_ns)) / @as(f64, @floatFromInt(ivf_ns))});

    std.debug.print("\nRecall (vs Flat as ground truth, top-{d}):\n", .{K});
    std.debug.print("  IVFPQ recall         = {d:.3}\n", .{pq_recall});
    std.debug.print("  IVFFlat recall       = {d:.3}\n", .{ivf_recall});

    std.debug.print("\nMemory footprint:\n", .{});
    std.debug.print("  IVFPQ total bytes    = {d} ({d:.1} KB)\n", .{ pq_bytes, @as(f64, @floatFromInt(pq_bytes)) / 1024.0 });
    std.debug.print("  Flat storage bytes   = {d} ({d:.1} KB)\n", .{ flat_storage_bytes, @as(f64, @floatFromInt(flat_storage_bytes)) / 1024.0 });
    std.debug.print("  IVFPQ compression    = {d:.2}x\n", .{@as(f64, @floatFromInt(flat_storage_bytes)) / @as(f64, @floatFromInt(pq_bytes))});
    const per_vec_bytes_pq = (PQ_M + @sizeOf(u32));
    const per_vec_bytes_raw = D * @sizeOf(f32);
    std.debug.print("  per-vec PQ           = {d} bytes (code {d} + id 4)\n", .{ per_vec_bytes_pq, PQ_M });
    std.debug.print("  per-vec raw f32      = {d} bytes\n", .{per_vec_bytes_raw});
    std.debug.print("  per-vec compression  = {d:.2}x\n", .{@as(f64, @floatFromInt(per_vec_bytes_raw)) / @as(f64, @floatFromInt(per_vec_bytes_pq))});

    // anti-DCE
    std.debug.print("\n(checksums: pq={d} ivf={d} flat={d})\n", .{ pq_chk, ivf_chk, flat_chk });
}
