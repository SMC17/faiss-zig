//! IVFFlat bench: train + add + query throughput at N=10000 D=128
//! nlist=100 nprobe=10, head-to-head with FlatIndex for recall + speedup.

const std = @import("std");
const faiss = @import("faiss");

const N: usize = 10_000;
const D: u32 = 128;
const K: usize = 10;
const Q: usize = 100;
const NLIST: u32 = 100;
const NPROBE: u32 = 10;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    // --- Generate dataset once, share across IVF + Flat for fair recall ---
    const dataset = try alloc.alloc(f32, N * D);
    defer alloc.free(dataset);
    for (dataset) |*v| v.* = rng.float(f32);

    // --- Build IVF ---
    var idx = try faiss.IndexIVFFlat.init(alloc, D, .{
        .nlist = NLIST,
        .nprobe = NPROBE,
        .n_iter = 10,
        .seed = 42,
    });
    defer idx.deinit();

    const train_start = std.Io.Clock.awake.now(io);
    try idx.train(dataset);
    const train_end = std.Io.Clock.awake.now(io);
    const train_ns: u64 = @intCast(train_start.durationTo(train_end).nanoseconds);

    const add_start = std.Io.Clock.awake.now(io);
    var i: usize = 0;
    while (i < N) : (i += 1) {
        _ = try idx.add(dataset[i * D ..][0..D]);
    }
    const add_end = std.Io.Clock.awake.now(io);
    const add_ns: u64 = @intCast(add_start.durationTo(add_end).nanoseconds);

    // --- Flat reference ---
    var flat = faiss.FlatIndex.init(alloc, D, .l2_squared);
    defer flat.deinit();
    i = 0;
    while (i < N) : (i += 1) {
        _ = try flat.add(dataset[i * D ..][0..D]);
    }

    // --- Queries ---
    const queries = try alloc.alloc(f32, Q * D);
    defer alloc.free(queries);
    for (queries) |*v| v.* = rng.float(f32);

    // Warmup
    {
        const w = try idx.search(alloc, queries[0..D], K);
        defer alloc.free(w);
    }

    // --- IVF query phase ---
    const ivf_qs = std.Io.Clock.awake.now(io);
    var ivf_top1: u64 = 0;
    var qi: usize = 0;
    while (qi < Q) : (qi += 1) {
        const q = queries[qi * D ..][0..D];
        const hits = try idx.search(alloc, q, K);
        defer alloc.free(hits);
        if (hits.len > 0) ivf_top1 += hits[0].id;
    }
    const ivf_ns: u64 = @intCast(ivf_qs.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // --- Flat query phase ---
    const flat_qs = std.Io.Clock.awake.now(io);
    var flat_top1: u64 = 0;
    qi = 0;
    while (qi < Q) : (qi += 1) {
        const q = queries[qi * D ..][0..D];
        const hits = try flat.search(alloc, q, K);
        defer alloc.free(hits);
        flat_top1 += hits[0].id;
    }
    const flat_ns: u64 = @intCast(flat_qs.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // --- Recall ---
    var total_recall: f32 = 0;
    qi = 0;
    while (qi < Q) : (qi += 1) {
        const q = queries[qi * D ..][0..D];
        const ivf_hits = try idx.search(alloc, q, K);
        defer alloc.free(ivf_hits);
        const flat_hits = try flat.search(alloc, q, K);
        defer alloc.free(flat_hits);
        var h_set = std.AutoHashMap(u32, void).init(alloc);
        defer h_set.deinit();
        for (ivf_hits) |h| try h_set.put(h.id, {});
        var matches: usize = 0;
        for (flat_hits) |h| {
            if (h_set.contains(h.id)) matches += 1;
        }
        total_recall += @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(K));
    }
    const recall = total_recall / @as(f32, @floatFromInt(Q));

    std.debug.print("=== faiss-zig IVFFlat vs Flat (N={d}, D={d}, K={d}, Q={d}, nlist={d}, nprobe={d}) ===\n", .{ N, D, K, Q, NLIST, NPROBE });
    std.debug.print("\nBuild phase:\n", .{});
    std.debug.print("  IVF train_ms         = {d}\n", .{train_ns / 1_000_000});
    std.debug.print("  IVF add_ms           = {d}\n", .{add_ns / 1_000_000});
    std.debug.print("  IVF ns_per_add       = {d}\n", .{add_ns / N});
    std.debug.print("\nQuery phase:\n", .{});
    std.debug.print("  IVF ns_per_query     = {d}\n", .{ivf_ns / Q});
    std.debug.print("  IVF us_per_query     = {d}\n", .{ivf_ns / Q / 1000});
    std.debug.print("  IVF queries_per_sec  = {d}\n", .{@as(u64, Q) * std.time.ns_per_s / ivf_ns});
    std.debug.print("\n  Flat ns_per_query    = {d}\n", .{flat_ns / Q});
    std.debug.print("  Flat us_per_query    = {d}\n", .{flat_ns / Q / 1000});
    std.debug.print("  Flat queries_per_sec = {d}\n", .{@as(u64, Q) * std.time.ns_per_s / flat_ns});
    std.debug.print("\n  IVF speedup vs Flat  = {d:.2}x\n", .{@as(f64, @floatFromInt(flat_ns)) / @as(f64, @floatFromInt(ivf_ns))});
    std.debug.print("  IVF top-{d} recall    = {d:.3}\n", .{ K, recall });

    // Anti-DCE
    std.debug.print("\n(checksums: ivf_top1_sum={d} flat_top1_sum={d})\n", .{ ivf_top1, flat_top1 });
}
