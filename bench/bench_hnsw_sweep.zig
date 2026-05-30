//! HNSW recall-vs-QPS Pareto sweep — varies ef_search across
//! {50, 100, 200, 400, 800} on a single built HNSW index, reports
//! recall@10 and queries/sec per ef_search value.
//!
//! Closes the recall-matched comparison cell that bench_hnsw.zig alone
//! couldn't address. Builds the index once with ef_construction=200,
//! then mutates self.config.ef_search between query batches per the
//! IndexHNSW design (see src/hnsw.zig:345 where ef_search is read
//! live from self.config).
//!
//! Same fixture as bench_hnsw.zig / bench_hnsw_vs_faiss_cpu.py:
//! N=10000 D=128 K=10 Q=100 seed=42. Direct comparable to faiss-cpu
//! HNSW recall 0.882 / QPS 836 measured 2026-05-28 on this hardware.

const std = @import("std");
const faiss = @import("faiss");

const N: usize = 10_000;
const D: u32 = 128;
const K: usize = 10;
const Q: usize = 100;

const EF_SWEEP = [_]u32{ 50, 100, 200, 400, 800 };

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    // Build HNSW once. ef_construction is the build-side parameter
    // (controls graph quality) and is held fixed at 200.
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

    // Flat reference for recall computation
    var flat = faiss.FlatIndex.init(alloc, D, .l2_squared);
    defer flat.deinit();
    var prng2: std.Random.DefaultPrng = .init(42);
    const rng2 = prng2.random();
    var ii: usize = 0;
    while (ii < N) : (ii += 1) {
        for (vec_buf) |*v| v.* = rng2.float(f32);
        _ = try flat.add(vec_buf);
    }

    // Queries — same vectors reused across every ef_search value
    const queries = try alloc.alloc(f32, Q * D);
    defer alloc.free(queries);
    for (queries) |*v| v.* = rng.float(f32);

    std.debug.print("=== faiss-zig HNSW ef_search sweep (N={d} D={d} K={d} Q={d}) ===\n", .{ N, D, K, Q });
    std.debug.print("build_ms = {d}  (ef_construction=200, M=16)\n\n", .{build_ns / 1_000_000});
    std.debug.print("ef_search | qps    | us/query | recall@{d}\n", .{K});
    std.debug.print("----------+--------+----------+----------\n", .{});

    // For each ef_search value: mutate config, warmup, time QPS, compute recall
    for (EF_SWEEP) |ef| {
        idx.config.ef_search = ef;

        // Warmup at this ef
        {
            const w = try idx.search(alloc, queries[0..D], K);
            alloc.free(w);
        }

        // Timed query phase
        const qs = std.Io.Clock.awake.now(io);
        var qi: usize = 0;
        var checksum: u64 = 0;
        while (qi < Q) : (qi += 1) {
            const q = queries[qi * D ..][0..D];
            const hits = try idx.search(alloc, q, K);
            defer alloc.free(hits);
            checksum += hits[0].id;
        }
        const total_ns: u64 = @intCast(qs.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

        // Recall vs Flat
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

        const qps = @as(u64, Q) * std.time.ns_per_s / total_ns;
        const us_per_query = total_ns / Q / 1000;
        std.debug.print("{d:9} | {d:6} | {d:8} | {d:.3}  (cksum {d})\n", .{ ef, qps, us_per_query, recall, checksum });
    }
}
