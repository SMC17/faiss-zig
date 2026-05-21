//! Multi-thread scaling bench for FlatIndex.searchBatch.
//!
//! Same workload as bench_flat (N=10K vectors of D=128, K=10) but
//! with Q=100 queries dispatched across T ∈ {1, 2, 4, 8} threads.
//!
//! Reports per-query throughput at each thread count + scaling
//! efficiency (1-thread baseline / actual).
//!
//! Compare against bench_faiss_cpu.py for an apples-to-apples
//! head-to-head with the C++ multi-thread upstream.

const std = @import("std");
const faiss = @import("faiss");

const N: usize = 10_000;
const D: u32 = 128;
const K: usize = 10;
const Q: usize = 100;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    var idx = faiss.FlatIndex.init(alloc, D, .l2_squared);
    defer idx.deinit();

    const vec_buf = try alloc.alloc(f32, D);
    defer alloc.free(vec_buf);

    var i: usize = 0;
    while (i < N) : (i += 1) {
        for (vec_buf) |*v| v.* = rng.float(f32);
        _ = try idx.add(vec_buf);
    }

    // Pre-allocate queries
    const query_storage = try alloc.alloc(f32, Q * D);
    defer alloc.free(query_storage);
    for (query_storage) |*v| v.* = rng.float(f32);

    const queries = try alloc.alloc([]const f32, Q);
    defer alloc.free(queries);
    for (queries, 0..) |*q, qi| {
        q.* = query_storage[qi * D ..][0..D];
    }

    const thread_counts = [_]usize{ 1, 2, 4, 8 };
    var single_thread_ns: u64 = 0;

    std.debug.print("threads | total_ms | ns/query | queries/sec | scaling\n", .{});
    std.debug.print("--------+----------+----------+-------------+--------\n", .{});

    for (thread_counts) |T| {
        // Warmup
        {
            const hits = try idx.searchBatch(alloc, queries, K, T);
            defer alloc.free(hits);
            for (hits) |h| alloc.free(h);
        }

        const t0 = std.Io.Clock.awake.now(io);
        const results = try idx.searchBatch(alloc, queries, K, T);
        const t1 = std.Io.Clock.awake.now(io);
        const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);

        for (results) |r| alloc.free(r);
        alloc.free(results);

        if (T == 1) single_thread_ns = ns;
        const scaling: f64 = @as(f64, @floatFromInt(single_thread_ns)) /
            @as(f64, @floatFromInt(ns));
        const ns_per_query = ns / Q;
        const qps = @as(u64, Q) * std.time.ns_per_s / ns;

        std.debug.print("{d:>7} | {d:>8} | {d:>8} | {d:>11} | {d:.2}x\n", .{
            T,
            ns / 1_000_000,
            ns_per_query,
            qps,
            scaling,
        });
    }
}
