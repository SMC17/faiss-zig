//! FlatIndex kNN throughput bench.
//!
//! Builds NxD random vectors, runs Q kNN queries, reports queries/sec
//! and ns/query. The benchmark is single-threaded scalar; SIMD
//! acceleration is v0.1+ work.
//!
//! Default config: N=10000, D=128, K=10, Q=100. Tunable via the
//! command line in the same order as examples/search.zig.
//!
//! Self-bench only — no faiss-cpp comparison yet. v0.1 will stage
//! the C++ faiss head-to-head similar to safetensors-zig's
//! bench/bench_rust.

const std = @import("std");
const faiss = @import("faiss");

const DEFAULT_N: usize = 10_000;
const DEFAULT_D: u32 = 128;
const DEFAULT_K: usize = 10;
const DEFAULT_Q: usize = 100;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer iter.deinit();
    _ = iter.next();
    const n = try std.fmt.parseInt(usize, iter.next() orelse "10000", 10);
    const d = try std.fmt.parseInt(u32, iter.next() orelse "128", 10);
    const k = try std.fmt.parseInt(usize, iter.next() orelse "10", 10);
    const q = try std.fmt.parseInt(usize, iter.next() orelse "100", 10);

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    var idx = faiss.FlatIndex.init(alloc, d, .l2_squared);
    defer idx.deinit();

    const vec_buf = try alloc.alloc(f32, d);
    defer alloc.free(vec_buf);

    // Build phase
    const build_start = std.Io.Clock.awake.now(io);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        for (vec_buf) |*v| v.* = rng.float(f32);
        _ = try idx.add(vec_buf);
    }
    const build_end = std.Io.Clock.awake.now(io);
    const build_ns: u64 = @intCast(build_start.durationTo(build_end).nanoseconds);

    // Warmup
    for (vec_buf) |*v| v.* = rng.float(f32);
    {
        const w = try idx.search(alloc, vec_buf, k);
        defer alloc.free(w);
    }

    // Query phase
    const query_start = std.Io.Clock.awake.now(io);
    var qi: usize = 0;
    while (qi < q) : (qi += 1) {
        for (vec_buf) |*v| v.* = rng.float(f32);
        const hits = try idx.search(alloc, vec_buf, k);
        alloc.free(hits);
    }
    const query_end = std.Io.Clock.awake.now(io);
    const query_ns: u64 = @intCast(query_start.durationTo(query_end).nanoseconds);

    const ns_per_query = query_ns / q;
    const queries_per_sec = @as(u64, q) * std.time.ns_per_s / query_ns;

    std.debug.print("config\n", .{});
    std.debug.print("  N      = {d}\n", .{n});
    std.debug.print("  D      = {d}\n", .{d});
    std.debug.print("  K      = {d}\n", .{k});
    std.debug.print("  Q      = {d}\n", .{q});
    std.debug.print("build\n", .{});
    std.debug.print("  total_ms = {d}\n", .{build_ns / 1_000_000});
    std.debug.print("  ns_per_add = {d}\n", .{build_ns / n});
    std.debug.print("query\n", .{});
    std.debug.print("  ns_per_query    = {d}\n", .{ns_per_query});
    std.debug.print("  us_per_query    = {d}\n", .{ns_per_query / 1000});
    std.debug.print("  queries_per_sec = {d}\n", .{queries_per_sec});
    std.debug.print("  vectors_per_sec = {d}\n", .{queries_per_sec * n});
}
