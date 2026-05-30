//! Flat-search throughput bench with comptime-specialized D=128
//! L2-squared distance.
//!
//! The src/root.zig `l2SquaredDistance` uses @Vector(8, f32) with a
//! scalar tail loop bounded at runtime `a.len % LANE`. For D=128 the
//! tail loop never executes (128 % 8 == 0) but the runtime check
//! remains in the binary.
//!
//! This bench replaces that with a comptime-known D=128 distance
//! function: exactly 16 SIMD iterations, no tail check, fully
//! amenable to LLVM unrolling. Compares against the library's
//! general FlatIndex.search at the same N=10000 D=128 K=10 Q=100
//! workload as bench_flat.zig (baseline 75 q/s on Ice Lake).
//!
//! Pre-reg target: ≥1.0x parity with faiss-cpu single-thread (187
//! q/s on this hardware). Falsifier: <0.7x of faiss-cpu.
//!
//! This isolates the comptime-spec lever from the framework-tax
//! cell to test whether removing runtime D handling moves the
//! single-thread throughput cell.

const std = @import("std");
const faiss = @import("faiss");

const N: usize = 10_000;
const D: comptime_int = 128;
const K: usize = 10;
const Q: usize = 100;
const LANE: comptime_int = 8;
const V = @Vector(LANE, f32);

/// Comptime-specialized L2-squared for fixed D=128. The compiler can
/// statically schedule the 16 SIMD iterations and emit straight-line
/// AVX2 / AVX-512 code with no loop overhead.
fn l2SquaredD128(a: *const [D]f32, b: *const [D]f32) f32 {
    var sum_vec: V = @splat(0.0);
    comptime var i: usize = 0;
    inline while (i < D) : (i += LANE) {
        const av: V = a[i..][0..LANE].*;
        const bv: V = b[i..][0..LANE].*;
        const d = av - bv;
        sum_vec += d * d;
    }
    return @reduce(.Add, sum_vec);
}

const Hit = faiss.Hit;

fn hitLessL2(_: void, x: Hit, y: Hit) bool {
    return x.score < y.score;
}

/// Open-coded Flat search with comptime-D=128 distance inlined into
/// the inner loop. No std.ArrayList for vectors — caller passes
/// pre-built array of shape [N][D].
fn searchD128(
    alloc: std.mem.Allocator,
    vectors: []const [D]f32,
    ids: []const u32,
    query: *const [D]f32,
    k: usize,
) ![]Hit {
    const n = vectors.len;
    var all_hits = try alloc.alloc(Hit, n);
    defer alloc.free(all_hits);
    for (vectors, ids, 0..) |*v, id, i| {
        all_hits[i] = .{ .id = id, .score = l2SquaredD128(query, v) };
    }
    std.sort.pdq(Hit, all_hits, {}, hitLessL2);
    const out = try alloc.alloc(Hit, k);
    @memcpy(out, all_hits[0..k]);
    return out;
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    // Generate the corpus + queries up front (out of timing).
    const vectors = try alloc.alloc([D]f32, N);
    defer alloc.free(vectors);
    const ids = try alloc.alloc(u32, N);
    defer alloc.free(ids);
    for (vectors, ids, 0..) |*v, *id, i| {
        for (v) |*x| x.* = rng.float(f32);
        id.* = @intCast(i);
    }
    const queries = try alloc.alloc([D]f32, Q);
    defer alloc.free(queries);
    for (queries) |*q| {
        for (q) |*x| x.* = rng.float(f32);
    }

    // === Path A: comptime-D=128 search ===
    {
        // Warmup
        const w = try searchD128(alloc, vectors, ids, &queries[0], K);
        alloc.free(w);

        const t0 = std.Io.Clock.awake.now(io);
        var checksum: u64 = 0;
        for (queries) |*q| {
            const hits = try searchD128(alloc, vectors, ids, q, K);
            defer alloc.free(hits);
            checksum += hits[0].id;
        }
        const total_ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

        const ns_per_query = total_ns / Q;
        const qps = @as(u64, Q) * std.time.ns_per_s / total_ns;
        std.debug.print("=== faiss-zig Flat search, comptime-D=128 distance (N={d} D={d} K={d} Q={d}) ===\n", .{ N, D, K, Q });
        std.debug.print("comptime_d128\n", .{});
        std.debug.print("  ns_per_query    = {d}\n", .{ns_per_query});
        std.debug.print("  us_per_query    = {d}\n", .{ns_per_query / 1000});
        std.debug.print("  queries_per_sec = {d}\n", .{qps});
        std.debug.print("  (cksum {d})\n\n", .{checksum});
    }

    // === Path B: library FlatIndex.search (runtime-D) ===
    {
        var idx = faiss.FlatIndex.init(alloc, D, .l2_squared);
        defer idx.deinit();
        const vec_flat = try alloc.alloc(f32, D);
        defer alloc.free(vec_flat);
        for (vectors) |*v| {
            @memcpy(vec_flat, v[0..D]);
            _ = try idx.add(vec_flat);
        }

        // Warmup
        const w = try idx.search(alloc, queries[0][0..D], K);
        alloc.free(w);

        const t0 = std.Io.Clock.awake.now(io);
        var checksum: u64 = 0;
        for (queries) |*q| {
            const hits = try idx.search(alloc, q[0..D], K);
            defer alloc.free(hits);
            checksum += hits[0].id;
        }
        const total_ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

        const ns_per_query = total_ns / Q;
        const qps = @as(u64, Q) * std.time.ns_per_s / total_ns;
        std.debug.print("library_runtime_d\n", .{});
        std.debug.print("  ns_per_query    = {d}\n", .{ns_per_query});
        std.debug.print("  us_per_query    = {d}\n", .{ns_per_query / 1000});
        std.debug.print("  queries_per_sec = {d}\n", .{qps});
        std.debug.print("  (cksum {d})\n\n", .{checksum});
    }

    std.debug.print("Reference (faiss-cpu single-thread): 187 q/s on this hardware\n", .{});
    std.debug.print("Reference (faiss-cpu T=8):           417 q/s on this hardware\n", .{});
}
