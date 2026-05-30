//! Standalone Flat L2-squared bench with explicit timing + verification.
//!
//! Direct measurement of brute-force L2-squared search at N ranging
//! from L3-resident to DRAM-bound. Bypasses the FlatIndex API to
//! eliminate any silent-truncation bugs in the bench_flat.zig path
//! (the existing bench reported constant 1298 q/s across N=10K-1M
//! which is impossible — caught as Type-II via wallclock check).
//!
//! Also reports actual bytes scanned per query so the bandwidth
//! sanity check is explicit.

const std = @import("std");

const D: comptime_int = 128;
const Q: comptime_int = 100;
const K: comptime_int = 10;
const LANE: comptime_int = 8;
const V = @Vector(LANE, f32);

const Hit = struct { id: u32, score: f32 };
fn hitLessL2(_: void, x: Hit, y: Hit) bool {
    return x.score < y.score;
}

inline fn l2SquaredD128(a: *const [D]f32, b: *const [D]f32) f32 {
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

fn searchFlat(
    alloc: std.mem.Allocator,
    vectors: []const [D]f32,
    ids: []const u32,
    query: *const [D]f32,
    hits_buf: []Hit,
) ![]Hit {
    const n = vectors.len;
    for (vectors, ids, 0..) |*v, id, i| {
        hits_buf[i] = .{ .id = id, .score = l2SquaredD128(query, v) };
    }
    _ = alloc;
    std.sort.pdq(Hit, hits_buf[0..n], {}, hitLessL2);
    return hits_buf[0..K];
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer iter.deinit();
    _ = iter.next();
    const N = try std.fmt.parseInt(usize, iter.next() orelse "10000", 10);

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

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
    const hits_buf = try alloc.alloc(Hit, N);
    defer alloc.free(hits_buf);

    // Warmup
    _ = try searchFlat(alloc, vectors, ids, &queries[0], hits_buf);

    const t0 = std.Io.Clock.awake.now(io);
    var checksum: u64 = 0;
    for (queries) |*q| {
        const hits = try searchFlat(alloc, vectors, ids, q, hits_buf);
        checksum += hits[0].id;
    }
    const total_ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const ns_per_query = total_ns / Q;
    const qps = @as(u64, Q) * std.time.ns_per_s / total_ns;

    // Bandwidth sanity check
    const bytes_per_query = N * D * @sizeOf(f32);
    const bytes_per_second = bytes_per_query * qps;
    const gbps = @as(f64, @floatFromInt(bytes_per_second)) / 1_000_000_000.0;

    std.debug.print("N             = {d}\n", .{N});
    std.debug.print("corpus_MB     = {d}\n", .{N * D * @sizeOf(f32) / 1024 / 1024});
    std.debug.print("ns_per_query  = {d}\n", .{ns_per_query});
    std.debug.print("us_per_query  = {d}\n", .{ns_per_query / 1000});
    std.debug.print("queries_per_s = {d}\n", .{qps});
    std.debug.print("effective_GBps_scan = {d:.2}\n", .{gbps});
    std.debug.print("(cksum {d})\n", .{checksum});
}
