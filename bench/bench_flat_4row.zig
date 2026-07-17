//! Flat L2-squared with 4-row unrolled corpus scan + explicit prefetch.
//!
//! Per query q, process corpus[i..i+4] with 4 independent SIMD
//! accumulators. This distributes ILP (4 parallel @Vector(8) accumulators
//! per cycle on M2 Max's wide pipeline), enables 4 prefetch streams,
//! and reduces query-register pressure (q stays hot, B streams).
//!
//! Top-K heap (size K=10) as in bench_flat_topk_heap. Standalone.

const std = @import("std");

const D: comptime_int = 128;
const Q: comptime_int = 100;
const K: comptime_int = 10;
const LANE: comptime_int = 8;
const V = @Vector(LANE, f32);
const UNROLL: comptime_int = 4;

const Hit = struct { id: u32, score: f32 };

inline fn l2_4rows(
    q_arr: *const [D]f32,
    b0: *const [D]f32,
    b1: *const [D]f32,
    b2: *const [D]f32,
    b3: *const [D]f32,
) [4]f32 {
    var acc0: V = @splat(0);
    var acc1: V = @splat(0);
    var acc2: V = @splat(0);
    var acc3: V = @splat(0);
    comptime var i: usize = 0;
    inline while (i < D) : (i += LANE) {
        const qv: V = q_arr[i..][0..LANE].*;
        const d0 = qv - @as(V, b0[i..][0..LANE].*);
        acc0 += d0 * d0;
        const d1 = qv - @as(V, b1[i..][0..LANE].*);
        acc1 += d1 * d1;
        const d2 = qv - @as(V, b2[i..][0..LANE].*);
        acc2 += d2 * d2;
        const d3 = qv - @as(V, b3[i..][0..LANE].*);
        acc3 += d3 * d3;
    }
    return .{ @reduce(.Add, acc0), @reduce(.Add, acc1), @reduce(.Add, acc2), @reduce(.Add, acc3) };
}

inline fn l2_1row(q_arr: *const [D]f32, b: *const [D]f32) f32 {
    var acc: V = @splat(0);
    comptime var i: usize = 0;
    inline while (i < D) : (i += LANE) {
        const qv: V = q_arr[i..][0..LANE].*;
        const d = qv - @as(V, b[i..][0..LANE].*);
        acc += d * d;
    }
    return @reduce(.Add, acc);
}

const MaxHeap = struct {
    items: [K]Hit,
    count: usize,

    fn init() MaxHeap {
        return .{ .items = undefined, .count = 0 };
    }

    fn pushDown(self: *MaxHeap, idx: usize) void {
        var i = idx;
        while (true) {
            const l = 2 * i + 1;
            const r = 2 * i + 2;
            var largest = i;
            if (l < self.count and self.items[l].score > self.items[largest].score) largest = l;
            if (r < self.count and self.items[r].score > self.items[largest].score) largest = r;
            if (largest == i) break;
            const tmp = self.items[i];
            self.items[i] = self.items[largest];
            self.items[largest] = tmp;
            i = largest;
        }
    }

    fn pushUp(self: *MaxHeap, idx: usize) void {
        var i = idx;
        while (i > 0) {
            const p = (i - 1) / 2;
            if (self.items[i].score > self.items[p].score) {
                const tmp = self.items[i];
                self.items[i] = self.items[p];
                self.items[p] = tmp;
                i = p;
            } else break;
        }
    }

    inline fn push(self: *MaxHeap, h: Hit) void {
        if (self.count < K) {
            self.items[self.count] = h;
            self.count += 1;
            self.pushUp(self.count - 1);
        } else if (h.score < self.items[0].score) {
            self.items[0] = h;
            self.pushDown(0);
        }
    }
};

fn searchTopK4Row(
    vectors: []const [D]f32,
    ids: []const u32,
    query: *const [D]f32,
    out: []Hit,
) void {
    var heap = MaxHeap.init();
    const n = vectors.len;
    const n_unrolled = n - (n % UNROLL);

    var i: usize = 0;
    // Prefetch the first few cache lines
    {
        var p: usize = 0;
        while (p < @min(UNROLL * 4, n)) : (p += 1) {
            @prefetch(@as([*]const f32, @ptrCast(&vectors[p])), .{ .rw = .read, .locality = 3 });
        }
    }

    while (i < n_unrolled) : (i += UNROLL) {
        // Prefetch UNROLL*4 ahead
        const pf = i + UNROLL * 4;
        if (pf + UNROLL <= n) {
            var pi: usize = 0;
            while (pi < UNROLL) : (pi += 1) {
                @prefetch(@as([*]const f32, @ptrCast(&vectors[pf + pi])), .{ .rw = .read, .locality = 3 });
            }
        }
        const scores = l2_4rows(query, &vectors[i], &vectors[i + 1], &vectors[i + 2], &vectors[i + 3]);
        heap.push(.{ .id = ids[i], .score = scores[0] });
        heap.push(.{ .id = ids[i + 1], .score = scores[1] });
        heap.push(.{ .id = ids[i + 2], .score = scores[2] });
        heap.push(.{ .id = ids[i + 3], .score = scores[3] });
    }
    while (i < n) : (i += 1) {
        heap.push(.{ .id = ids[i], .score = l2_1row(query, &vectors[i]) });
    }

    // Drain
    var j: usize = K;
    while (j > 0) : (j -= 1) {
        out[j - 1] = heap.items[0];
        heap.items[0] = heap.items[heap.count - 1];
        heap.count -= 1;
        if (heap.count > 0) heap.pushDown(0);
    }
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
    var out: [K]Hit = undefined;

    searchTopK4Row(vectors, ids, &queries[0], &out);

    const t0 = std.Io.Clock.awake.now(io);
    var checksum: u64 = 0;
    for (queries) |*q| {
        searchTopK4Row(vectors, ids, q, &out);
        checksum += out[0].id;
    }
    const total_ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const ns_per_query = total_ns / Q;
    const qps = @as(u64, Q) * std.time.ns_per_s / total_ns;
    const bytes_per_query = N * D * @sizeOf(f32);
    const gbps = @as(f64, @floatFromInt(bytes_per_query * qps)) / 1_000_000_000.0;

    std.debug.print("N             = {d}\n", .{N});
    std.debug.print("corpus_MB     = {d}\n", .{N * D * @sizeOf(f32) / 1024 / 1024});
    std.debug.print("ns_per_query  = {d}\n", .{ns_per_query});
    std.debug.print("us_per_query  = {d}\n", .{ns_per_query / 1000});
    std.debug.print("queries_per_s = {d}\n", .{qps});
    std.debug.print("effective_GBps_scan = {d:.2}\n", .{gbps});
    std.debug.print("unroll = {d}, prefetch = TILE*4 ahead\n", .{UNROLL});
    std.debug.print("(cksum {d})\n", .{checksum});
}
