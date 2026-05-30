//! Flat L2-squared search with top-K partial-sort heap.
//!
//! The full std.sort.pdq path used in bench_flat_standalone is
//! O(N log N) which dominates at large N. faiss-cpu uses a top-K
//! max-heap of size K and only updates when a distance beats the
//! current max — that's O(N log K) inside the loop, much faster.
//!
//! Hypothesis: top-K heap closes the 41.8x gap to faiss-cpu at N=10K
//! and the 99x gap at N=1M by an order of magnitude or more.

const std = @import("std");

const D: comptime_int = 128;
const Q: comptime_int = 100;
const K: comptime_int = 10;
const LANE: comptime_int = 8;
const V = @Vector(LANE, f32);

const Hit = struct { id: u32, score: f32 };

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

// Max-heap of size K rooted at index 0; the largest score is at the root.
// When we encounter a new score < root, we pop root and insert the new.
// Final K elements are the top-K smallest by score (best matches).
const MaxHeap = struct {
    items: [K]Hit,
    count: usize,

    fn init() MaxHeap {
        return .{ .items = undefined, .count = 0 };
    }

    fn parent(i: usize) usize { return (i - 1) / 2; }
    fn left(i: usize) usize { return 2 * i + 1; }
    fn right(i: usize) usize { return 2 * i + 2; }

    fn pushUp(self: *MaxHeap, idx: usize) void {
        var i = idx;
        while (i > 0) {
            const p = parent(i);
            if (self.items[i].score > self.items[p].score) {
                const tmp = self.items[i];
                self.items[i] = self.items[p];
                self.items[p] = tmp;
                i = p;
            } else break;
        }
    }

    fn pushDown(self: *MaxHeap, idx: usize) void {
        var i = idx;
        while (true) {
            const l = left(i);
            const r = right(i);
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

    inline fn rootScore(self: *const MaxHeap) f32 {
        return self.items[0].score;
    }

    inline fn rootValid(self: *const MaxHeap) bool {
        return self.count == K;
    }

    fn push(self: *MaxHeap, h: Hit) void {
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

fn searchTopK(
    vectors: []const [D]f32,
    ids: []const u32,
    query: *const [D]f32,
    out: []Hit,
) void {
    var heap = MaxHeap.init();
    for (vectors, ids) |*v, id| {
        const score = l2SquaredD128(query, v);
        heap.push(.{ .id = id, .score = score });
    }
    // Extract sorted ascending: drain the heap K times (each is the max remaining)
    // and place in reverse order.
    var i: usize = K;
    while (i > 0) : (i -= 1) {
        out[i - 1] = heap.items[0];
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

    // Warmup
    searchTopK(vectors, ids, &queries[0], &out);

    const t0 = std.Io.Clock.awake.now(io);
    var checksum: u64 = 0;
    for (queries) |*q| {
        searchTopK(vectors, ids, q, &out);
        checksum += out[0].id;
    }
    const total_ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    const ns_per_query = total_ns / Q;
    const qps = @as(u64, Q) * std.time.ns_per_s / total_ns;

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
