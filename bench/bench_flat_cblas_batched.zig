//! Batched Flat L2 — all Q queries vs full corpus via single cblas_sgemm.
//!
//! Compute -2 · Q · B^T in one shot where Q is [Q, D], B is [N, D],
//! result is [Q, N] dot-product matrix. Add per-row b_norms + per-query
//! q_norm, then top-K per query.
//!
//! Amortizes sgemm dispatch overhead across all Q queries. For Q=100
//! D=128 N=1M the total work is 12.8 GFLOPs — under 200ms on M2 Max AMX.

const std = @import("std");

extern "c" fn cblas_sgemm(
    order: c_int, // 101 = row major
    transA: c_int, // 111 = no trans
    transB: c_int, // 112 = trans
    M: c_int,
    N: c_int,
    K_dim: c_int,
    alpha: f32,
    A: [*]const f32,
    lda: c_int,
    B: [*]const f32,
    ldb: c_int,
    beta: f32,
    C: [*]f32,
    ldc: c_int,
) void;

const D: comptime_int = 128;
const Q: comptime_int = 100;
const K: comptime_int = 10;

const Hit = struct { id: u32, score: f32 };

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

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer iter.deinit();
    _ = iter.next();
    const N = try std.fmt.parseInt(usize, iter.next() orelse "10000", 10);

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    const b_flat = try alloc.alloc(f32, N * D);
    defer alloc.free(b_flat);
    for (b_flat) |*x| x.* = rng.float(f32);

    const b_norms = try alloc.alloc(f32, N);
    defer alloc.free(b_norms);
    for (0..N) |i| {
        var s: f32 = 0;
        const row = b_flat[i * D ..][0..D];
        for (row) |v| s += v * v;
        b_norms[i] = s;
    }

    const queries = try alloc.alloc(f32, Q * D);
    defer alloc.free(queries);
    for (queries) |*x| x.* = rng.float(f32);

    const q_norms = try alloc.alloc(f32, Q);
    defer alloc.free(q_norms);
    for (0..Q) |i| {
        var s: f32 = 0;
        const row = queries[i * D ..][0..D];
        for (row) |v| s += v * v;
        q_norms[i] = s;
    }

    // Dots matrix [Q, N] — -2 · Q · B^T
    const dots = try alloc.alloc(f32, Q * N);
    defer alloc.free(dots);

    var checksum: u64 = 0;

    // Warmup
    cblas_sgemm(101, 111, 112, Q, @intCast(N), D, -2.0, queries.ptr, D, b_flat.ptr, D, 0.0, dots.ptr, @intCast(N));

    const t0 = std.Io.Clock.awake.now(io);

    // Single sgemm for all Q queries
    cblas_sgemm(101, 111, 112, Q, @intCast(N), D, -2.0, queries.ptr, D, b_flat.ptr, D, 0.0, dots.ptr, @intCast(N));

    // Add norms + top-K per query
    for (0..Q) |qi| {
        const q_sq = q_norms[qi];
        const row = dots[qi * N ..][0..N];
        var heap = MaxHeap.init();
        for (0..N) |i| {
            const score = q_sq + b_norms[i] + row[i];
            heap.push(.{ .id = @intCast(i), .score = score });
        }
        // Drain
        var out: [K]Hit = undefined;
        var j: usize = K;
        while (j > 0) : (j -= 1) {
            out[j - 1] = heap.items[0];
            heap.items[0] = heap.items[heap.count - 1];
            heap.count -= 1;
            if (heap.count > 0) heap.pushDown(0);
        }
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
    std.debug.print("path = cblas_sgemm batched Q queries\n", .{});
    std.debug.print("(cksum {d})\n", .{checksum});
}
