//! Flat L2-squared via cblas_sgemv cross-term refactor.
//!
//!     L2(q, b) = |q|² - 2·q·b + |b|²
//!
//! Per-query: compute |q|², dispatch cblas_sgemv on row-major corpus
//! to get -2·Q·B^T, add precomputed |b|² per row, take top-K.
//!
//! Norms precomputed once at index-build; per-query is one sgemv +
//! N-element add + top-K heap. This is the FAISS internal trick.

const std = @import("std");

extern "c" fn cblas_sgemv(
    order: c_int, // 101 = row major
    trans: c_int, // 111 = no trans
    M: c_int,
    N: c_int,
    alpha: f32,
    A: [*]const f32,
    lda: c_int,
    X: [*]const f32,
    incX: c_int,
    beta: f32,
    Y: [*]f32,
    incY: c_int,
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

    // Corpus flattened as row-major B[N, D]
    const b_flat = try alloc.alloc(f32, N * D);
    defer alloc.free(b_flat);
    for (b_flat) |*x| x.* = rng.float(f32);

    // Precompute |b|² for each row
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

    // Scratch: output vector of N dot products + accumulated scores
    const dots = try alloc.alloc(f32, N);
    defer alloc.free(dots);

    var out: [K]Hit = undefined;

    // Warmup
    cblas_sgemv(101, 111, @intCast(N), D, -2.0, b_flat.ptr, D, queries[0..D].ptr, 1, 0.0, dots.ptr, 1);

    const t0 = std.Io.Clock.awake.now(io);
    var checksum: u64 = 0;
    for (0..Q) |qi| {
        const q = queries[qi * D ..][0..D];

        // |q|² (added uniformly to every distance — doesn't affect ranking,
        // but keep it so scores are real L2² values for verification)
        var q_sq: f32 = 0;
        for (q) |v| q_sq += v * v;

        // -2 · B · q via sgemv
        cblas_sgemv(101, 111, @intCast(N), D, -2.0, b_flat.ptr, D, q.ptr, 1, 0.0, dots.ptr, 1);

        // Add q_sq + b_norms[i] per row, push into top-K heap
        var heap = MaxHeap.init();
        for (0..N) |i| {
            const score = q_sq + b_norms[i] + dots[i];
            heap.push(.{ .id = @intCast(i), .score = score });
        }

        // Drain (descending heap → ascending top-K)
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
    std.debug.print("path = cblas_sgemv cross-term + precomputed |b|^2 norms\n", .{});
    std.debug.print("(cksum {d})\n", .{checksum});
}
