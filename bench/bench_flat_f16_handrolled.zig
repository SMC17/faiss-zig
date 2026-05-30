//! Hand-rolled @Vector(8, f16) half-precision Flat L2 matmul.
//!
//! On Apple Silicon (M2 Max), ARMv8.2-A FP16 SIMD gives us 8 lanes of
//! FP16 FMA per 128-bit NEON register. f16 storage halves the corpus
//! bandwidth vs f32, AND f16 ALUs run at the same rate as f32 ones —
//! so we should hit ~2x the effective scan rate.
//!
//! Layout: cross-term L2(q, b) = |q|^2 + |b|^2 - 2*dot(q, b). We
//! precompute |b|^2 in f32 (precision matters for ranking), store
//! B as raw f16 (16 bytes per D=128 row = halves bandwidth), and
//! compute -2*dot(q, b) using @Vector(8, f16).
//!
//! Reference: cblas_sgemm batched (bench_flat_cblas_batched.zig) hit
//! 20,646 q/s at N=10K. Falsifier: <1.2x speedup.

const std = @import("std");

const D: comptime_int = 128;
const Q: comptime_int = 100;
const K: comptime_int = 10;
const VEC: comptime_int = 8; // ARMv8.2 NEON f16 lanes

const Hit = struct { id: u32, score: f32 };

inline fn f32ToF16(x: f32) f16 {
    return @floatCast(x);
}

inline fn f16ToF32(h: f16) f32 {
    return @floatCast(h);
}

/// Single-query dot product using @Vector(8, f16) accumulator.
/// D=128 → 16 vector ops per dot. LLVM compiles this to FMLA.8h on Apple.
inline fn dotF16(q: [*]const f16, b: [*]const f16) f32 {
    var acc: @Vector(VEC, f16) = @splat(0.0);
    comptime var i: usize = 0;
    inline while (i < D) : (i += VEC) {
        const qv: @Vector(VEC, f16) = q[i..][0..VEC].*;
        const bv: @Vector(VEC, f16) = b[i..][0..VEC].*;
        acc += qv * bv;
    }
    return @floatCast(@reduce(.Add, acc));
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

    // Build B in f32 for reference + precomputed |b|², then convert to f16 storage
    const b_f32 = try alloc.alloc(f32, N * D);
    defer alloc.free(b_f32);
    for (b_f32) |*x| x.* = rng.float(f32);

    const b_f16 = try alloc.alloc(f16, N * D);
    defer alloc.free(b_f16);
    for (b_f32, 0..) |v, i| b_f16[i] = f32ToF16(v);

    // |b|² precomputed in f32 for ranking precision
    const b_norms = try alloc.alloc(f32, N);
    defer alloc.free(b_norms);
    for (0..N) |i| {
        var s: f32 = 0;
        const row = b_f32[i * D ..][0..D];
        for (row) |v| s += v * v;
        b_norms[i] = s;
    }

    // Queries
    const q_f32 = try alloc.alloc(f32, Q * D);
    defer alloc.free(q_f32);
    for (q_f32) |*x| x.* = rng.float(f32);

    const q_f16 = try alloc.alloc(f16, Q * D);
    defer alloc.free(q_f16);
    for (q_f32, 0..) |v, i| q_f16[i] = f32ToF16(v);

    const q_norms = try alloc.alloc(f32, Q);
    defer alloc.free(q_norms);
    for (0..Q) |i| {
        var s: f32 = 0;
        const row = q_f32[i * D ..][0..D];
        for (row) |v| s += v * v;
        q_norms[i] = s;
    }

    // Warmup pass
    var heap_warm = MaxHeap.init();
    {
        const q_ptr = q_f16[0..].ptr;
        for (0..N) |i| {
            const b_ptr = b_f16[i * D ..].ptr;
            const dot = dotF16(q_ptr, b_ptr);
            const score = q_norms[0] + b_norms[i] - 2.0 * dot;
            heap_warm.push(.{ .id = @intCast(i), .score = score });
        }
    }

    const t0 = std.Io.Clock.awake.now(io);

    var checksum: u64 = 0;
    for (0..Q) |qi| {
        const q_ptr = q_f16[qi * D ..].ptr;
        const q_sq = q_norms[qi];
        var heap = MaxHeap.init();
        for (0..N) |i| {
            const b_ptr = b_f16[i * D ..].ptr;
            const dot = dotF16(q_ptr, b_ptr);
            const score = q_sq + b_norms[i] - 2.0 * dot;
            heap.push(.{ .id = @intCast(i), .score = score });
        }
        // Drain top-K (just for checksum)
        var jd: usize = heap.count;
        while (jd > 0) : (jd -= 1) {
            checksum += heap.items[jd - 1].id;
        }
    }

    const total_ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
    const ns_per_query = total_ns / Q;
    const qps = @as(u64, Q) * std.time.ns_per_s / total_ns;

    std.debug.print("N             = {d}\n", .{N});
    std.debug.print("corpus_MB_f16 = {d}\n", .{N * D * @sizeOf(f16) / 1024 / 1024});
    std.debug.print("corpus_MB_f32 = {d}\n", .{N * D * @sizeOf(f32) / 1024 / 1024});
    std.debug.print("ns_per_query  = {d}\n", .{ns_per_query});
    std.debug.print("us_per_query  = {d}\n", .{ns_per_query / 1000});
    std.debug.print("queries_per_s = {d}\n", .{qps});
    std.debug.print("path = @Vector(8, f16) handrolled FMLA.8h on f16 corpus + f16 queries\n", .{});
    std.debug.print("(cksum {d})\n", .{checksum});
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
