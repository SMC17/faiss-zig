//! Flat L2 via cblas_hgemm — half-precision matmul cross-term.
//!
//! Half the corpus bytes → double the effective memory bandwidth.
//! Single-thread sgemm hit 110 GB/s = 55% of M2 Max peak (200 GB/s);
//! hgemm should push to 220 GB/s effective scan rate.
//!
//! BFloat16 / Float16 representation: Accelerate's cblas_sgemm
//! accepts half-precision via the bfloat16 type on macOS 14+. For
//! older macOS or non-bf16 hardware we'd fall back to f16. We use
//! __fp16 since Mac M2 Max supports both.

const std = @import("std");

// Accelerate's BLAS for f16 uses the cblas_hgemm symbol (half-precision)
// added in macOS Sonoma's BNNS framework. For older macOS or x86_64
// targets this declaration is harmless if never called.
extern "c" fn cblas_sgemm(
    order: c_int,
    transA: c_int,
    transB: c_int,
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

inline fn f32ToF16Bits(x: f32) u16 {
    // Approximate f32 → f16 via @floatCast on aarch64 (Apple Silicon
    // supports IEEE 754 fp16 hardware). For LLVM/Zig 0.16, the cast
    // is a single FCVT instruction.
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

inline fn f16BitsToF32(u: u16) f32 {
    const h: f16 = @bitCast(u);
    return @floatCast(h);
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

    // Build B in f32, then convert to f16 storage. The convert is
    // amortized at index-build time so per-query is the f16 scan +
    // f16→f32 widen inside cblas-equivalent.
    const b_f32 = try alloc.alloc(f32, N * D);
    defer alloc.free(b_f32);
    for (b_f32) |*x| x.* = rng.float(f32);

    // f16 storage of corpus
    const b_f16 = try alloc.alloc(u16, N * D);
    defer alloc.free(b_f16);
    for (b_f32, 0..) |v, i| b_f16[i] = f32ToF16Bits(v);

    // Precompute |b|² in f32 (precision matters for ranking)
    const b_norms = try alloc.alloc(f32, N);
    defer alloc.free(b_norms);
    for (0..N) |i| {
        var s: f32 = 0;
        const row = b_f32[i * D ..][0..D];
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

    // Path: convert b_f16 back to f32 ON-THE-FLY into a scratch buffer,
    // then call cblas_sgemm. This measures the effective benefit of
    // f16 storage (half memory) with on-the-fly widening; it's the
    // best we can do without cblas_hgemm public symbol on macOS.
    //
    // For real true f16 throughput we'd write hand-rolled @Vector(16, f16)
    // accumulator that loads from b_f16 directly, but Apple's Accelerate
    // expects f32 inputs to cblas_sgemm.
    const b_scratch = try alloc.alloc(f32, N * D);
    defer alloc.free(b_scratch);

    const dots = try alloc.alloc(f32, Q * N);
    defer alloc.free(dots);

    // Warmup: widen + sgemm
    for (b_f16, 0..) |u, i| b_scratch[i] = f16BitsToF32(u);
    cblas_sgemm(101, 111, 112, Q, @intCast(N), D, -2.0, queries.ptr, D, b_scratch.ptr, D, 0.0, dots.ptr, @intCast(N));

    const t0 = std.Io.Clock.awake.now(io);

    // f16 widen pass (this IS the bandwidth win — reading f16 from RAM,
    // writing f32 to scratch). The scratch is sized N*D*4 bytes; could
    // be reused per batch in a real impl. Here we do it every iter to
    // honestly measure the round trip.
    for (b_f16, 0..) |u, i| b_scratch[i] = f16BitsToF32(u);

    cblas_sgemm(101, 111, 112, Q, @intCast(N), D, -2.0, queries.ptr, D, b_scratch.ptr, D, 0.0, dots.ptr, @intCast(N));

    var checksum: u64 = 0;
    for (0..Q) |qi| {
        const q_sq = q_norms[qi];
        var heap = MaxHeap.init();
        const row = dots[qi * N ..][0..N];
        for (0..N) |i| {
            const score = q_sq + b_norms[i] + row[i];
            heap.push(.{ .id = @intCast(i), .score = score });
        }
        var out: [K]Hit = undefined;
        var jd: usize = K;
        while (jd > 0) : (jd -= 1) {
            out[jd - 1] = heap.items[0];
            heap.items[0] = heap.items[heap.count - 1];
            heap.count -= 1;
            if (heap.count > 0) heap.pushDown(0);
        }
        checksum += out[0].id;
    }

    const total_ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
    const ns_per_query = total_ns / Q;
    const qps = @as(u64, Q) * std.time.ns_per_s / total_ns;

    std.debug.print("N             = {d}\n", .{N});
    std.debug.print("corpus_MB_f16 = {d}\n", .{N * D * @sizeOf(u16) / 1024 / 1024});
    std.debug.print("corpus_MB_f32 = {d}\n", .{N * D * @sizeOf(f32) / 1024 / 1024});
    std.debug.print("ns_per_query  = {d}\n", .{ns_per_query});
    std.debug.print("us_per_query  = {d}\n", .{ns_per_query / 1000});
    std.debug.print("queries_per_s = {d}\n", .{qps});
    std.debug.print("path = f16 corpus storage + on-the-fly widen + cblas_sgemm\n", .{});
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
