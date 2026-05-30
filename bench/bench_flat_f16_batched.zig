//! Batched @Vector(8, f16) Flat L2 — tile corpus, reuse for all Q queries.
//!
//! Cache-tile strategy: corpus tile of 64 rows = 64 * 128 * 2 = 16 KB fits
//! L1 cache (128 KB L1d on M2 P-core). For each tile, compute the
//! cross-term dot(q, b) for ALL Q queries against ALL 64 corpus rows
//! while the tile is resident in L1. Corpus is read O(N) times,
//! queries are read O(N / TILE) times.
//!
//! Pre-reg: exp-1780018129-484847944. Falsifier: <1.3x vs sgemm batched.

const std = @import("std");

const D: comptime_int = 128;
const Q: comptime_int = 100;
const K: comptime_int = 10;
const VEC: comptime_int = 8;
const TILE: comptime_int = 64; // corpus rows per tile

const Hit = struct { id: u32, score: f32 };

inline fn f32ToF16(x: f32) f16 {
    return @floatCast(x);
}

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

    const b_f32 = try alloc.alloc(f32, N * D);
    defer alloc.free(b_f32);
    for (b_f32) |*x| x.* = rng.float(f32);

    const b_f16 = try alloc.alloc(f16, N * D);
    defer alloc.free(b_f16);
    for (b_f32, 0..) |v, i| b_f16[i] = f32ToF16(v);

    const b_norms = try alloc.alloc(f32, N);
    defer alloc.free(b_norms);
    for (0..N) |i| {
        var s: f32 = 0;
        const row = b_f32[i * D ..][0..D];
        for (row) |v| s += v * v;
        b_norms[i] = s;
    }

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

    // Per-query heaps (allocated outside the timed loop)
    const heaps = try alloc.alloc(MaxHeap, Q);
    defer alloc.free(heaps);

    // Warmup
    for (heaps) |*h| h.* = MaxHeap.init();
    {
        var tile_start: usize = 0;
        while (tile_start < N) : (tile_start += TILE) {
            const tile_end = @min(tile_start + TILE, N);
            for (tile_start..tile_end) |bi| {
                const b_ptr = b_f16[bi * D ..].ptr;
                const b_sq = b_norms[bi];
                for (0..Q) |qi| {
                    const q_ptr = q_f16[qi * D ..].ptr;
                    const dot = dotF16(q_ptr, b_ptr);
                    const score = q_norms[qi] + b_sq - 2.0 * dot;
                    heaps[qi].push(.{ .id = @intCast(bi), .score = score });
                }
            }
        }
    }

    // Timed pass
    for (heaps) |*h| h.* = MaxHeap.init();
    const t0 = std.Io.Clock.awake.now(io);

    var tile_start: usize = 0;
    while (tile_start < N) : (tile_start += TILE) {
        const tile_end = @min(tile_start + TILE, N);
        // For each corpus row in the tile, score against all Q queries.
        // Corpus row stays L1-resident for Q dot products.
        for (tile_start..tile_end) |bi| {
            const b_ptr = b_f16[bi * D ..].ptr;
            const b_sq = b_norms[bi];
            for (0..Q) |qi| {
                const q_ptr = q_f16[qi * D ..].ptr;
                const dot = dotF16(q_ptr, b_ptr);
                const score = q_norms[qi] + b_sq - 2.0 * dot;
                heaps[qi].push(.{ .id = @intCast(bi), .score = score });
            }
        }
    }

    var checksum: u64 = 0;
    for (heaps) |h| {
        for (h.items[0..h.count]) |hit| checksum += hit.id;
    }

    const total_ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
    const ns_per_query = total_ns / Q;
    const qps = @as(u64, Q) * std.time.ns_per_s / total_ns;

    std.debug.print("N             = {d}\n", .{N});
    std.debug.print("corpus_MB_f16 = {d}\n", .{N * D * @sizeOf(f16) / 1024 / 1024});
    std.debug.print("corpus_MB_f32 = {d}\n", .{N * D * @sizeOf(f32) / 1024 / 1024});
    std.debug.print("tile_rows     = {d}\n", .{TILE});
    std.debug.print("ns_per_query  = {d}\n", .{ns_per_query});
    std.debug.print("us_per_query  = {d}\n", .{ns_per_query / 1000});
    std.debug.print("queries_per_s = {d}\n", .{qps});
    std.debug.print("path = @Vector(8, f16) batched tiled (TILE={d} corpus rows L1-resident × Q queries)\n", .{TILE});
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
