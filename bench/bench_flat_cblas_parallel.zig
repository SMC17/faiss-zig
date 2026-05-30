//! Application-level parallel cblas Flat L2.
//!
//! Split corpus into NTHREADS equal chunks; each thread runs sgemm on
//! its slice of B against all Q queries. Merge per-query top-K across
//! all threads at the end.
//!
//! Accelerate's sgemm appears single-threaded for our workload
//! (VECLIB_MAXIMUM_THREADS=8 produced no change). Splitting at the
//! application layer leverages multiple AMX-eligible perf cores
//! simultaneously.

const std = @import("std");

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
const NTHREADS: comptime_int = 2;

const Hit = struct { id: u32, score: f32 };

const ThreadArgs = struct {
    queries: []const f32,
    b_slice: []const f32,
    b_norms_slice: []const f32,
    q_norms: []const f32,
    n_slice: usize,
    base_id: u32,
    out_dots: []f32,
};

fn workerSgemm(args: *ThreadArgs) void {
    cblas_sgemm(
        101, 111, 112,
        Q, @intCast(args.n_slice), D,
        -2.0,
        args.queries.ptr, D,
        args.b_slice.ptr, D,
        0.0,
        args.out_dots.ptr, @intCast(args.n_slice),
    );
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

    // Per-thread output dots arrays
    var thread_args: [NTHREADS]ThreadArgs = undefined;
    var thread_dots: [NTHREADS][]f32 = undefined;
    const chunk = N / NTHREADS;
    for (0..NTHREADS) |t| {
        const start = t * chunk;
        const end = if (t == NTHREADS - 1) N else start + chunk;
        const slice_n = end - start;
        thread_dots[t] = try alloc.alloc(f32, Q * slice_n);
        thread_args[t] = .{
            .queries = queries,
            .b_slice = b_flat[start * D ..][0..(slice_n * D)],
            .b_norms_slice = b_norms[start..end],
            .q_norms = q_norms,
            .n_slice = slice_n,
            .base_id = @intCast(start),
            .out_dots = thread_dots[t],
        };
    }
    defer for (thread_dots) |d| alloc.free(d);

    var threads: [NTHREADS]std.Thread = undefined;

    // Warmup
    for (0..NTHREADS) |t| {
        threads[t] = try std.Thread.spawn(.{}, workerSgemm, .{&thread_args[t]});
    }
    for (threads) |th| th.join();

    const t0 = std.Io.Clock.awake.now(io);

    // Parallel sgemm — spawn threads, each runs its sgemm on its B slice
    for (0..NTHREADS) |t| {
        threads[t] = try std.Thread.spawn(.{}, workerSgemm, .{&thread_args[t]});
    }
    for (threads) |th| th.join();

    // Merge per-query: maintain max-heap of size K across all threads
    var checksum: u64 = 0;
    for (0..Q) |qi| {
        const q_sq = q_norms[qi];
        var heap = MaxHeap.init();
        for (0..NTHREADS) |t| {
            const dots = thread_args[t].out_dots[qi * thread_args[t].n_slice ..][0..thread_args[t].n_slice];
            const base = thread_args[t].base_id;
            for (dots, 0..) |dot, j| {
                const score = q_sq + thread_args[t].b_norms_slice[j] + dot;
                heap.push(.{ .id = base + @as(u32, @intCast(j)), .score = score });
            }
        }
        // Drain
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
    const bytes_per_query = N * D * @sizeOf(f32);
    const gbps = @as(f64, @floatFromInt(bytes_per_query * qps)) / 1_000_000_000.0;

    std.debug.print("N             = {d}\n", .{N});
    std.debug.print("corpus_MB     = {d}\n", .{N * D * @sizeOf(f32) / 1024 / 1024});
    std.debug.print("nthreads      = {d}\n", .{NTHREADS});
    std.debug.print("ns_per_query  = {d}\n", .{ns_per_query});
    std.debug.print("us_per_query  = {d}\n", .{ns_per_query / 1000});
    std.debug.print("queries_per_s = {d}\n", .{qps});
    std.debug.print("effective_GBps_scan = {d:.2}\n", .{gbps});
    std.debug.print("(cksum {d})\n", .{checksum});
}

const MaxHeap = struct {
    items: [K]Hit,
    count: usize,

    fn init() MaxHeap { return .{ .items = undefined, .count = 0 }; }

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
