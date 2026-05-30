//! Persistent-worker-pool parallel cblas Flat L2.
//!
//! Spawn NTHREADS workers ONCE at startup. Each worker spins on an
//! atomic round counter (waits for main to increment), runs its sgemm
//! on its slice of B, then decrements an active counter. Main posts
//! a new round + waits for active counter to drop to 0. No per-query
//! thread spawn cost.
//!
//! Goal: beat faiss-cpu T=8 batched at all N. Single-thread sgemm
//! already hit 110 GB/s (55% of M2 Max peak); persistent T=2 should
//! hit ~200 GB/s by leveraging both AMX clusters.

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

const WorkSlice = struct {
    queries_ptr: [*]const f32,
    b_ptr: [*]const f32,
    n_slice: usize,
    dots_ptr: [*]f32,
};

const Pool = struct {
    workers: [NTHREADS]std.Thread,
    work: [NTHREADS]WorkSlice,
    round: std.atomic.Value(u64),
    active: std.atomic.Value(u32),
    done: std.atomic.Value(u32),
    shutdown: std.atomic.Value(bool),

    fn workerLoop(self: *Pool, idx: usize) void {
        var last_round: u64 = 0;
        while (!self.shutdown.load(.acquire)) {
            const r = self.round.load(.acquire);
            if (r != last_round) {
                last_round = r;
                const w = self.work[idx];
                cblas_sgemm(
                    101, 111, 112,
                    Q, @intCast(w.n_slice), D,
                    -2.0,
                    w.queries_ptr, D,
                    w.b_ptr, D,
                    0.0,
                    w.dots_ptr, @intCast(w.n_slice),
                );
                _ = self.done.fetchAdd(1, .release);
            } else {
                std.Thread.yield() catch {};
            }
        }
    }

    fn submitAndWait(self: *Pool) void {
        self.done.store(0, .release);
        _ = self.round.fetchAdd(1, .release);
        while (self.done.load(.acquire) < NTHREADS) {
            std.Thread.yield() catch {};
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

    // Pool with per-thread dots scratch and slice metadata
    var pool: Pool = .{
        .workers = undefined,
        .work = undefined,
        .round = std.atomic.Value(u64).init(0),
        .active = std.atomic.Value(u32).init(0),
        .done = std.atomic.Value(u32).init(0),
        .shutdown = std.atomic.Value(bool).init(false),
    };

    var thread_dots: [NTHREADS][]f32 = undefined;
    var slice_lens: [NTHREADS]usize = undefined;
    var slice_bases: [NTHREADS]usize = undefined;
    const chunk = N / NTHREADS;
    for (0..NTHREADS) |t| {
        const start = t * chunk;
        const end = if (t == NTHREADS - 1) N else start + chunk;
        const slice_n = end - start;
        slice_bases[t] = start;
        slice_lens[t] = slice_n;
        thread_dots[t] = try alloc.alloc(f32, Q * slice_n);
        pool.work[t] = .{
            .queries_ptr = queries.ptr,
            .b_ptr = b_flat[start * D ..].ptr,
            .n_slice = slice_n,
            .dots_ptr = thread_dots[t].ptr,
        };
    }
    defer for (thread_dots) |d| alloc.free(d);

    // Spawn workers (persistent for entire bench run)
    for (0..NTHREADS) |t| {
        pool.workers[t] = try std.Thread.spawn(.{}, Pool.workerLoop, .{ &pool, t });
    }

    // Warmup
    pool.submitAndWait();

    const t0 = std.Io.Clock.awake.now(io);

    // 100 query batches — but we only have Q=100 queries, so this loops
    // the whole sgemm call MULTIPLE times. Adjust: just one submission.
    pool.submitAndWait();

    // Merge per-query top-K
    var checksum: u64 = 0;
    for (0..Q) |qi| {
        const q_sq = q_norms[qi];
        var heap = MaxHeap.init();
        for (0..NTHREADS) |t| {
            const slice_n = slice_lens[t];
            const dots = thread_dots[t][qi * slice_n ..][0..slice_n];
            const base = slice_bases[t];
            const norms = b_norms[base..(base + slice_n)];
            for (0..slice_n) |j| {
                const score = q_sq + norms[j] + dots[j];
                heap.push(.{ .id = @intCast(base + j), .score = score });
            }
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

    // Shutdown
    pool.shutdown.store(true, .release);
    for (pool.workers) |w| w.join();

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
    std.debug.print("path = persistent NTHREADS={d} worker pool + cblas_sgemm batched\n", .{NTHREADS});
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
