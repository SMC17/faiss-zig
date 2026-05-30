//! faiss-zig — pure-Zig vector similarity search.
//!
//! v0.0.1 scope: Flat (brute-force) index with three distance metrics
//! (Euclidean L2-squared, inner product, cosine similarity). Scalar
//! baseline; SIMD acceleration is v0.1 work.
//!
//! Comparable upstream:
//! - facebookresearch/faiss — C++ canonical, SIMD-tuned, GPU support
//! - hnswlib — HNSW reference
//! - usearch — Rust SIMD library
//!
//! Honest non-claim: this v0.0.1 is a correctness-first build. We do
//! not yet match faiss perf, do not yet implement IVF / HNSW / PQ.
//! The architecture is shaped to admit those at v0.1+ without API
//! breakage on the Index trait.

const std = @import("std");

pub const Error = error{
    DimensionMismatch,
    EmptyIndex,
    InvalidK,
    InvalidVector,
};

pub const Metric = enum {
    /// Squared Euclidean distance: sum((a-b)^2). Smaller = closer.
    /// We use squared (not sqrt) because (a) sqrt is monotonic so
    /// ranking is unchanged, (b) it's faster.
    l2_squared,
    /// Inner product: sum(a*b). Larger = closer (rank descending).
    inner_product,
    /// Cosine similarity: dot(a, b) / (|a| * |b|). Larger = closer.
    /// Requires non-zero vector norms; returns InvalidVector if a
    /// stored or query vector has zero norm.
    cosine,
};

/// A (id, score) pair returned by kNN search. Ordering depends on
/// the metric: ascending score for l2_squared, descending for
/// inner_product and cosine.
pub const Hit = struct {
    id: u32,
    score: f32,
};

/// Flat (brute-force) index. Stores all vectors contiguously; kNN
/// search is O(N * D) per query. Memory: N * D * sizeof(f32).
///
/// For N up to ~1M and D up to ~1024, single-thread Flat is
/// competitive on consumer CPUs (~10–50ms per query). For larger N
/// or higher D, switch to an approximate index (IVF, HNSW) when
/// they land in v0.1+.
pub const FlatIndex = struct {
    allocator: std.mem.Allocator,
    dim: u32,
    metric: Metric,
    vectors: std.ArrayList(f32) = .empty,
    /// For cosine metric, stored vector norms are cached. Empty for
    /// other metrics.
    norms: std.ArrayList(f32) = .empty,
    /// One-to-one with vector slots; user IDs are u32.
    ids: std.ArrayList(u32) = .empty,
    /// Next auto-assigned ID if the caller passes 0 to add().
    next_id: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, dim: u32, metric: Metric) FlatIndex {
        return .{
            .allocator = allocator,
            .dim = dim,
            .metric = metric,
        };
    }

    pub fn deinit(self: *FlatIndex) void {
        self.vectors.deinit(self.allocator);
        self.norms.deinit(self.allocator);
        self.ids.deinit(self.allocator);
    }

    pub fn len(self: *const FlatIndex) usize {
        return self.ids.items.len;
    }

    /// Add one vector with caller-provided ID. Returns the assigned ID
    /// (caller's ID, unchanged).
    pub fn addWithId(self: *FlatIndex, id: u32, vector: []const f32) !u32 {
        if (vector.len != self.dim) return Error.DimensionMismatch;
        try self.vectors.appendSlice(self.allocator, vector);
        try self.ids.append(self.allocator, id);
        if (self.metric == .cosine) {
            const n = vectorNorm(vector);
            if (n == 0.0) return Error.InvalidVector;
            try self.norms.append(self.allocator, n);
        }
        if (id >= self.next_id) self.next_id = id + 1;
        return id;
    }

    /// Add a vector with auto-assigned ID. Returns the assigned ID.
    pub fn add(self: *FlatIndex, vector: []const f32) !u32 {
        const id = self.next_id;
        return self.addWithId(id, vector);
    }

    /// Search for the K nearest neighbours to `query`. Returns at most
    /// `k` hits sorted by score (ascending for l2_squared, descending
    /// for inner_product and cosine). If `k > len()`, returns len()
    /// hits. Caller owns the returned slice.
    pub fn search(
        self: *const FlatIndex,
        alloc: std.mem.Allocator,
        query: []const f32,
        k: usize,
    ) ![]Hit {
        if (query.len != self.dim) return Error.DimensionMismatch;
        if (k == 0) return Error.InvalidK;
        const n = self.len();
        if (n == 0) return Error.EmptyIndex;

        const effective_k = @min(k, n);
        const direction: topk_heap.TopKDirection = switch (self.metric) {
            .l2_squared => .ascending,
            .inner_product, .cosine => .descending,
        };
        var heap = try topk_heap.TopKHeap.init(alloc, effective_k, direction);
        defer heap.deinit();

        const query_norm = if (self.metric == .cosine) blk: {
            const qn = vectorNorm(query);
            if (qn == 0.0) return Error.InvalidVector;
            break :blk qn;
        } else 0.0;

        var i: usize = 0;
        while (i < n) : (i += 1) {
            const start = i * self.dim;
            const v = self.vectors.items[start .. start + self.dim];
            const score = switch (self.metric) {
                .l2_squared => l2SquaredDistance(query, v),
                .inner_product => innerProduct(query, v),
                .cosine => innerProduct(query, v) / (query_norm * self.norms.items[i]),
            };
            heap.offer(.{ .id = self.ids.items[i], .score = score });
        }

        return try heap.intoSortedSlice(alloc);
    }

    /// Multi-threaded batched kNN search. Spawns `n_threads` worker
    /// threads (clamped to queries.len) and distributes the queries
    /// in contiguous chunks. Each thread runs the single-query search
    /// path independently — no shared mutable state during the
    /// distance computation.
    ///
    /// Returns out[i] = top-k hits for queries[i]; caller frees each
    /// inner slice + the outer slice.
    ///
    /// On any error in a worker thread, the worker panics. The
    /// alloc-fallible variant is the single-thread search() above.
    /// This batch path is bench-grade.
    pub fn searchBatch(
        self: *const FlatIndex,
        alloc: std.mem.Allocator,
        queries: []const []const f32,
        k: usize,
        n_threads: usize,
    ) ![][]Hit {
        const out = try alloc.alloc([]Hit, queries.len);
        errdefer alloc.free(out);

        const T = @min(@max(n_threads, 1), queries.len);
        if (T == 0) return out;
        if (T == 1) {
            // Fall through to single-thread to avoid spawn overhead.
            for (queries, 0..) |q, i| {
                out[i] = try self.search(alloc, q, k);
            }
            return out;
        }

        var threads = try alloc.alloc(std.Thread, T);
        defer alloc.free(threads);

        const WorkerArgs = struct {
            self: *const FlatIndex,
            alloc: std.mem.Allocator,
            queries: []const []const f32,
            k: usize,
            out: [][]Hit,
            start: usize,
            end: usize,
        };
        const args = try alloc.alloc(WorkerArgs, T);
        defer alloc.free(args);

        const chunk = (queries.len + T - 1) / T;
        var t: usize = 0;
        while (t < T) : (t += 1) {
            const start = t * chunk;
            const end = @min(start + chunk, queries.len);
            args[t] = .{
                .self = self,
                .alloc = alloc,
                .queries = queries,
                .k = k,
                .out = out,
                .start = start,
                .end = end,
            };
            threads[t] = try std.Thread.spawn(.{}, struct {
                fn run(a: *WorkerArgs) void {
                    var i = a.start;
                    while (i < a.end) : (i += 1) {
                        a.out[i] = a.self.search(a.alloc, a.queries[i], a.k) catch |err| {
                            std.debug.panic("faiss-zig searchBatch worker failed: {}", .{err});
                        };
                    }
                }
            }.run, .{&args[t]});
        }

        t = 0;
        while (t < T) : (t += 1) threads[t].join();
        return out;
    }
};

// Re-export HNSW
pub const hnsw = @import("hnsw.zig");
pub const IndexHNSW = hnsw.IndexHNSW;
pub const HNSWConfig = hnsw.Config;

// Re-export IVFFlat
pub const ivf = @import("ivf.zig");
pub const IndexIVFFlat = ivf.IndexIVFFlat;
pub const IVFConfig = ivf.IVFConfig;

// Re-export IVFPQ
pub const ivf_pq = @import("ivf_pq.zig");
pub const IndexIVFPQ = ivf_pq.IndexIVFPQ;
pub const IVFPQConfig = ivf_pq.IVFPQConfig;

// ============================================================================
// Distance / similarity primitives — SIMD where dimension fits a clean
// multiple of the lane width, scalar tail otherwise.
//
// We use @Vector(8, f32) as the canonical lane width: matches 256-bit
// AVX2 on x86_64, NEON 128-bit + unroll on aarch64, and lowers to
// scalar on architectures without 256-bit SIMD. The compiler chooses
// the lowering; we don't intrinsic-hardcode anything.
// ============================================================================

const Vec8 = @Vector(8, f32);
const LANE = 8;

fn l2SquaredDistance(a: []const f32, b: []const f32) f32 {
    var sum_vec: Vec8 = @splat(0.0);
    var i: usize = 0;
    const tail = a.len - (a.len % LANE);
    while (i < tail) : (i += LANE) {
        const av: Vec8 = a[i..][0..LANE].*;
        const bv: Vec8 = b[i..][0..LANE].*;
        const d = av - bv;
        sum_vec += d * d;
    }
    var sum: f32 = @reduce(.Add, sum_vec);
    while (i < a.len) : (i += 1) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

fn innerProduct(a: []const f32, b: []const f32) f32 {
    var sum_vec: Vec8 = @splat(0.0);
    var i: usize = 0;
    const tail = a.len - (a.len % LANE);
    while (i < tail) : (i += LANE) {
        const av: Vec8 = a[i..][0..LANE].*;
        const bv: Vec8 = b[i..][0..LANE].*;
        sum_vec += av * bv;
    }
    var sum: f32 = @reduce(.Add, sum_vec);
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }
    return sum;
}

fn vectorNorm(v: []const f32) f32 {
    var sum_vec: Vec8 = @splat(0.0);
    var i: usize = 0;
    const tail = v.len - (v.len % LANE);
    while (i < tail) : (i += LANE) {
        const vv: Vec8 = v[i..][0..LANE].*;
        sum_vec += vv * vv;
    }
    var sum: f32 = @reduce(.Add, sum_vec);
    while (i < v.len) : (i += 1) {
        sum += v[i] * v[i];
    }
    return @sqrt(sum);
}

/// Public SIMD-path distance functions for benches + external callers
/// that want to compute distances without going through the Index
/// wrapper.
pub const distance = struct {
    pub const l2SquaredDistance = @import("root.zig").l2SquaredDistance;
    pub const innerProduct = @import("root.zig").innerProduct;
    pub const vectorNorm = @import("root.zig").vectorNorm;
};

/// Scalar reference implementations — kept for testing the SIMD path
/// against. Public so callers + benches can compare.
pub const scalar = struct {
    pub fn l2SquaredDistance(a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0.0;
        var i: usize = 0;
        while (i < a.len) : (i += 1) {
            const d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    pub fn innerProduct(a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0.0;
        var i: usize = 0;
        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    pub fn vectorNorm(v: []const f32) f32 {
        var sum: f32 = 0.0;
        var i: usize = 0;
        while (i < v.len) : (i += 1) {
            sum += v[i] * v[i];
        }
        return @sqrt(sum);
    }
};

fn hitLessL2(_: void, x: Hit, y: Hit) bool {
    return x.score < y.score;
}

fn hitMoreIP(_: void, x: Hit, y: Hit) bool {
    return x.score > y.score;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

// Force test discovery into transitively-imported modules (Zig 0.16
// only auto-collects from the test root and direct `pub const`
// re-exports won't pull in tests under those declarations unless
// referenced from a test block).
pub const topk_heap = @import("topk_heap.zig");
pub const TopKHeap = topk_heap.TopKHeap;
pub const TopKDirection = topk_heap.TopKDirection;
pub const TopKHeapError = topk_heap.Error;

pub const recall_at_k = @import("recall_at_k.zig");
pub const recallAtK = recall_at_k.recallAtK;
pub const meanRecallOverQueries = recall_at_k.meanRecallOverQueries;
pub const RecallAtKError = recall_at_k.Error;

test {
    _ = hnsw;
    _ = ivf;
    _ = ivf_pq;
}

test "FlatIndex.l2 — empty index returns EmptyIndex" {
    var idx = FlatIndex.init(testing.allocator, 3, .l2_squared);
    defer idx.deinit();
    const query = [_]f32{ 0, 0, 0 };
    try testing.expectError(Error.EmptyIndex, idx.search(testing.allocator, &query, 5));
}

test "FlatIndex.l2 — single vector finds itself" {
    var idx = FlatIndex.init(testing.allocator, 3, .l2_squared);
    defer idx.deinit();
    const v = [_]f32{ 1, 2, 3 };
    const id = try idx.add(&v);
    try testing.expectEqual(@as(u32, 0), id);
    const hits = try idx.search(testing.allocator, &v, 1);
    defer testing.allocator.free(hits);
    try testing.expectEqual(@as(usize, 1), hits.len);
    try testing.expectEqual(@as(u32, 0), hits[0].id);
    try testing.expectApproxEqAbs(@as(f32, 0.0), hits[0].score, 1e-6);
}

test "FlatIndex.l2 — kNN returns sorted by distance ascending" {
    var idx = FlatIndex.init(testing.allocator, 2, .l2_squared);
    defer idx.deinit();
    _ = try idx.add(&[_]f32{ 0, 0 });
    _ = try idx.add(&[_]f32{ 1, 0 });
    _ = try idx.add(&[_]f32{ 0, 1 });
    _ = try idx.add(&[_]f32{ 10, 10 });

    const query = [_]f32{ 0, 0 };
    const hits = try idx.search(testing.allocator, &query, 3);
    defer testing.allocator.free(hits);
    try testing.expectEqual(@as(usize, 3), hits.len);
    try testing.expectEqual(@as(u32, 0), hits[0].id);
    try testing.expectApproxEqAbs(@as(f32, 0.0), hits[0].score, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), hits[1].score, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), hits[2].score, 1e-6);
    try testing.expect(hits[1].score <= hits[2].score);
}

test "FlatIndex.l2 — dimension mismatch on add" {
    var idx = FlatIndex.init(testing.allocator, 3, .l2_squared);
    defer idx.deinit();
    try testing.expectError(
        Error.DimensionMismatch,
        idx.add(&[_]f32{ 1, 2 }),
    );
}

test "FlatIndex.l2 — dimension mismatch on search" {
    var idx = FlatIndex.init(testing.allocator, 3, .l2_squared);
    defer idx.deinit();
    _ = try idx.add(&[_]f32{ 1, 2, 3 });
    try testing.expectError(
        Error.DimensionMismatch,
        idx.search(testing.allocator, &[_]f32{ 1, 2 }, 1),
    );
}

test "FlatIndex.inner_product — sorted descending" {
    var idx = FlatIndex.init(testing.allocator, 2, .inner_product);
    defer idx.deinit();
    _ = try idx.add(&[_]f32{ 1, 0 });    // dot with (1,0) = 1
    _ = try idx.add(&[_]f32{ 0, 1 });    // dot with (1,0) = 0
    _ = try idx.add(&[_]f32{ -1, 0 });   // dot with (1,0) = -1
    _ = try idx.add(&[_]f32{ 0.5, 0.5 }); // dot with (1,0) = 0.5

    const query = [_]f32{ 1, 0 };
    const hits = try idx.search(testing.allocator, &query, 4);
    defer testing.allocator.free(hits);
    try testing.expectEqual(@as(u32, 0), hits[0].id);
    try testing.expectApproxEqAbs(@as(f32, 1.0), hits[0].score, 1e-6);
    try testing.expect(hits[1].score >= hits[2].score);
    try testing.expect(hits[2].score >= hits[3].score);
}

test "FlatIndex.cosine — sorted descending, normalised" {
    var idx = FlatIndex.init(testing.allocator, 2, .cosine);
    defer idx.deinit();
    _ = try idx.add(&[_]f32{ 1, 0 });   // cos with (1,0) = 1
    _ = try idx.add(&[_]f32{ 2, 0 });   // cos with (1,0) = 1 (same direction)
    _ = try idx.add(&[_]f32{ 0, 1 });   // cos with (1,0) = 0
    _ = try idx.add(&[_]f32{ -1, 0 });  // cos with (1,0) = -1

    const query = [_]f32{ 1, 0 };
    const hits = try idx.search(testing.allocator, &query, 4);
    defer testing.allocator.free(hits);
    // First two have cosine 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), hits[0].score, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), hits[1].score, 1e-6);
    // Then 0.0, then -1.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), hits[2].score, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1.0), hits[3].score, 1e-6);
}

test "FlatIndex.cosine — zero-norm vector rejected" {
    var idx = FlatIndex.init(testing.allocator, 2, .cosine);
    defer idx.deinit();
    try testing.expectError(
        Error.InvalidVector,
        idx.add(&[_]f32{ 0, 0 }),
    );
}

test "FlatIndex.cosine — zero-norm query rejected" {
    var idx = FlatIndex.init(testing.allocator, 2, .cosine);
    defer idx.deinit();
    _ = try idx.add(&[_]f32{ 1, 0 });
    try testing.expectError(
        Error.InvalidVector,
        idx.search(testing.allocator, &[_]f32{ 0, 0 }, 1),
    );
}

test "FlatIndex.l2 — k larger than index returns all" {
    var idx = FlatIndex.init(testing.allocator, 2, .l2_squared);
    defer idx.deinit();
    _ = try idx.add(&[_]f32{ 1, 0 });
    _ = try idx.add(&[_]f32{ 0, 1 });

    const hits = try idx.search(testing.allocator, &[_]f32{ 0, 0 }, 100);
    defer testing.allocator.free(hits);
    try testing.expectEqual(@as(usize, 2), hits.len);
}

test "FlatIndex.l2 — addWithId preserves caller IDs" {
    var idx = FlatIndex.init(testing.allocator, 2, .l2_squared);
    defer idx.deinit();
    _ = try idx.addWithId(42, &[_]f32{ 1, 2 });
    _ = try idx.addWithId(7, &[_]f32{ 3, 4 });

    const hits = try idx.search(testing.allocator, &[_]f32{ 1, 2 }, 2);
    defer testing.allocator.free(hits);
    try testing.expectEqual(@as(u32, 42), hits[0].id);
    try testing.expectEqual(@as(u32, 7), hits[1].id);
}

test "k=0 returns InvalidK" {
    var idx = FlatIndex.init(testing.allocator, 2, .l2_squared);
    defer idx.deinit();
    _ = try idx.add(&[_]f32{ 1, 0 });
    try testing.expectError(
        Error.InvalidK,
        idx.search(testing.allocator, &[_]f32{ 1, 0 }, 0),
    );
}

test "distance primitives — l2SquaredDistance" {
    const a = [_]f32{ 0, 0, 0 };
    const b = [_]f32{ 1, 2, 2 };
    // (1)^2 + (2)^2 + (2)^2 = 1 + 4 + 4 = 9
    try testing.expectApproxEqAbs(@as(f32, 9.0), l2SquaredDistance(&a, &b), 1e-6);
}

test "distance primitives — innerProduct" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try testing.expectApproxEqAbs(@as(f32, 32.0), innerProduct(&a, &b), 1e-6);
}

test "distance primitives — vectorNorm" {
    const a = [_]f32{ 3, 4 };
    try testing.expectApproxEqAbs(@as(f32, 5.0), vectorNorm(&a), 1e-6);
}

test "SIMD agrees with scalar — l2_squared on aligned + tail lengths" {
    var prng: std.Random.DefaultPrng = .init(7);
    const rng = prng.random();
    inline for ([_]usize{ 8, 16, 64, 100, 128, 129, 256, 1024, 1025 }) |D| {
        var a: [1025]f32 = undefined;
        var b: [1025]f32 = undefined;
        for (a[0..D]) |*x| x.* = rng.float(f32) * 100.0;
        for (b[0..D]) |*x| x.* = rng.float(f32) * 100.0;
        const simd = l2SquaredDistance(a[0..D], b[0..D]);
        const scal = scalar.l2SquaredDistance(a[0..D], b[0..D]);
        // SIMD reduction order differs; tolerate small relative error.
        const rel = @abs(simd - scal) / @max(@abs(scal), 1e-6);
        try testing.expect(rel < 1e-4);
    }
}

test "SIMD agrees with scalar — innerProduct on aligned + tail lengths" {
    var prng: std.Random.DefaultPrng = .init(11);
    const rng = prng.random();
    inline for ([_]usize{ 8, 16, 64, 100, 128, 129, 256, 1024, 1025 }) |D| {
        var a: [1025]f32 = undefined;
        var b: [1025]f32 = undefined;
        for (a[0..D]) |*x| x.* = rng.float(f32);
        for (b[0..D]) |*x| x.* = rng.float(f32);
        const simd = innerProduct(a[0..D], b[0..D]);
        const scal = scalar.innerProduct(a[0..D], b[0..D]);
        const rel = @abs(simd - scal) / @max(@abs(scal), 1e-6);
        try testing.expect(rel < 1e-4);
    }
}

test "SIMD agrees with scalar — vectorNorm on aligned + tail lengths" {
    var prng: std.Random.DefaultPrng = .init(13);
    const rng = prng.random();
    inline for ([_]usize{ 8, 16, 64, 100, 128, 129, 256, 1024, 1025 }) |D| {
        var v: [1025]f32 = undefined;
        for (v[0..D]) |*x| x.* = rng.float(f32);
        const simd = vectorNorm(v[0..D]);
        const scal = scalar.vectorNorm(v[0..D]);
        const rel = @abs(simd - scal) / @max(@abs(scal), 1e-6);
        try testing.expect(rel < 1e-4);
    }
}
