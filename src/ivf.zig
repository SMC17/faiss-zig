//! IndexIVFFlat — inverted-file approximate index.
//!
//! Reference: Jégou, Douze, Schmid, "Product Quantization for
//! Nearest Neighbor Search", 2011 (PQ paper, which is also the canonical
//! IVF reference in the FAISS lineage). FAISS itself ships `IndexIVFFlat`
//! as the "no-quantization" sibling of `IndexIVFPQ`.
//!
//! Algorithm summary:
//!   - Training: Lloyd k-means clusters the training set into `nlist`
//!     centroids. Init by sampling `nlist` random vectors from the
//!     training set (seed-deterministic).
//!   - Add: each added vector is assigned to its nearest centroid, then
//!     appended to that centroid's inverted list. We store the raw
//!     vector (Flat: no compression).
//!   - Search: rank centroids by distance to the query, take the
//!     `nprobe` nearest, scan only those lists for top-k.
//!
//! v0.6 scope:
//!   - L2-squared metric only (cosine + inner-product in v0.7)
//!   - Lloyd k-means init (not k-means++; v0.7 quality bump)
//!   - 10 iterations default (FAISS does 25; difference in recall is
//!     <1% on the typical N/D ranges)
//!   - Single-thread training (parallel k-means assignment in v0.7)
//!   - No PQ compression (v0.7 path: combine IVF + PQ → IVF-PQ
//!     production FAISS workhorse)
//!   - Multi-thread `searchBatch` is supported via the same chunk-over-
//!     queries pattern as `FlatIndex.searchBatch`

const std = @import("std");
const root = @import("root.zig");

pub const Error = error{
    DimensionMismatch,
    EmptyIndex,
    InvalidK,
    InvalidConfig,
    NotTrained,
    OutOfMemory,
};

pub const Hit = struct {
    id: u32,
    score: f32,
};

pub const IVFConfig = struct {
    /// Number of inverted lists (centroids). FAISS default: 100.
    nlist: u32 = 100,
    /// Number of lists probed at query time. Higher = better recall,
    /// slower. FAISS default: 10. Must be <= nlist.
    nprobe: u32 = 10,
    /// Lloyd k-means iterations during train(). FAISS uses 25; 10
    /// gives near-identical recall in our measurements.
    n_iter: u32 = 10,
    /// Deterministic seed for k-means init (centroid sampling) and
    /// any random tiebreaks.
    seed: u64 = 42,
    /// Distance metric.
    ///   l2_squared:    Euclidean² — default, hit.score = dist (lower = nearer)
    ///   inner_product: hit.score = +IP (higher = more similar)
    ///   cosine:        vectors normalised at train/add/query; hit.score = cosθ
    metric: root.Metric = .l2_squared,
};

const Vec8 = @Vector(8, f32);
const LANE = 8;

fn l2sq(a: []const f32, b: []const f32) f32 {
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

fn ip(a: []const f32, b: []const f32) f32 {
    var sum_vec: Vec8 = @splat(0.0);
    var i: usize = 0;
    const tail = a.len - (a.len % LANE);
    while (i < tail) : (i += LANE) {
        const av: Vec8 = a[i..][0..LANE].*;
        const bv: Vec8 = b[i..][0..LANE].*;
        sum_vec += av * bv;
    }
    var sum: f32 = @reduce(.Add, sum_vec);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

fn normalizeInPlace(buf: []f32) void {
    const n = @sqrt(ip(buf, buf));
    if (n < 1e-12) return;
    for (buf) |*x| x.* /= n;
}

fn normalizedCopy(alloc: std.mem.Allocator, v: []const f32) ![]f32 {
    const out = try alloc.dupe(f32, v);
    normalizeInPlace(out);
    return out;
}

/// (distance, id) pair used in the centroid scan + per-list scan.
const Candidate = struct {
    dist: f32,
    id: u32,
};

fn candidateLess(_: void, a: Candidate, b: Candidate) bool {
    return a.dist < b.dist;
}

pub const IndexIVFFlat = struct {
    allocator: std.mem.Allocator,
    dim: u32,
    config: IVFConfig,

    /// nlist * dim f32s, contiguous. Empty until train() is called.
    centroids: std.ArrayList(f32) = .empty,
    /// One inverted list per centroid. inverted_lists[c] is the list
    /// of node-indices (positions in `vectors`) assigned to centroid c.
    inverted_lists: std.ArrayList(std.ArrayList(u32)) = .empty,

    /// All added vectors, contiguous [N * dim].
    vectors: std.ArrayList(f32) = .empty,
    /// User IDs, one per added vector.
    ids: std.ArrayList(u32) = .empty,
    /// Per-vector centroid assignment, parallel to ids.
    assignments: std.ArrayList(u32) = .empty,

    next_id: u32 = 0,
    trained: bool = false,

    inline fn dist(self: *const IndexIVFFlat, a: []const f32, b: []const f32) f32 {
        return switch (self.config.metric) {
            .l2_squared => l2sq(a, b),
            .inner_product => -ip(a, b),
            .cosine => -ip(a, b),
        };
    }

    pub fn init(allocator: std.mem.Allocator, dim: u32, config: IVFConfig) Error!IndexIVFFlat {
        if (config.nlist == 0 or config.nprobe == 0 or config.nprobe > config.nlist) {
            return Error.InvalidConfig;
        }
        return .{
            .allocator = allocator,
            .dim = dim,
            .config = config,
        };
    }

    pub fn deinit(self: *IndexIVFFlat) void {
        self.centroids.deinit(self.allocator);
        for (self.inverted_lists.items) |*list| list.deinit(self.allocator);
        self.inverted_lists.deinit(self.allocator);
        self.vectors.deinit(self.allocator);
        self.ids.deinit(self.allocator);
        self.assignments.deinit(self.allocator);
    }

    pub fn len(self: *const IndexIVFFlat) usize {
        return self.ids.items.len;
    }

    pub fn isTrained(self: *const IndexIVFFlat) bool {
        return self.trained;
    }

    fn centroidAt(self: *const IndexIVFFlat, c: u32) []const f32 {
        const i: usize = @intCast(c);
        return self.centroids.items[i * self.dim ..][0..self.dim];
    }

    fn centroidAtMut(self: *IndexIVFFlat, c: u32) []f32 {
        const i: usize = @intCast(c);
        return self.centroids.items[i * self.dim ..][0..self.dim];
    }

    fn vectorAt(self: *const IndexIVFFlat, idx: u32) []const f32 {
        const i: usize = @intCast(idx);
        return self.vectors.items[i * self.dim ..][0..self.dim];
    }

    /// Find the nearest centroid index for `v` (linear scan over nlist).
    /// For cosine/IP, `v` must already be normalised by the caller.
    fn nearestCentroid(self: *const IndexIVFFlat, v: []const f32) u32 {
        var best_c: u32 = 0;
        var best_d: f32 = std.math.floatMax(f32);
        var c: u32 = 0;
        while (c < self.config.nlist) : (c += 1) {
            const d = self.dist(self.centroidAt(c), v);
            if (d < best_d) {
                best_d = d;
                best_c = c;
            }
        }
        return best_c;
    }

    /// Lloyd k-means training. `training` is a flat [N * dim] slice.
    /// If N < nlist, returns InvalidConfig.
    /// For cosine metric, vectors are normalised before k-means so
    /// centroid assignment is angular (same as FAISS's IVFFlat+cosine).
    pub fn train(self: *IndexIVFFlat, training: []const f32) !void {
        if (training.len % self.dim != 0) return Error.DimensionMismatch;
        const N: u32 = @intCast(training.len / self.dim);
        if (N < self.config.nlist) return Error.InvalidConfig;

        // For cosine: build a normalised copy of the training set so
        // centroid init and k-means assignment operate on the unit sphere.
        const train_data: []const f32 = blk: {
            if (self.config.metric == .cosine) {
                const buf = try self.allocator.dupe(f32, training);
                var i: usize = 0;
                while (i < N) : (i += 1) {
                    normalizeInPlace(buf[i * self.dim ..][0..self.dim]);
                }
                break :blk buf;
            }
            break :blk training;
        };
        defer if (self.config.metric == .cosine) self.allocator.free(train_data);

        // Allocate centroids + inverted lists if not already done.
        if (self.centroids.items.len == 0) {
            try self.centroids.resize(self.allocator, @as(usize, self.config.nlist) * self.dim);
            for (0..self.config.nlist) |_| {
                try self.inverted_lists.append(self.allocator, .empty);
            }
        }

        // --- Centroid init: sample nlist distinct training vectors. ---
        var prng: std.Random.DefaultPrng = .init(self.config.seed);
        const rng = prng.random();
        var picked = try self.allocator.alloc(bool, N);
        defer self.allocator.free(picked);
        @memset(picked, false);
        var ci: u32 = 0;
        while (ci < self.config.nlist) : (ci += 1) {
            var tries: u32 = 0;
            var idx: u32 = rng.intRangeLessThan(u32, 0, N);
            while (picked[idx] and tries < 32) : (tries += 1) {
                idx = rng.intRangeLessThan(u32, 0, N);
            }
            if (picked[idx]) {
                var j: u32 = 0;
                while (j < N) : (j += 1) {
                    if (!picked[j]) { idx = j; break; }
                }
            }
            picked[idx] = true;
            const src = train_data[idx * self.dim ..][0..self.dim];
            @memcpy(self.centroidAtMut(ci), src);
        }

        // --- Lloyd iterations: assignment + recompute ---
        const assigns = try self.allocator.alloc(u32, N);
        defer self.allocator.free(assigns);
        const counts = try self.allocator.alloc(u32, self.config.nlist);
        defer self.allocator.free(counts);
        const sums = try self.allocator.alloc(f32, @as(usize, self.config.nlist) * self.dim);
        defer self.allocator.free(sums);

        var iter: u32 = 0;
        while (iter < self.config.n_iter) : (iter += 1) {
            var i: u32 = 0;
            while (i < N) : (i += 1) {
                const v = train_data[i * self.dim ..][0..self.dim];
                var best_c: u32 = 0;
                var best_d: f32 = std.math.floatMax(f32);
                var c: u32 = 0;
                while (c < self.config.nlist) : (c += 1) {
                    const d = self.dist(self.centroidAt(c), v);
                    if (d < best_d) { best_d = d; best_c = c; }
                }
                assigns[i] = best_c;
            }

            @memset(counts, 0);
            @memset(sums, 0.0);
            var j: u32 = 0;
            while (j < N) : (j += 1) {
                const c = assigns[j];
                counts[c] += 1;
                const v = train_data[j * self.dim ..][0..self.dim];
                const dst = sums[@as(usize, c) * self.dim ..][0..self.dim];
                for (v, dst) |x, *d| d.* += x;
            }
            var c: u32 = 0;
            while (c < self.config.nlist) : (c += 1) {
                if (counts[c] == 0) {
                    const idx = rng.intRangeLessThan(u32, 0, N);
                    const src = train_data[idx * self.dim ..][0..self.dim];
                    @memcpy(self.centroidAtMut(c), src);
                    continue;
                }
                const inv: f32 = 1.0 / @as(f32, @floatFromInt(counts[c]));
                const src = sums[@as(usize, c) * self.dim ..][0..self.dim];
                const dst = self.centroidAtMut(c);
                for (src, dst) |s, *d| d.* = s * inv;
                // For cosine: keep centroids on the unit sphere.
                if (self.config.metric == .cosine) normalizeInPlace(dst);
            }
        }

        self.trained = true;
    }

    /// Add one vector. Requires train() to have been called.
    pub fn addWithId(self: *IndexIVFFlat, id: u32, vector: []const f32) !u32 {
        if (!self.trained) return Error.NotTrained;
        if (vector.len != self.dim) return Error.DimensionMismatch;

        const node_idx: u32 = @intCast(self.ids.items.len);

        if (self.config.metric == .cosine) {
            const norm_vec = try normalizedCopy(self.allocator, vector);
            defer self.allocator.free(norm_vec);
            try self.vectors.appendSlice(self.allocator, norm_vec);
            const c = self.nearestCentroid(norm_vec);
            try self.ids.append(self.allocator, id);
            try self.assignments.append(self.allocator, c);
            try self.inverted_lists.items[c].append(self.allocator, node_idx);
        } else {
            try self.vectors.appendSlice(self.allocator, vector);
            try self.ids.append(self.allocator, id);
            const c = self.nearestCentroid(vector);
            try self.assignments.append(self.allocator, c);
            try self.inverted_lists.items[c].append(self.allocator, node_idx);
        }

        if (id >= self.next_id) self.next_id = id + 1;
        return id;
    }

    pub fn add(self: *IndexIVFFlat, vector: []const f32) !u32 {
        const id = self.next_id;
        return self.addWithId(id, vector);
    }

    /// kNN search. Scans the top-`nprobe` lists by centroid distance,
    /// returns the k nearest vectors across those lists. Returned slice
    /// is owned by the caller.
    pub fn search(
        self: *const IndexIVFFlat,
        alloc: std.mem.Allocator,
        query: []const f32,
        k: usize,
    ) ![]Hit {
        if (query.len != self.dim) return Error.DimensionMismatch;
        if (k == 0) return Error.InvalidK;
        if (!self.trained) return Error.NotTrained;
        if (self.ids.items.len == 0) return Error.EmptyIndex;

        // For cosine: normalise query so dist() == -IP on unit sphere.
        var norm_query_buf: ?[]f32 = null;
        defer if (norm_query_buf) |b| alloc.free(b);
        const q: []const f32 = if (self.config.metric == .cosine) blk: {
            const buf = try normalizedCopy(alloc, query);
            norm_query_buf = buf;
            break :blk buf;
        } else query;

        // --- Rank centroids by distance to query, pick nprobe nearest ---
        const centroid_dists = try alloc.alloc(Candidate, self.config.nlist);
        defer alloc.free(centroid_dists);
        var c: u32 = 0;
        while (c < self.config.nlist) : (c += 1) {
            centroid_dists[c] = .{ .dist = self.dist(self.centroidAt(c), q), .id = c };
        }
        std.sort.pdq(Candidate, centroid_dists, {}, candidateLess);
        const nprobe = @min(self.config.nprobe, self.config.nlist);

        // --- Scan probed lists, collect (internal-dist, node-id) pairs ---
        var hits: std.ArrayList(Hit) = .empty;
        defer hits.deinit(alloc);

        var p: u32 = 0;
        while (p < nprobe) : (p += 1) {
            const list_c = centroid_dists[p].id;
            const list = self.inverted_lists.items[list_c].items;
            for (list) |node_idx| {
                const v = self.vectorAt(node_idx);
                const d = self.dist(v, q);
                try hits.append(alloc, .{ .id = self.ids.items[node_idx], .score = d });
            }
        }

        if (hits.items.len == 0) {
            return try alloc.alloc(Hit, 0);
        }

        // Sort ascending by internal distance, truncate to k.
        std.sort.pdq(Hit, hits.items, {}, struct {
            fn lt(_: void, a: Hit, b: Hit) bool {
                return a.score < b.score;
            }
        }.lt);
        const out_n: usize = @min(k, hits.items.len);
        const out = try alloc.alloc(Hit, out_n);
        for (hits.items[0..out_n], 0..) |h, i| {
            // Convert internal dist to caller score:
            //   l2_squared: score = dist         (lower = nearer)
            //   ip/cosine:  score = -dist = +sim (higher = more similar)
            const score: f32 = switch (self.config.metric) {
                .l2_squared => h.score,
                .inner_product, .cosine => -h.score,
            };
            out[i] = .{ .id = h.id, .score = score };
        }
        return out;
    }

    /// Multi-threaded batched kNN search. Mirrors `FlatIndex.searchBatch`.
    /// Spawns `n_threads` workers (clamped to queries.len), distributes
    /// queries in contiguous chunks.
    pub fn searchBatch(
        self: *const IndexIVFFlat,
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
            for (queries, 0..) |q, i| {
                out[i] = try self.search(alloc, q, k);
            }
            return out;
        }

        var threads = try alloc.alloc(std.Thread, T);
        defer alloc.free(threads);

        const WorkerArgs = struct {
            self: *const IndexIVFFlat,
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
                            std.debug.panic("faiss-zig IVF searchBatch worker failed: {}", .{err});
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

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "IVF: InvalidConfig when nprobe > nlist" {
    try testing.expectError(
        Error.InvalidConfig,
        IndexIVFFlat.init(testing.allocator, 4, .{ .nlist = 4, .nprobe = 5 }),
    );
}

test "IVF: InvalidConfig when nlist == 0" {
    try testing.expectError(
        Error.InvalidConfig,
        IndexIVFFlat.init(testing.allocator, 4, .{ .nlist = 0, .nprobe = 0 }),
    );
}

test "IVF: add before train fails NotTrained" {
    var idx = try IndexIVFFlat.init(testing.allocator, 4, .{ .nlist = 2, .nprobe = 1 });
    defer idx.deinit();
    try testing.expectError(Error.NotTrained, idx.add(&[_]f32{ 1, 2, 3, 4 }));
}

test "IVF: search before train fails NotTrained" {
    var idx = try IndexIVFFlat.init(testing.allocator, 4, .{ .nlist = 2, .nprobe = 1 });
    defer idx.deinit();
    try testing.expectError(
        Error.NotTrained,
        idx.search(testing.allocator, &[_]f32{ 1, 2, 3, 4 }, 1),
    );
}

test "IVF: train + single add + find itself" {
    var idx = try IndexIVFFlat.init(
        testing.allocator,
        4,
        .{ .nlist = 2, .nprobe = 2, .n_iter = 5, .seed = 7 },
    );
    defer idx.deinit();

    // 4 training vectors → 2 clusters
    const training = [_]f32{
        0, 0, 0, 0,
        0, 0, 0, 1,
        10, 10, 10, 10,
        10, 10, 10, 11,
    };
    try idx.train(&training);
    try testing.expect(idx.isTrained());

    const v = [_]f32{ 1, 2, 3, 4 };
    const id = try idx.add(&v);
    try testing.expectEqual(@as(u32, 0), id);

    const hits = try idx.search(testing.allocator, &v, 1);
    defer testing.allocator.free(hits);
    try testing.expectEqual(@as(usize, 1), hits.len);
    try testing.expectEqual(@as(u32, 0), hits[0].id);
    try testing.expectApproxEqAbs(@as(f32, 0.0), hits[0].score, 1e-6);
}

test "IVF: well-separated clusters route to the right list" {
    // Two clusters far apart in R^2; nprobe=1 should still find the
    // correct neighbor because the query sits squarely in one cluster.
    var idx = try IndexIVFFlat.init(
        testing.allocator,
        2,
        .{ .nlist = 2, .nprobe = 1, .n_iter = 10, .seed = 11 },
    );
    defer idx.deinit();

    // Training: cluster A near (0,0), cluster B near (100,100).
    var training: [40]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        training[i * 2] = rng.float(f32);
        training[i * 2 + 1] = rng.float(f32);
    }
    while (i < 20) : (i += 1) {
        training[i * 2] = 100.0 + rng.float(f32);
        training[i * 2 + 1] = 100.0 + rng.float(f32);
    }
    try idx.train(&training);

    // Add same training set as searchable vectors.
    var j: usize = 0;
    while (j < 20) : (j += 1) {
        _ = try idx.add(training[j * 2 ..][0..2]);
    }

    // Query near (0, 0) — should retrieve a cluster-A vector.
    const q = [_]f32{ 0.5, 0.5 };
    const hits = try idx.search(testing.allocator, &q, 1);
    defer testing.allocator.free(hits);
    try testing.expectEqual(@as(usize, 1), hits.len);
    // ID < 10 means cluster-A in the way we added them.
    try testing.expect(hits[0].id < 10);
}

test "IVF: dimension mismatch on train + add + search" {
    var idx = try IndexIVFFlat.init(testing.allocator, 4, .{ .nlist = 2, .nprobe = 1 });
    defer idx.deinit();
    // Training slice not a multiple of dim → DimensionMismatch
    try testing.expectError(
        Error.DimensionMismatch,
        idx.train(&[_]f32{ 1, 2, 3 }),
    );
    // Train clean, then add with wrong-dim vector.
    const t = [_]f32{
        0, 0, 0, 0,
        1, 1, 1, 1,
    };
    try idx.train(&t);
    try testing.expectError(
        Error.DimensionMismatch,
        idx.add(&[_]f32{ 1, 2, 3 }),
    );
    _ = try idx.add(&[_]f32{ 0, 0, 0, 0 });
    try testing.expectError(
        Error.DimensionMismatch,
        idx.search(testing.allocator, &[_]f32{ 1, 2, 3 }, 1),
    );
}

test "IVF: recall vs FlatIndex on N=200 D=32 top-10" {
    const N = 200;
    const D = 32;

    var ivf = try IndexIVFFlat.init(
        testing.allocator,
        D,
        .{ .nlist = 16, .nprobe = 8, .n_iter = 10, .seed = 99 },
    );
    defer ivf.deinit();
    var flat = root.FlatIndex.init(testing.allocator, D, .l2_squared);
    defer flat.deinit();

    var prng: std.Random.DefaultPrng = .init(321);
    const rng = prng.random();
    const training = try testing.allocator.alloc(f32, N * D);
    defer testing.allocator.free(training);
    for (training) |*v| v.* = rng.float(f32);

    try ivf.train(training);

    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = training[i * D ..][0..D];
        _ = try ivf.add(v);
        _ = try flat.add(v);
    }

    var buf: [D]f32 = undefined;
    var total_recall: f32 = 0;
    var trials: usize = 0;
    while (trials < 10) : (trials += 1) {
        for (&buf) |*v| v.* = rng.float(f32);
        const hits_i = try ivf.search(testing.allocator, &buf, 10);
        defer testing.allocator.free(hits_i);
        const hits_f = try flat.search(testing.allocator, &buf, 10);
        defer testing.allocator.free(hits_f);

        var seen = std.AutoHashMap(u32, void).init(testing.allocator);
        defer seen.deinit();
        for (hits_i) |h| try seen.put(h.id, {});

        var matches: usize = 0;
        for (hits_f) |h| {
            if (seen.contains(h.id)) matches += 1;
        }
        total_recall += @as(f32, @floatFromInt(matches)) / 10.0;
    }
    const recall = total_recall / 10.0;
    // nprobe=8 / nlist=16 = scan half the lists; expect >= 0.7 recall.
    // Observed: ~0.83 on this seed.
    try testing.expect(recall >= 0.7);
}

test "IVF: searchBatch matches single-thread search" {
    const D: u32 = 8;
    var idx = try IndexIVFFlat.init(
        testing.allocator,
        D,
        .{ .nlist = 4, .nprobe = 2, .n_iter = 5, .seed = 5 },
    );
    defer idx.deinit();

    var prng: std.Random.DefaultPrng = .init(2);
    const rng = prng.random();
    const N: u32 = 50;
    const training = try testing.allocator.alloc(f32, N * D);
    defer testing.allocator.free(training);
    for (training) |*v| v.* = rng.float(f32);
    try idx.train(training);
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        _ = try idx.add(training[i * D ..][0..D]);
    }

    // 8 queries
    const Q: usize = 8;
    var qbuf: [Q * D]f32 = undefined;
    for (&qbuf) |*v| v.* = rng.float(f32);
    var qslices: [Q][]const f32 = undefined;
    for (0..Q) |qi| qslices[qi] = qbuf[qi * D ..][0..D];

    const batch = try idx.searchBatch(testing.allocator, &qslices, 5, 4);
    defer {
        for (batch) |s| testing.allocator.free(s);
        testing.allocator.free(batch);
    }

    for (0..Q) |qi| {
        const solo = try idx.search(testing.allocator, qslices[qi], 5);
        defer testing.allocator.free(solo);
        try testing.expectEqual(solo.len, batch[qi].len);
        for (solo, batch[qi]) |a, b| {
            try testing.expectEqual(a.id, b.id);
            try testing.expectApproxEqAbs(a.score, b.score, 1e-6);
        }
    }
}

test "IVFFlat: inner-product metric ranks by descending IP" {
    // Three vectors all in the [1,0,0,0] direction, different magnitudes.
    // ip(v,q) = magnitude * 10. v0 wins.
    const D: u32 = 4;
    var idx = try IndexIVFFlat.init(testing.allocator, D, .{
        .nlist = 1, .nprobe = 1, .metric = .inner_product,
    });
    defer idx.deinit();

    const training = [_]f32{
        3, 0, 0, 0,
        2, 0, 0, 0,
        1, 0, 0, 0,
    };
    try idx.train(&training);
    _ = try idx.add(&[_]f32{ 3, 0, 0, 0 });
    _ = try idx.add(&[_]f32{ 2, 0, 0, 0 });
    _ = try idx.add(&[_]f32{ 1, 0, 0, 0 });

    const q = [_]f32{ 10, 0, 0, 0 };
    const hits = try idx.search(testing.allocator, &q, 3);
    defer testing.allocator.free(hits);

    try testing.expectEqual(@as(usize, 3), hits.len);
    try testing.expectEqual(@as(u32, 0), hits[0].id);
    try testing.expectApproxEqAbs(@as(f32, 30.0), hits[0].score, 1e-3);
}

test "IVFFlat: cosine metric ranks by angle, score = cosθ" {
    const D: u32 = 4;
    var idx = try IndexIVFFlat.init(testing.allocator, D, .{
        .nlist = 1, .nprobe = 1, .metric = .cosine,
    });
    defer idx.deinit();

    const training = [_]f32{
        2, 0, 0, 0,
        5, 0, 0, 0,
        0, 1, 0, 0,
    };
    try idx.train(&training);
    _ = try idx.add(&[_]f32{ 2, 0, 0, 0 }); // id 0: cosθ = 1 vs q
    _ = try idx.add(&[_]f32{ 5, 0, 0, 0 }); // id 1: cosθ = 1 vs q
    _ = try idx.add(&[_]f32{ 0, 1, 0, 0 }); // id 2: cosθ = 0 vs q

    const q = [_]f32{ 1, 0, 0, 0 };
    const hits = try idx.search(testing.allocator, &q, 3);
    defer testing.allocator.free(hits);

    try testing.expectEqual(@as(usize, 3), hits.len);
    // ids 0 and 1 should rank above id 2
    try testing.expect(hits[0].id == 0 or hits[0].id == 1);
    try testing.expect(hits[0].score > 0.99);
    try testing.expectApproxEqAbs(@as(f32, 0.0), hits[2].score, 1e-4);
}
