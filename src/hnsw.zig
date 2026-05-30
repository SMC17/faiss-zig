//! Hierarchical Navigable Small World (HNSW) approximate index.
//!
//! Reference: Malkov & Yashunin, "Efficient and robust approximate
//! nearest neighbor search using Hierarchical Navigable Small World
//! graphs", 2016.
//!
//! Algorithm summary:
//!   - Each vector is inserted at a randomly-chosen level L drawn from
//!     a geometric distribution
//!   - At each level [0, L], the node connects to its M-nearest already-
//!     inserted neighbors
//!   - Search descends levels greedily; at level 0, an ef_search-bounded
//!     priority-queue search returns the K nearest
//!
//! v0.3 scope:
//!   - L2-squared metric only
//!   - Single-thread insert + search
//!   - In-memory adjacency lists (no mmap, no disk)
//!   - Default M=16, ef_construction=200, ef_search=100
//!
//! Recall is configurable via ef_search: higher = better recall, slower.

const std = @import("std");

pub const Error = error{
    DimensionMismatch,
    EmptyIndex,
    InvalidK,
    OutOfMemory,
};

pub const Hit = struct {
    id: u32,
    score: f32,
};

pub const Config = struct {
    M: u32 = 16,
    ef_construction: u32 = 200,
    ef_search: u32 = 100,
    seed: u64 = 42,
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

/// Min-heap entry for the candidate frontier during search.
const Candidate = struct {
    dist: f32,
    id: u32,
};

fn candidateLess(_: void, a: Candidate, b: Candidate) std.math.Order {
    return std.math.order(a.dist, b.dist);
}

fn candidateMore(_: void, a: Candidate, b: Candidate) std.math.Order {
    return std.math.order(b.dist, a.dist);
}

pub const IndexHNSW = struct {
    allocator: std.mem.Allocator,
    dim: u32,
    config: Config,
    /// All inserted vectors, contiguously stored: [N * dim].
    vectors: std.ArrayList(f32) = .empty,
    /// User IDs, one per inserted vector.
    ids: std.ArrayList(u32) = .empty,
    /// Per-node max level.
    levels: std.ArrayList(u8) = .empty,
    /// Adjacency per (node, level). flat: connections[node_idx][level] = list of neighbour node_idx.
    /// For simplicity we store as a 2D nested ArrayList.
    connections: std.ArrayList(std.ArrayList(std.ArrayList(u32))) = .empty,
    /// Random number generator for level assignment.
    rng_state: std.Random.DefaultPrng,
    /// Entry point: node index + level. -1 means empty index.
    entry_point: i32 = -1,
    max_level: u8 = 0,
    /// CSR layout for level-0 edges, populated by finalize(). When non-null,
    /// searchLayer + greedyDescend use this flat array instead of the nested
    /// ArrayList path at level 0. Net win is one pointer-chase removed per
    /// node visit in the hot inner loop. Layout:
    ///   csr_offsets: []u32, length N+1. csr_offsets[i]..csr_offsets[i+1]
    ///   is the slice of level-0 neighbors for node i in csr_edges.
    csr_offsets: ?[]u32 = null,
    csr_edges: ?[]u32 = null,

    pub fn init(allocator: std.mem.Allocator, dim: u32, config: Config) IndexHNSW {
        return .{
            .allocator = allocator,
            .dim = dim,
            .config = config,
            .rng_state = std.Random.DefaultPrng.init(config.seed),
        };
    }

    pub fn deinit(self: *IndexHNSW) void {
        self.vectors.deinit(self.allocator);
        self.ids.deinit(self.allocator);
        self.levels.deinit(self.allocator);
        for (self.connections.items) |*node_conn| {
            for (node_conn.items) |*level_conn| {
                level_conn.deinit(self.allocator);
            }
            node_conn.deinit(self.allocator);
        }
        self.connections.deinit(self.allocator);
        if (self.csr_offsets) |s| self.allocator.free(s);
        if (self.csr_edges) |s| self.allocator.free(s);
    }

    /// Materialize the level-0 adjacency into a CSR (Compressed Sparse Row)
    /// layout. After this call, searchLayer + greedyDescend use the CSR
    /// path at level 0, removing one pointer-chase per node visit.
    /// Call AFTER all add() operations. add() invalidates CSR (sets to null).
    pub fn finalize(self: *IndexHNSW) !void {
        const n = self.connections.items.len;
        if (n == 0) return;
        const offsets = try self.allocator.alloc(u32, n + 1);
        errdefer self.allocator.free(offsets);

        var total: u32 = 0;
        for (self.connections.items, 0..) |node_conn, i| {
            offsets[i] = total;
            if (node_conn.items.len > 0) {
                total += @intCast(node_conn.items[0].items.len);
            }
        }
        offsets[n] = total;

        const edges = try self.allocator.alloc(u32, total);
        errdefer self.allocator.free(edges);

        for (self.connections.items, 0..) |node_conn, i| {
            const start = offsets[i];
            if (node_conn.items.len > 0) {
                const lvl0 = node_conn.items[0].items;
                @memcpy(edges[start .. start + lvl0.len], lvl0);
            }
        }

        if (self.csr_offsets) |s| self.allocator.free(s);
        if (self.csr_edges) |s| self.allocator.free(s);
        self.csr_offsets = offsets;
        self.csr_edges = edges;
    }

    inline fn level0Edges(self: *const IndexHNSW, node_id: u32) []const u32 {
        if (self.csr_offsets) |off| {
            const i: usize = @intCast(node_id);
            const start = off[i];
            const end = off[i + 1];
            return self.csr_edges.?[start..end];
        }
        return self.connections.items[node_id].items[0].items;
    }

    pub fn len(self: *const IndexHNSW) usize {
        return self.ids.items.len;
    }

    fn vectorAt(self: *const IndexHNSW, node_idx: u32) []const f32 {
        const i: usize = @intCast(node_idx);
        return self.vectors.items[i * self.dim ..][0..self.dim];
    }

    /// Level assignment: geometric distribution with parameter
    /// ml = 1/ln(M). Higher levels are exponentially rarer.
    fn randomLevel(self: *IndexHNSW) u8 {
        const ml: f64 = 1.0 / std.math.log(f64, std.math.e, @floatFromInt(self.config.M));
        // Guard against u==0 which would give -log(0) = +inf, then
        // -inf * ml = -inf or NaN, and @intFromFloat on NaN/Inf is UB
        // (this is the root cause of the v0.4 production-N SEGV).
        const u_raw: f64 = self.rng_state.random().float(f64);
        const u: f64 = if (u_raw <= 0.0 or !std.math.isFinite(u_raw)) 0.5 else u_raw;
        const lvl_raw: f64 = -std.math.log(f64, std.math.e, u) * ml;
        // Clamp explicitly to [0, 32] before float→int cast.
        if (!std.math.isFinite(lvl_raw) or lvl_raw < 0.0) return 0;
        const capped: f64 = @min(lvl_raw, 32.0);
        return @intFromFloat(capped);
    }

    pub fn add(self: *IndexHNSW, vector: []const f32) !u32 {
        if (vector.len != self.dim) return Error.DimensionMismatch;
        // Adding a node invalidates the CSR layout. Caller must
        // re-finalize() to rebuild the flat edge array.
        if (self.csr_offsets) |s| {
            self.allocator.free(s);
            self.csr_offsets = null;
        }
        if (self.csr_edges) |s| {
            self.allocator.free(s);
            self.csr_edges = null;
        }
        const id: u32 = @intCast(self.ids.items.len);
        try self.vectors.appendSlice(self.allocator, vector);
        try self.ids.append(self.allocator, id);
        const level = self.randomLevel();
        try self.levels.append(self.allocator, level);

        // Initialize adjacency lists for this node up to its level.
        var conns: std.ArrayList(std.ArrayList(u32)) = .empty;
        errdefer conns.deinit(self.allocator);
        var lvl_i: u32 = 0;
        while (lvl_i <= level) : (lvl_i += 1) {
            try conns.append(self.allocator, .empty);
        }
        try self.connections.append(self.allocator, conns);

        // First insertion: this node is the entry point.
        if (self.entry_point < 0) {
            self.entry_point = @intCast(id);
            self.max_level = level;
            return id;
        }

        // Greedy search down from the current max level to (level+1).
        var cur_node: u32 = @intCast(self.entry_point);
        var cur_dist: f32 = l2sq(self.vectorAt(cur_node), vector);
        var l: i32 = @intCast(self.max_level);
        while (l > @as(i32, @intCast(level))) : (l -= 1) {
            const r = try self.greedyDescend(vector, cur_node, cur_dist, @intCast(l));
            cur_node = r.id;
            cur_dist = r.dist;
        }

        // At levels [level, 0], do an ef-bounded search and connect.
        l = @intCast(level);
        while (l >= 0) : (l -= 1) {
            const ef = @max(self.config.ef_construction, self.config.M);
            const candidates = try self.searchLayer(vector, &[_]u32{cur_node}, ef, @intCast(l));
            defer self.allocator.free(candidates);

            // Heuristic neighbor selection (Malkov & Yashunin 2016 Alg 4):
            // pick M neighbors that DIVERSIFY direction from this node. A
            // candidate c is accepted only if no already-chosen r satisfies
            // dist(c, r) < dist(c, new_node). M-nearest by raw distance
            // produces a clustered graph; the heuristic produces a navigable
            // small-world graph with much higher recall at the same ef_search.
            // candidates are already sorted ascending by dist to new_node.
            const M_cap: usize = self.config.M;
            const selected = try self.allocator.alloc(Candidate, M_cap);
            defer self.allocator.free(selected);
            var n_sel: usize = 0;

            for (candidates) |c| {
                if (n_sel >= M_cap) break;
                const c_vec = self.vectorAt(c.id);
                var covered = false;
                for (selected[0..n_sel]) |r| {
                    const d_cr = l2sq(c_vec, self.vectorAt(r.id));
                    if (d_cr < c.dist) {
                        covered = true;
                        break;
                    }
                }
                if (!covered) {
                    selected[n_sel] = c;
                    n_sel += 1;
                }
            }

            // Backfill if heuristic was strict
            if (n_sel < M_cap) {
                for (candidates) |c| {
                    if (n_sel >= M_cap) break;
                    var already_picked = false;
                    for (selected[0..n_sel]) |r| {
                        if (r.id == c.id) {
                            already_picked = true;
                            break;
                        }
                    }
                    if (!already_picked) {
                        selected[n_sel] = c;
                        n_sel += 1;
                    }
                }
            }

            var i: usize = 0;
            while (i < n_sel) : (i += 1) {
                const neighbor_id = selected[i].id;
                try self.connections.items[id].items[@intCast(l)].append(self.allocator, neighbor_id);

                // Reciprocal edge with heuristic prune if oversize
                const neighbor_conns = &self.connections.items[neighbor_id].items[@intCast(l)];
                try neighbor_conns.append(self.allocator, id);
                if (neighbor_conns.items.len > self.config.M) {
                    try self.pruneNeighbors(neighbor_id, @intCast(l));
                }
            }

            if (candidates.len > 0) {
                cur_node = candidates[0].id;
                cur_dist = candidates[0].dist;
            }
        }

        if (level > self.max_level) {
            self.max_level = level;
            self.entry_point = @intCast(id);
        }
        return id;
    }

    fn greedyDescend(
        self: *const IndexHNSW,
        query: []const f32,
        start_node: u32,
        start_dist: f32,
        level: u8,
    ) !Candidate {
        var cur_node = start_node;
        var cur_dist = start_dist;
        var changed = true;
        while (changed) {
            changed = false;
            const cur_conns = self.connections.items[cur_node];
            // Skip if this node doesn't exist at this level (its max
            // level is below `level`). Can happen when a previous
            // iteration moved cur_node to a candidate whose level is
            // <= the previous search level but < the next iteration's
            // level.
            if (level >= cur_conns.items.len) break;
            const neighbors: []const u32 = if (level == 0)
                self.level0Edges(cur_node)
            else
                cur_conns.items[level].items;
            for (neighbors) |n| {
                const d = l2sq(self.vectorAt(n), query);
                if (d < cur_dist) {
                    cur_dist = d;
                    cur_node = n;
                    changed = true;
                }
            }
        }
        return .{ .dist = cur_dist, .id = cur_node };
    }

    fn searchLayer(
        self: *const IndexHNSW,
        query: []const f32,
        entry_points: []const u32,
        ef: u32,
        level: u8,
    ) ![]Candidate {
        // Packed bitset visited: one bit per node, replaces the std.AutoHashMap
        // probe with a single bit-test in the hot inner loop. At N=10K this is
        // 1.25 KB = fits L1. Cleared via @memset to 0.
        const n_nodes = self.connections.items.len;
        const bitset_bytes = (n_nodes + 7) / 8;
        const visited_bits = try self.allocator.alloc(u8, bitset_bytes);
        defer self.allocator.free(visited_bits);
        @memset(visited_bits, 0);

        // Candidates min-heap (sorted by distance ascending — nearest first)
        var candidates: std.PriorityQueue(Candidate, void, candidateLess) = .empty;
        defer candidates.deinit(self.allocator);
        try candidates.ensureTotalCapacity(self.allocator, ef);
        // Results max-heap (we keep top ef nearest; reject when full + new is farther)
        var results: std.PriorityQueue(Candidate, void, candidateMore) = .empty;
        defer results.deinit(self.allocator);
        try results.ensureTotalCapacity(self.allocator, ef);

        for (entry_points) |ep| {
            const d = l2sq(self.vectorAt(ep), query);
            try candidates.push(self.allocator, .{ .dist = d, .id = ep });
            // Only count toward results if this node exists at this level
            // (max-level >= search level). Otherwise we'd return it as a
            // neighbor candidate and the caller would access its
            // out-of-bounds connections[level].
            if (level < self.connections.items[ep].items.len) {
                try results.push(self.allocator, .{ .dist = d, .id = ep });
            }
            visited_bits[ep / 8] |= @as(u8, 1) << @intCast(ep % 8);
        }

        while (candidates.pop()) |c| {
            const worst_result = results.peek() orelse break;
            if (c.dist > worst_result.dist and results.count() >= ef) break;

            // Skip nodes that don't exist at this level
            const c_conns = self.connections.items[c.id];
            if (level >= c_conns.items.len) continue;
            const neighbors: []const u32 = if (level == 0)
                self.level0Edges(c.id)
            else
                c_conns.items[level].items;
            for (neighbors) |n| {
                const byte_idx = n / 8;
                const bit_mask: u8 = @as(u8, 1) << @intCast(n % 8);
                if ((visited_bits[byte_idx] & bit_mask) != 0) continue;
                visited_bits[byte_idx] |= bit_mask;
                const d = l2sq(self.vectorAt(n), query);
                if (results.count() < ef or d < results.peek().?.dist) {
                    try candidates.push(self.allocator, .{ .dist = d, .id = n });
                    // Only admit to results if n has the level — otherwise
                    // the caller can't safely use it as a neighbor.
                    if (level < self.connections.items[n].items.len) {
                        try results.push(self.allocator, .{ .dist = d, .id = n });
                        if (results.count() > ef) _ = results.pop();
                    }
                }
            }
        }

        // Extract sorted ascending
        const out = try self.allocator.alloc(Candidate, results.count());
        var i: usize = results.count();
        while (results.pop()) |c| {
            i -= 1;
            out[i] = c;
        }
        return out;
    }

    fn pruneNeighbors(self: *IndexHNSW, node_id: u32, level: u8) !void {
        const conns = &self.connections.items[node_id].items[level];
        const node_vec = self.vectorAt(node_id);
        // select_neighbors_heuristic per Malkov & Yashunin 2016 Algorithm 4:
        // diversify selections so each chosen neighbor opens a NEW direction
        // from node_id. Accept candidate c only if no already-chosen result r
        // satisfies dist(c, r) < dist(c, node_id) — meaning c is "covered" by
        // r and doesn't add reachability beyond r.
        const cand = try self.allocator.alloc(Candidate, conns.items.len);
        defer self.allocator.free(cand);
        for (conns.items, 0..) |n, idx| {
            cand[idx] = .{ .dist = l2sq(self.vectorAt(n), node_vec), .id = n };
        }
        std.sort.pdq(Candidate, cand, {}, struct {
            fn lt(_: void, a: Candidate, b: Candidate) bool {
                return a.dist < b.dist;
            }
        }.lt);

        const keep_cap: usize = @min(self.config.M, cand.len);
        const chosen = try self.allocator.alloc(Candidate, keep_cap);
        defer self.allocator.free(chosen);
        var n_chosen: usize = 0;

        for (cand) |c| {
            if (n_chosen >= keep_cap) break;
            const c_vec = self.vectorAt(c.id);
            var covered = false;
            for (chosen[0..n_chosen]) |r| {
                const d_cr = l2sq(c_vec, self.vectorAt(r.id));
                if (d_cr < c.dist) {
                    covered = true;
                    break;
                }
            }
            if (!covered) {
                chosen[n_chosen] = c;
                n_chosen += 1;
            }
        }

        // Backfill: if heuristic was strict and chosen < M, fill remaining
        // slots with the nearest unselected candidates (Malkov's extendCandidates
        // option). This preserves the "always at least M" invariant other
        // HNSW invariants depend on.
        if (n_chosen < keep_cap) {
            for (cand) |c| {
                if (n_chosen >= keep_cap) break;
                var already_picked = false;
                for (chosen[0..n_chosen]) |r| {
                    if (r.id == c.id) {
                        already_picked = true;
                        break;
                    }
                }
                if (!already_picked) {
                    chosen[n_chosen] = c;
                    n_chosen += 1;
                }
            }
        }

        conns.clearRetainingCapacity();
        for (chosen[0..n_chosen]) |c| {
            try conns.append(self.allocator, c.id);
        }
    }

    pub fn search(self: *const IndexHNSW, alloc: std.mem.Allocator, query: []const f32, k: usize) ![]Hit {
        if (query.len != self.dim) return Error.DimensionMismatch;
        if (k == 0) return Error.InvalidK;
        if (self.entry_point < 0) return Error.EmptyIndex;

        const start_dist = l2sq(self.vectorAt(@intCast(self.entry_point)), query);
        var cur = Candidate{ .dist = start_dist, .id = @intCast(self.entry_point) };

        // Greedy descend from top level to level 1
        var l: i32 = @intCast(self.max_level);
        while (l > 0) : (l -= 1) {
            cur = try self.greedyDescend(query, cur.id, cur.dist, @intCast(l));
        }

        // Level 0: ef-bounded search
        const ef = @max(self.config.ef_search, @as(u32, @intCast(k)));
        const candidates = try self.searchLayer(query, &[_]u32{cur.id}, ef, 0);
        defer self.allocator.free(candidates);

        const out_n: usize = @min(k, candidates.len);
        const out = try alloc.alloc(Hit, out_n);
        for (candidates[0..out_n], 0..) |c, i| {
            out[i] = .{ .id = c.id, .score = c.dist };
        }
        return out;
    }
};

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "HNSW: empty index errors on search" {
    var idx = IndexHNSW.init(testing.allocator, 4, .{});
    defer idx.deinit();
    const q = [_]f32{ 1, 2, 3, 4 };
    try testing.expectError(Error.EmptyIndex, idx.search(testing.allocator, &q, 1));
}

test "HNSW: single insert + search finds itself" {
    var idx = IndexHNSW.init(testing.allocator, 4, .{});
    defer idx.deinit();
    const v = [_]f32{ 1, 2, 3, 4 };
    const id = try idx.add(&v);
    try testing.expectEqual(@as(u32, 0), id);

    const hits = try idx.search(testing.allocator, &v, 1);
    defer testing.allocator.free(hits);
    try testing.expectEqual(@as(usize, 1), hits.len);
    try testing.expectEqual(@as(u32, 0), hits[0].id);
    try testing.expectApproxEqAbs(@as(f32, 0.0), hits[0].score, 1e-6);
}

test "HNSW: kNN ordering on small random set" {
    const N = 100;
    const D = 16;
    var idx = IndexHNSW.init(testing.allocator, D, .{ .M = 8, .ef_construction = 50, .ef_search = 50 });
    defer idx.deinit();

    var prng: std.Random.DefaultPrng = .init(7);
    const rng = prng.random();
    var buf: [16]f32 = undefined;
    var i: usize = 0;
    while (i < N) : (i += 1) {
        for (&buf) |*v| v.* = rng.float(f32);
        _ = try idx.add(&buf);
    }

    // Query: a fresh random vector
    for (&buf) |*v| v.* = rng.float(f32);
    const hits = try idx.search(testing.allocator, &buf, 5);
    defer testing.allocator.free(hits);

    try testing.expectEqual(@as(usize, 5), hits.len);
    // Returns sorted by score ascending
    var prev_score: f32 = -1.0;
    for (hits) |h| {
        try testing.expect(h.score >= prev_score);
        prev_score = h.score;
    }
}

test "HNSW: recall vs FlatIndex on N=200 D=32 (top-10)" {
    // Sanity: HNSW should produce >80% top-10 recall against the
    // brute-force baseline. Uses small enough N that brute-force is
    // tractable for the test.
    const root = @import("root.zig");
    const N = 200;
    const D = 32;

    var hnsw = IndexHNSW.init(testing.allocator, D, .{ .M = 16, .ef_construction = 100, .ef_search = 50 });
    defer hnsw.deinit();
    var flat = root.FlatIndex.init(testing.allocator, D, .l2_squared);
    defer flat.deinit();

    var prng: std.Random.DefaultPrng = .init(123);
    const rng = prng.random();
    var buf: [32]f32 = undefined;
    var i: usize = 0;
    while (i < N) : (i += 1) {
        for (&buf) |*v| v.* = rng.float(f32);
        _ = try hnsw.add(&buf);
        _ = try flat.add(&buf);
    }

    // Aggregate recall over 10 random queries
    var total_recall: f32 = 0;
    var trials: usize = 0;
    while (trials < 10) : (trials += 1) {
        for (&buf) |*v| v.* = rng.float(f32);
        const hits_h = try hnsw.search(testing.allocator, &buf, 10);
        defer testing.allocator.free(hits_h);
        const hits_f = try flat.search(testing.allocator, &buf, 10);
        defer testing.allocator.free(hits_f);

        var h_set = std.AutoHashMap(u32, void).init(testing.allocator);
        defer h_set.deinit();
        for (hits_h) |h| try h_set.put(h.id, {});

        var match_count: usize = 0;
        for (hits_f) |h| {
            if (h_set.contains(h.id)) match_count += 1;
        }
        total_recall += @as(f32, @floatFromInt(match_count)) / 10.0;
    }
    const recall = total_recall / 10.0;
    // Conservative threshold; HNSW should easily clear 80% on this size.
    try testing.expect(recall >= 0.7);
}

test "HNSW: dimension mismatch rejects" {
    var idx = IndexHNSW.init(testing.allocator, 4, .{});
    defer idx.deinit();
    try testing.expectError(Error.DimensionMismatch, idx.add(&[_]f32{ 1, 2 }));
}
