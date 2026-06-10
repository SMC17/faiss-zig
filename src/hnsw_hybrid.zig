//! Morphogenetic Hybrid HNSW Search (Zig 0.16.0)
//! 
//! DOCTRINE: Phase Omega (Hybrid Structural-Euclidean Search)
//!
//! This substrate implements a hybrid search path where topological invariants
//! act as a structural filter before Euclidean distance refinement.

const std = @import("std");
const morph = @import("vector_morphogenetic.zig");
const hnsw_mod = @import("hnsw.zig");
const topo = @import("topological_vector.zig");

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

/// Hybrid Structural-Euclidean Distance
/// Performs a 'Truth-Filter' via topological hash before refining with L2.
fn hybridDistance(
    query_vec: []const f32,
    query_topo: topo.TopologicalFeature,
    target_vec: []const f32,
    target_topo: topo.TopologicalFeature,
) f32 {
    const d_topo = query_topo.fastDistance(target_topo);
    
    // Truth Filter: Discard structural mismatches (Threshold=4 bits)
    if (d_topo > 4.0) return std.math.floatMax(f32);
    
    // Refine with Euclidean
    return l2sq(query_vec, target_vec);
}

pub const SearchResult = struct { id: u32, dist: f32 };

/// Hybrid Greedy Descent through the graph levels.
fn greedyDescendHybrid(
    self: *const hnsw_mod.IndexHNSW,
    query: []const f32,
    query_topo: topo.TopologicalFeature,
    cur_id: u32,
    cur_dist: f32,
    level: u8,
) !SearchResult {
    var best_id = cur_id;
    var best_dist = cur_dist;
    const hashes = self.topo_hashes.?;

    while (true) {
        var changed = false;
        const neighbors: []const u32 = if (level == 0 and self.csr_edges != null)
            self.level0Edges(best_id)
        else
            self.connections.items[best_id].items[level].items;

        for (neighbors) |n| {
            const d = hybridDistance(query, query_topo, self.vectors.items[n * self.dim ..][0..self.dim], hashes[n]);
            if (d < best_dist) {
                best_dist = d;
                best_id = n;
                changed = true;
            }
        }
        if (!changed) break;
    }
    return SearchResult{ .id = best_id, .dist = best_dist };
}

/// ZERO-ALLOCATION Hybrid HNSW Search Layer.
pub fn searchLayerHybrid(
    self: *const hnsw_mod.IndexHNSW,
    query: []const f32,
    query_topo: topo.TopologicalFeature,
    entry_points: []const u32,
    level: u8,
    ws: *morph.SearchWorkspace,
) !void {
    ws.reset();
    const visited = ws.visited;
    const hashes = self.topo_hashes.?;

    for (entry_points) |ep| {
        const d = hybridDistance(query, query_topo, self.vectors.items[ep * self.dim ..][0..self.dim], hashes[ep]);
        ws.pushCandidate(.{ .dist = d, .id = ep });
        visited[ep / 8] |= @as(u8, 1) << @intCast(ep % 8);
    }

    while (ws.popCandidate()) |c| {
        if (c.dist == std.math.floatMax(f32)) continue; // Early discard

        const neighbors = if (level == 0 and self.csr_edges != null) 
            self.level0Edges(c.id)
        else 
            self.connections.items[c.id].items[level].items;

        for (neighbors) |n| {
            const byte_idx = n / 8;
            const bit_mask: u8 = @as(u8, 1) << @intCast(n % 8);
            if ((visited[byte_idx] & bit_mask) != 0) continue;
            visited[byte_idx] |= bit_mask;
            
            const d = hybridDistance(query, query_topo, self.vectors.items[n * self.dim ..][0..self.dim], hashes[n]);
            ws.pushCandidate(.{ .dist = d, .id = n });
        }
    }
}

/// Full Hybrid HNSW Search (Multi-level).
pub fn searchHybrid(
    self: *const hnsw_mod.IndexHNSW,
    query: []const f32,
    query_topo: topo.TopologicalFeature,
    k: usize,
    ws: *morph.SearchWorkspace,
) !void {
    if (self.entry_point < 0) return;
    
    const ep_idx: usize = @intCast(self.entry_point);
    const start_dist = hybridDistance(query, query_topo, self.vectors.items[ep_idx * self.dim ..][0..self.dim], self.topo_hashes.?[ep_idx]);
    var cur = SearchResult{ .id = @as(u32, @intCast(self.entry_point)), .dist = start_dist };

    // 2. Greedy descend from top level to level 1
    var l: i32 = @intCast(self.max_level);
    while (l > 0) : (l -= 1) {
        cur = try greedyDescendHybrid(self, query, query_topo, cur.id, cur.dist, @intCast(l));
    }

    // 3. Level 0: ef-bounded search
    try searchLayerHybrid(self, query, query_topo, &[_]u32{cur.id}, @intCast(l), ws);
    _ = k;
}
