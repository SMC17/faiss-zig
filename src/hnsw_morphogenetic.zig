//! Morphogenetic HNSW Search Substrate (Zig 0.16.0)
//!
//! DOCTRINE: Substrate First (Zero-Alloc, Zero-Copy, Cache-Locality)
//!
//! This substrate implements a zero-allocation search path for HNSW
//! utilizing the Morphogenetic SearchWorkspace.

const std = @import("std");
const morph = @import("vector_morphogenetic.zig");
const hnsw_mod = @import("hnsw.zig");

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

const topo = @import("topological_vector.zig");
pub const TopologicalFeature = topo.TopologicalFeature;

/// ZERO-ALLOCATION HNSW Search Layer.
/// Uses a pre-allocated SearchWorkspace to eliminate hot-path jitter.
pub fn searchLayerMorphogenetic(
    self: *const hnsw_mod.IndexHNSW,
    query: []const f32,
    entry_points: []const u32,
    level: u8,
    ws: *morph.SearchWorkspace,
) !void {
    ws.reset();
    const visited = ws.visited;

    for (entry_points) |ep| {
        const d = l2sq(self.vectors.items[ep * self.dim ..][0..self.dim], query);
        ws.pushCandidate(.{ .dist = d, .id = ep });
        visited[ep / 8] |= @as(u8, 1) << @intCast(ep % 8);
    }

    while (ws.popCandidate()) |c| {
        // results_heap logic (simplified for this Phase Omega preview)
        // ... (heap-tracking logic identical to original but using ws.results_heap)

        // Mock neighbor traversal to show zero-alloc path
        const neighbors = if (level == 0 and self.csr_edges != null)
            self.level0Edges(c.id)
        else
            self.connections.items[c.id].items[level].items;

        for (neighbors) |n| {
            const byte_idx = n / 8;
            const bit_mask: u8 = @as(u8, 1) << @intCast(n % 8);
            if ((visited[byte_idx] & bit_mask) != 0) continue;
            visited[byte_idx] |= bit_mask;

            const d = l2sq(self.vectors.items[n * self.dim ..][0..self.dim], query);
            ws.pushCandidate(.{ .dist = d, .id = n });
        }
    }
}
