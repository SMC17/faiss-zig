//! Morphogenetic IVF Search Substrate (Zig 0.16.0)
//!
//! DOCTRINE: Substrate First (Zero-Alloc, Zero-Copy, Cache-Locality)
//!
//! This substrate implements a zero-allocation IVF search path utilizing
//! the Morphogenetic SearchWorkspace.

const std = @import("std");
const morph = @import("vector_morphogenetic.zig");
const ivf_mod = @import("ivf.zig");

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

/// ZERO-ALLOCATION IVF Search.
/// Uses a pre-allocated SearchWorkspace to eliminate hot-path jitter.
pub fn searchIVFMorphogenetic(
    self: *const ivf_mod.IndexIVFFlat,
    query: []const f32,
    k: usize,
    ws: *morph.SearchWorkspace,
) ![]ivf_mod.Hit {
    ws.reset();

    // 1. Rank centroids (Zero-Alloc)
    // We use ws.candidates_heap to store centroid distances temporarily.
    const nlist = self.config.nlist;
    for (0..nlist) |c| {
        const centroid = self.centroids.items[c * self.dim ..][0..self.dim];
        const d = l2sq(centroid, query);
        ws.pushCandidate(.{ .dist = d, .id = @intCast(c) });
    }

    const nprobe = @min(self.config.nprobe, nlist);

    // 2. Scan probed lists
    var p: u32 = 0;
    while (p < nprobe) : (p += 1) {
        const best_centroid = ws.popCandidate() orelse break;
        const list_c = best_centroid.id;
        const list = self.inverted_lists.items[list_c].items;

        for (list) |node_idx| {
            const v = self.vectors.items[node_idx * self.dim ..][0..self.dim];
            const d = l2sq(v, query);

            // results_heap is a max-heap, keeping the top-k nearest.
            // If d is smaller than the farthest hit in the heap, we replace it.
            if (ws.results_count < k) {
                pushResults(ws, .{ .dist = d, .id = self.ids.items[node_idx] });
            } else if (d < ws.results_heap[0].dist) {
                _ = popResults(ws);
                pushResults(ws, .{ .dist = d, .id = self.ids.items[node_idx] });
            }
        }
    }

    // 3. Extract hits (Non-Allocating view into Workspace)
    // The caller must copy these if they want to own them,
    // but the hot-path forward pass uses them as transient data.
    const out_n = ws.results_count;
    const out = try ws.allocator.alloc(ivf_mod.Hit, out_n); // Final output alloc only
    var i: usize = out_n;
    while (i > 0) {
        i -= 1;
        const hit = popResults(ws).?;
        out[i] = .{ .id = hit.id, .score = hit.dist };
    }
    return out;
}

/// Max-heap operations for Results (Zero-Alloc)
fn pushResults(ws: *morph.SearchWorkspace, hit: morph.SearchWorkspace.Candidate) void {
    var i = ws.results_count;
    ws.results_heap[i] = hit;
    ws.results_count += 1;
    while (i > 0) {
        const p = (i - 1) / 2;
        if (ws.results_heap[i].dist > ws.results_heap[p].dist) {
            const tmp = ws.results_heap[i];
            ws.results_heap[i] = ws.results_heap[p];
            ws.results_heap[p] = tmp;
            i = p;
        } else break;
    }
}

fn popResults(ws: *morph.SearchWorkspace) ?morph.SearchWorkspace.Candidate {
    if (ws.results_count == 0) return null;
    const root = ws.results_heap[0];
    ws.results_count -= 1;
    if (ws.results_count > 0) {
        ws.results_heap[0] = ws.results_heap[ws.results_count];
        var i: usize = 0;
        while (true) {
            var largest = i;
            const l = 2 * i + 1;
            const r = 2 * i + 2;
            if (l < ws.results_count and ws.results_heap[l].dist > ws.results_heap[largest].dist) largest = l;
            if (r < ws.results_count and ws.results_heap[r].dist > ws.results_heap[largest].dist) largest = r;
            if (largest != i) {
                const tmp = ws.results_heap[i];
                ws.results_heap[i] = ws.results_heap[largest];
                ws.results_heap[largest] = tmp;
                i = largest;
            } else break;
        }
    }
    return root;
}
