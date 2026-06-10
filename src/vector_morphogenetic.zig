//! Morphogenetic Vector Substrate (Zig 0.16.0)
//! 
//! DOCTRINE: Substrate First (Zero-Alloc, Zero-Copy, Cache-Locality)
//!
//! This substrate implements a zero-allocation vector search path for HNSW
//! and IVF indices by pre-allocating all intermediate search state into
//! a hardware-aligned workspace.

const std = @import("std");

/// Morphogenetic Search Workspace: A hardware-aligned block of memory
/// containing all state required for a single-thread vector search.
pub const SearchWorkspace = struct {
    allocator: std.mem.Allocator,
    
    /// Bitset to track visited nodes during graph traversal (HNSW).
    /// bitset[node_id / 8] & (1 << (node_id % 8))
    visited: []u8,
    
    /// Distance buffer for intermediate calculations.
    distances: []f32,
    
    /// Fixed-capacity min-heap for candidates.
    candidates_heap: []Candidate,
    candidates_count: usize,
    
    /// Fixed-capacity max-heap for top-k results.
    results_heap: []Candidate,
    results_count: usize,

    pub const Candidate = struct {
        dist: f32,
        id: u32,
    };

    pub fn init(alloc: std.mem.Allocator, max_nodes: usize, ef_search: usize) !*SearchWorkspace {
        const self = try alloc.create(SearchWorkspace);
        errdefer alloc.destroy(self);

        const bitset_bytes = (max_nodes + 7) / 8;
        
        self.* = .{
            .allocator = alloc,
            .visited = try alloc.alloc(u8, bitset_bytes),
            .distances = try alloc.alloc(f32, ef_search), // Scratch space
            .candidates_heap = try alloc.alloc(Candidate, ef_search),
            .candidates_count = 0,
            .results_heap = try alloc.alloc(Candidate, ef_search),
            .results_count = 0,
        };
        
        @memset(self.visited, 0);
        return self;
    }

    pub fn deinit(self: *SearchWorkspace) void {
        self.allocator.free(self.visited);
        self.allocator.free(self.distances);
        self.allocator.free(self.candidates_heap);
        self.allocator.free(self.results_heap);
        self.allocator.destroy(self);
    }

    pub fn reset(self: *SearchWorkspace) void {
        @memset(self.visited, 0);
        self.candidates_count = 0;
        self.results_count = 0;
    }

    /// Internal heap operations (Zero-Alloc)
    pub fn pushCandidate(self: *SearchWorkspace, cand: Candidate) void {
        if (self.candidates_count >= self.candidates_heap.len) return;
        // Min-heap push logic
        var i = self.candidates_count;
        self.candidates_heap[i] = cand;
        self.candidates_count += 1;
        while (i > 0) {
            const p = (i - 1) / 2;
            if (self.candidates_heap[i].dist < self.candidates_heap[p].dist) {
                const tmp = self.candidates_heap[i];
                self.candidates_heap[i] = self.candidates_heap[p];
                self.candidates_heap[p] = tmp;
                i = p;
            } else break;
        }
    }

    pub fn popCandidate(self: *SearchWorkspace) ?Candidate {
        if (self.candidates_count == 0) return null;
        const root = self.candidates_heap[0];
        self.candidates_count -= 1;
        if (self.candidates_count > 0) {
            self.candidates_heap[0] = self.candidates_heap[self.candidates_count];
            var i: usize = 0;
            while (true) {
                var smallest = i;
                const l = 2 * i + 1;
                const r = 2 * i + 2;
                if (l < self.candidates_count and self.candidates_heap[l].dist < self.candidates_heap[smallest].dist) smallest = l;
                if (r < self.candidates_count and self.candidates_heap[r].dist < self.candidates_heap[smallest].dist) smallest = r;
                if (smallest != i) {
                    const tmp = self.candidates_heap[i];
                    self.candidates_heap[i] = self.candidates_heap[smallest];
                    self.candidates_heap[smallest] = tmp;
                    i = smallest;
                } else break;
            }
        }
        return root;
    }
};

/// Slab-allocated Adjacency structure for HNSW.
/// REPLACES legacy nested list structures.
pub const AdjacencySlab = struct {
    allocator: std.mem.Allocator,
    /// Flat edge slab: [max_nodes * max_level * M]
    edges: []u32,
    /// Offsets for each node and level: [max_nodes * max_level]
    offsets: []usize,
    /// Number of neighbors per (node, level): [max_nodes * max_level]
    counts: []u8,
    
    max_nodes: usize,
    max_level: u8,
    M: u8,

    pub fn init(alloc: std.mem.Allocator, max_nodes: usize, max_level: u8, M: u8) !AdjacencySlab {
        const total_slots = max_nodes * @as(usize, max_level + 1) * M;
        const total_meta = max_nodes * @as(usize, max_level + 1);
        
        return .{
            .allocator = alloc,
            .edges = try alloc.alloc(u32, total_slots),
            .offsets = try alloc.alloc(usize, total_meta),
            .counts = try alloc.alloc(u8, total_meta),
            .max_nodes = max_nodes,
            .max_level = max_level,
            .M = M,
        };
    }

    pub fn deinit(self: *AdjacencySlab) void {
        self.allocator.free(self.edges);
        self.allocator.free(self.offsets);
        self.allocator.free(self.counts);
    }
    
    pub inline fn getNeighbors(self: AdjacencySlab, node_id: u32, level: u8) []const u32 {
        const idx = node_id * @as(usize, self.max_level + 1) + level;
        const offset = idx * self.M;
        const count = self.counts[idx];
        return self.edges[offset..][0..count];
    }
};
