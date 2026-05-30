//! topk_heap: v0.0.4 substrate primitive.
//!
//! FlatIndex.search currently allocates N Hits, computes every
//! score, sorts the full slice with pdq, then memcpy's the top K.
//! Cost is O(N log N) sort plus a 2N allocation. For K much
//! smaller than N (the common case), a fixed-size K-heap walks
//! the N scores in a single pass with O(N log K) work and
//! O(K) memory.
//!
//! Port as a faiss-zig substrate primitive. Generic over the
//! score direction so the same heap serves l2_squared (ascending
//! = best) and inner_product / cosine (descending = best).
//!
//! Scope of this revision:
//!   TopKDirection enum { ascending, descending }.
//!   TopKHeap fixed-capacity max-heap over Hit. Internally always
//!     a max-heap relative to "worse than"; the direction flips
//!     the comparator so the root is always the candidate to
//!     evict.
//!   init(allocator, k, direction) allocates K Hits.
//!   offer(hit) inserts if heap not yet full, otherwise replaces
//!     the root if the new hit is better. O(log K).
//!   intoSortedSlice(allocator) returns the heap contents sorted
//!     best-first. Caller owns.
//!   len() and capacity() accessors.
//!
//! Out of scope: lock-free concurrent inserts (caller composes
//! via per-thread heaps then a merge step), duplicate-id rejection
//! (caller-policy).

const std = @import("std");
const faiss = @import("root.zig");

pub const Hit = faiss.Hit;

pub const TopKDirection = enum(u8) {
    ascending = 0,
    descending = 1,
};

pub const Error = error{
    OutOfMemory,
    InvalidK,
};

pub const TopKHeap = struct {
    allocator: std.mem.Allocator,
    items: []Hit,
    len_: usize,
    direction: TopKDirection,

    pub fn init(
        allocator: std.mem.Allocator,
        k: usize,
        direction: TopKDirection,
    ) Error!TopKHeap {
        if (k == 0) return Error.InvalidK;
        const items = allocator.alloc(Hit, k) catch return Error.OutOfMemory;
        return .{
            .allocator = allocator,
            .items = items,
            .len_ = 0,
            .direction = direction,
        };
    }

    pub fn deinit(self: *TopKHeap) void {
        self.allocator.free(self.items);
    }

    pub fn len(self: TopKHeap) usize {
        return self.len_;
    }

    pub fn capacity(self: TopKHeap) usize {
        return self.items.len;
    }

    /// True iff `a` is worse than `b` per the direction. For
    /// ascending (smaller = better), "worse" means score > b's;
    /// for descending (larger = better), "worse" means score < b's.
    fn worseThan(self: TopKHeap, a: Hit, b: Hit) bool {
        return switch (self.direction) {
            .ascending => a.score > b.score,
            .descending => a.score < b.score,
        };
    }

    fn siftUp(self: *TopKHeap, from: usize) void {
        var i = from;
        while (i > 0) {
            const parent = (i - 1) / 2;
            if (self.worseThan(self.items[i], self.items[parent])) {
                const tmp = self.items[i];
                self.items[i] = self.items[parent];
                self.items[parent] = tmp;
                i = parent;
            } else break;
        }
    }

    fn siftDown(self: *TopKHeap, from: usize) void {
        var i = from;
        const n = self.len_;
        while (true) {
            const l = 2 * i + 1;
            const r = 2 * i + 2;
            var worst = i;
            if (l < n and self.worseThan(self.items[l], self.items[worst])) worst = l;
            if (r < n and self.worseThan(self.items[r], self.items[worst])) worst = r;
            if (worst == i) break;
            const tmp = self.items[i];
            self.items[i] = self.items[worst];
            self.items[worst] = tmp;
            i = worst;
        }
    }

    /// Insert if heap not yet full; otherwise replace the root
    /// (the current worst) if the offered hit is better. O(log K).
    pub fn offer(self: *TopKHeap, hit: Hit) void {
        if (self.len_ < self.items.len) {
            self.items[self.len_] = hit;
            self.len_ += 1;
            self.siftUp(self.len_ - 1);
            return;
        }
        if (self.worseThan(self.items[0], hit)) {
            self.items[0] = hit;
            self.siftDown(0);
        }
    }

    /// Allocate and return the heap contents sorted best-first
    /// per the direction. Caller owns the returned slice.
    pub fn intoSortedSlice(
        self: TopKHeap,
        allocator: std.mem.Allocator,
    ) Error![]Hit {
        const out = allocator.alloc(Hit, self.len_) catch return Error.OutOfMemory;
        @memcpy(out, self.items[0..self.len_]);
        const Ctx = struct {
            dir: TopKDirection,
            fn less(c: @This(), a: Hit, b: Hit) bool {
                return switch (c.dir) {
                    .ascending => a.score < b.score,
                    .descending => a.score > b.score,
                };
            }
        };
        std.sort.pdq(Hit, out, Ctx{ .dir = self.direction }, Ctx.less);
        return out;
    }
};

// --- Tests -----------------------------------------------------------------

const testing = std.testing;

test "init refuses k == 0" {
    try testing.expectError(Error.InvalidK, TopKHeap.init(testing.allocator, 0, .ascending));
}

test "ascending heap keeps the three smallest scores" {
    var h = try TopKHeap.init(testing.allocator, 3, .ascending);
    defer h.deinit();
    const scores = [_]f32{ 5.0, 1.0, 4.0, 2.0, 7.0, 0.5, 3.0 };
    for (scores, 0..) |s, i| h.offer(.{ .id = @intCast(i), .score = s });
    try testing.expectEqual(@as(usize, 3), h.len());
    const out = try h.intoSortedSlice(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(@as(f32, 0.5), out[0].score);
    try testing.expectEqual(@as(f32, 1.0), out[1].score);
    try testing.expectEqual(@as(f32, 2.0), out[2].score);
}

test "descending heap keeps the three largest scores" {
    var h = try TopKHeap.init(testing.allocator, 3, .descending);
    defer h.deinit();
    const scores = [_]f32{ 5.0, 1.0, 4.0, 2.0, 7.0, 0.5, 3.0 };
    for (scores, 0..) |s, i| h.offer(.{ .id = @intCast(i), .score = s });
    const out = try h.intoSortedSlice(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(@as(f32, 7.0), out[0].score);
    try testing.expectEqual(@as(f32, 5.0), out[1].score);
    try testing.expectEqual(@as(f32, 4.0), out[2].score);
}

test "heap not yet full returns all offered hits" {
    var h = try TopKHeap.init(testing.allocator, 10, .ascending);
    defer h.deinit();
    h.offer(.{ .id = 1, .score = 2.0 });
    h.offer(.{ .id = 2, .score = 1.0 });
    const out = try h.intoSortedSlice(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqual(@as(u32, 2), out[0].id);
    try testing.expectEqual(@as(u32, 1), out[1].id);
}

test "offer at capacity drops worse-than-root and keeps best" {
    var h = try TopKHeap.init(testing.allocator, 2, .descending);
    defer h.deinit();
    h.offer(.{ .id = 1, .score = 100.0 });
    h.offer(.{ .id = 2, .score = 99.0 });
    h.offer(.{ .id = 3, .score = 50.0 });
    h.offer(.{ .id = 4, .score = 101.0 });
    const out = try h.intoSortedSlice(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqual(@as(u32, 4), out[0].id);
    try testing.expectEqual(@as(u32, 1), out[1].id);
}

test "k == 1 keeps just the best score" {
    var h = try TopKHeap.init(testing.allocator, 1, .ascending);
    defer h.deinit();
    h.offer(.{ .id = 1, .score = 5.0 });
    h.offer(.{ .id = 2, .score = 2.0 });
    h.offer(.{ .id = 3, .score = 7.0 });
    const out = try h.intoSortedSlice(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(@as(usize, 1), out.len);
    try testing.expectEqual(@as(u32, 2), out[0].id);
}

test "len and capacity track offer correctly" {
    var h = try TopKHeap.init(testing.allocator, 3, .ascending);
    defer h.deinit();
    try testing.expectEqual(@as(usize, 0), h.len());
    try testing.expectEqual(@as(usize, 3), h.capacity());
    h.offer(.{ .id = 1, .score = 1.0 });
    try testing.expectEqual(@as(usize, 1), h.len());
    h.offer(.{ .id = 2, .score = 2.0 });
    h.offer(.{ .id = 3, .score = 3.0 });
    h.offer(.{ .id = 4, .score = 4.0 });
    try testing.expectEqual(@as(usize, 3), h.len());
}

test "ties on score are stable enough to keep K winners" {
    var h = try TopKHeap.init(testing.allocator, 3, .descending);
    defer h.deinit();
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    for (ids) |id| h.offer(.{ .id = id, .score = 1.0 });
    const out = try h.intoSortedSlice(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(@as(usize, 3), out.len);
    for (out) |h_| try testing.expectEqual(@as(f32, 1.0), h_.score);
}
