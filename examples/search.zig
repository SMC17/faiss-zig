//! faiss-search — demo CLI that builds a synthetic FlatIndex and
//! prints kNN results for a deterministic query.
//!
//! v0.0.1 doesn't have a serialised vector format yet; this example
//! exists to exercise the public API end-to-end and produce visible
//! output. v0.1 will add a binary file format + a real CLI that
//! reads vectors from disk.
//!
//! Usage: faiss-search [N] [D] [K]
//!   N = number of vectors (default 1000)
//!   D = dimensionality   (default 64)
//!   K = top-K to print   (default 10)

const std = @import("std");
const faiss = @import("faiss");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer iter.deinit();
    _ = iter.next();
    const n_str = iter.next() orelse "1000";
    const d_str = iter.next() orelse "64";
    const k_str = iter.next() orelse "10";

    const n = try std.fmt.parseInt(usize, n_str, 10);
    const d = try std.fmt.parseInt(u32, d_str, 10);
    const k = try std.fmt.parseInt(usize, k_str, 10);

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    var idx = faiss.FlatIndex.init(alloc, d, .l2_squared);
    defer idx.deinit();

    const vec_buf = try alloc.alloc(f32, d);
    defer alloc.free(vec_buf);

    var i: usize = 0;
    while (i < n) : (i += 1) {
        for (vec_buf) |*v| v.* = rng.float(f32);
        _ = try idx.add(vec_buf);
    }

    // Query: a fresh random vector
    for (vec_buf) |*v| v.* = rng.float(f32);
    const hits = try idx.search(alloc, vec_buf, k);
    defer alloc.free(hits);

    std.debug.print("FlatIndex N={d} D={d} metric=l2_squared\n", .{ n, d });
    std.debug.print("Top {d} hits for random query:\n", .{hits.len});
    for (hits, 0..) |h, j| {
        std.debug.print("  {d:>3}: id={d:>6}  score={d:.6}\n", .{ j, h.id, h.score });
    }
}
