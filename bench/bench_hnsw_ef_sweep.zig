const std = @import("std");
const faiss = @import("faiss");

const N: usize = 10_000;
const D: u32 = 128;
const K: usize = 10;
const Q: usize = 100;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const ef_consts = [_]u32{ 100, 200, 400, 800 };
    for (ef_consts) |ef_c| {
        var prng: std.Random.DefaultPrng = .init(42);
        const rng = prng.random();

        var idx = faiss.IndexHNSW.init(alloc, D, .{
            .M = 16,
            .ef_construction = ef_c,
            .ef_search = 100,
            .seed = 42,
        });
        defer idx.deinit();

        const vec_buf = try alloc.alloc(f32, D);
        defer alloc.free(vec_buf);

        var i: usize = 0;
        while (i < N) : (i += 1) {
            for (vec_buf) |*v| v.* = rng.float(f32);
            _ = try idx.add(vec_buf);
        }

        var flat = faiss.FlatIndex.init(alloc, D, .l2_squared);
        defer flat.deinit();
        var prng2: std.Random.DefaultPrng = .init(42);
        const rng2 = prng2.random();
        var ii: usize = 0;
        while (ii < N) : (ii += 1) {
            for (vec_buf) |*v| v.* = rng2.float(f32);
            _ = try flat.add(vec_buf);
        }

        const queries = try alloc.alloc(f32, Q * D);
        defer alloc.free(queries);
        for (queries) |*v| v.* = rng.float(f32);

        const qs = std.Io.Clock.awake.now(io);
        var qi: usize = 0;
        var total_recall: f32 = 0;
        while (qi < Q) : (qi += 1) {
            const q = queries[qi * D ..][0..D];
            const hnsw_hits = try idx.search(alloc, q, K);
            defer alloc.free(hnsw_hits);
            const flat_hits = try flat.search(alloc, q, K);
            defer alloc.free(flat_hits);
            var h_set = std.AutoHashMap(u32, void).init(alloc);
            defer h_set.deinit();
            for (hnsw_hits) |h| try h_set.put(h.id, {});
            var matches: usize = 0;
            for (flat_hits) |h| if (h_set.contains(h.id)) { matches += 1; };
            total_recall += @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(K));
        }
        const total_ns: u64 = @intCast(qs.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
        std.debug.print("ef_construction={d}  us_per_query={d}  recall={d:.3}\n", .{ ef_c, total_ns / Q / 1000, total_recall / @as(f32, @floatFromInt(Q)) });
    }
}
