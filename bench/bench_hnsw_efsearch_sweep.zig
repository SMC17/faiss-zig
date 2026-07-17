//! ef_search recall/throughput Pareto. With heuristic select_neighbors
//! at insertion + prune, how does ef_search trade?

const std = @import("std");
const faiss = @import("faiss");

const N: usize = 10_000;
const D: u32 = 128;
const K: usize = 10;
const Q: usize = 100;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    var idx = faiss.IndexHNSW.init(alloc, D, .{
        .M = 16,
        .ef_construction = 400,
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

    // Compute ground truth once
    const gt = try alloc.alloc([K]u32, Q);
    defer alloc.free(gt);
    for (0..Q) |qi| {
        const q = queries[qi * D ..][0..D];
        const flat_hits = try flat.search(alloc, q, K);
        defer alloc.free(flat_hits);
        for (0..K) |k| gt[qi][k] = flat_hits[k].id;
    }

    const ef_searches = [_]u32{ 100, 200, 400, 800, 1600 };
    for (ef_searches) |ef_s| {
        idx.config.ef_search = ef_s;

        // warmup
        for (0..10) |qi| {
            const q = queries[(qi % Q) * D ..][0..D];
            const w = try idx.search(alloc, q, K);
            defer alloc.free(w);
        }

        const t0 = std.Io.Clock.awake.now(io);
        var total_recall: f32 = 0;
        for (0..Q) |qi| {
            const q = queries[qi * D ..][0..D];
            const hnsw_hits = try idx.search(alloc, q, K);
            defer alloc.free(hnsw_hits);
            var matches: usize = 0;
            for (hnsw_hits) |h| {
                for (gt[qi]) |g| if (h.id == g) {
                    matches += 1;
                    break;
                };
            }
            total_recall += @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(K));
        }
        const total_ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);
        std.debug.print("ef_search={d:4}  us_per_query={d:5}  recall={d:.3}\n", .{ ef_s, total_ns / Q / 1000, total_recall / @as(f32, @floatFromInt(Q)) });
    }
}
