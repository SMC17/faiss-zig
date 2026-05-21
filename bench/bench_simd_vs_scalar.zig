//! SIMD vs scalar distance-loop throughput.
//!
//! Measures raw inner-loop time for both the @Vector(8, f32) path and
//! the explicit scalar path. We deliberately bypass the FlatIndex
//! wrapper to isolate the distance computation from sort + alloc.
//!
//! Expected result on modern x86_64 at ReleaseFast: the Zig
//! compiler auto-vectorizes the scalar loop, so SIMD-explicit and
//! scalar-as-written produce similar wall time. The @Vector code is
//! explicit insurance for compilers / targets that don't
//! auto-vectorize, and documents the intent.
//!
//! On Debug builds or on targets without 256-bit SIMD, the explicit
//! @Vector path should pull ahead measurably.

const std = @import("std");
const faiss = @import("faiss");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var prng: std.Random.DefaultPrng = .init(42);
    const rng = prng.random();

    // Test a range of dimensions including non-multiples of 8 (tail loop).
    const dims = [_]usize{ 16, 32, 64, 96, 128, 192, 256, 384, 512, 1024 };

    // Per-dim, allocate one query and N stored vectors and compute
    // distances for every pair, measuring time.
    const N: usize = 100_000;

    std.debug.print("config N={d} (vectors per dim)  trials_per_path=3 (median reported)\n", .{N});
    std.debug.print("dim  | simd_ns/dist | scalar_ns/dist | ratio (scalar/simd)\n", .{});
    std.debug.print("-----+--------------+----------------+--------------------\n", .{});

    for (dims) |D| {
        const buf = try alloc.alloc(f32, D * (N + 1));
        defer alloc.free(buf);
        for (buf) |*v| v.* = rng.float(f32);

        const query = buf[0..D];
        const stored = buf[D..];

        // Warmup
        var sink_simd: f32 = 0.0;
        var sink_scal: f32 = 0.0;
        {
            var i: usize = 0;
            while (i < N) : (i += 1) {
                sink_simd += faiss.distance.l2SquaredDistance(query, stored[i * D ..][0..D]);
                sink_scal += faiss.scalar.l2SquaredDistance(query, stored[i * D ..][0..D]);
            }
        }

        // Median of 3 trials per path
        var simd_trials = [_]u64{0} ** 3;
        var scal_trials = [_]u64{0} ** 3;
        for (0..3) |trial| {
            // SIMD trial
            {
                const t0 = std.Io.Clock.awake.now(io);
                var i: usize = 0;
                while (i < N) : (i += 1) {
                    sink_simd += faiss.distance.l2SquaredDistance(query, stored[i * D ..][0..D]);
                }
                const t1 = std.Io.Clock.awake.now(io);
                simd_trials[trial] = @intCast(t0.durationTo(t1).nanoseconds);
            }
            // Scalar trial
            {
                const t0 = std.Io.Clock.awake.now(io);
                var i: usize = 0;
                while (i < N) : (i += 1) {
                    sink_scal += faiss.scalar.l2SquaredDistance(query, stored[i * D ..][0..D]);
                }
                const t1 = std.Io.Clock.awake.now(io);
                scal_trials[trial] = @intCast(t0.durationTo(t1).nanoseconds);
            }
        }
        // Sort and take median
        std.sort.pdq(u64, &simd_trials, {}, std.sort.asc(u64));
        std.sort.pdq(u64, &scal_trials, {}, std.sort.asc(u64));
        const simd_ns = simd_trials[1] / N;
        const scal_ns = scal_trials[1] / N;
        const ratio: f64 = @as(f64, @floatFromInt(scal_ns)) / @as(f64, @floatFromInt(simd_ns));

        std.debug.print("{d:>4} | {d:>12} | {d:>14} | {d:.2}x\n", .{ D, simd_ns, scal_ns, ratio });

        // Anti-DCE
        std.mem.doNotOptimizeAway(sink_simd);
        std.mem.doNotOptimizeAway(sink_scal);
    }
}
