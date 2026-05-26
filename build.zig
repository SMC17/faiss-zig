const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("faiss", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tests = b.addTest(.{
        .root_module = mod,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Search CLI: search <vectors.bin> <query.bin> --k 10
    const search_mod = b.createModule(.{
        .root_source_file = b.path("examples/search.zig"),
        .target = target,
        .optimize = optimize,
    });
    search_mod.addImport("faiss", mod);
    const search_exe = b.addExecutable(.{
        .name = "faiss-search",
        .root_module = search_mod,
    });
    b.installArtifact(search_exe);
    const run_search = b.addRunArtifact(search_exe);
    if (b.args) |args| run_search.addArgs(args);
    const search_step = b.step("search", "Run the kNN search CLI");
    search_step.dependOn(&run_search.step);

    // Flat-index brute-force bench. ReleaseFast.
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_flat.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("faiss", mod);
    const bench_exe = b.addExecutable(.{
        .name = "bench-flat",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    run_bench.has_side_effects = true;
    const bench_step = b.step("bench", "Flat-index kNN throughput bench");
    bench_step.dependOn(&run_bench.step);

    // SIMD vs scalar speedup measurement.
    const bench_simd_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_simd_vs_scalar.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_simd_mod.addImport("faiss", mod);
    const bench_simd_exe = b.addExecutable(.{
        .name = "bench-simd",
        .root_module = bench_simd_mod,
    });
    const run_bench_simd = b.addRunArtifact(bench_simd_exe);
    run_bench_simd.has_side_effects = true;
    const bench_simd_step = b.step("bench-simd", "SIMD vs scalar speedup bench");
    bench_simd_step.dependOn(&run_bench_simd.step);

    // Multi-thread scaling bench.
    const bench_mt_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_multithread.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mt_mod.addImport("faiss", mod);
    const bench_mt_exe = b.addExecutable(.{
        .name = "bench-multithread",
        .root_module = bench_mt_mod,
    });
    const run_bench_mt = b.addRunArtifact(bench_mt_exe);
    run_bench_mt.has_side_effects = true;
    const bench_mt_step = b.step("bench-multithread", "Thread-scaling bench");
    bench_mt_step.dependOn(&run_bench_mt.step);

    // HNSW bench. ReleaseFast for production timing.
    const bench_hnsw_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_hnsw.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_hnsw_mod.addImport("faiss", mod);
    const bench_hnsw_exe = b.addExecutable(.{
        .name = "bench-hnsw",
        .root_module = bench_hnsw_mod,
    });
    const run_bench_hnsw = b.addRunArtifact(bench_hnsw_exe);
    run_bench_hnsw.has_side_effects = true;
    const bench_hnsw_step = b.step("bench-hnsw", "HNSW bench at N=10000 D=128");
    bench_hnsw_step.dependOn(&run_bench_hnsw.step);

    // IVFFlat bench. ReleaseFast for production timing.
    const bench_ivf_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_ivf.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_ivf_mod.addImport("faiss", mod);
    const bench_ivf_exe = b.addExecutable(.{
        .name = "bench-ivf",
        .root_module = bench_ivf_mod,
    });
    const run_bench_ivf = b.addRunArtifact(bench_ivf_exe);
    run_bench_ivf.has_side_effects = true;
    const bench_ivf_step = b.step("bench-ivf", "IVFFlat bench at N=10000 D=128 nlist=100 nprobe=10");
    bench_ivf_step.dependOn(&run_bench_ivf.step);

    // IVFPQ bench. ReleaseFast for production timing.
    const bench_ivf_pq_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_ivf_pq.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_ivf_pq_mod.addImport("faiss", mod);
    const bench_ivf_pq_exe = b.addExecutable(.{
        .name = "bench-ivf-pq",
        .root_module = bench_ivf_pq_mod,
    });
    const run_bench_ivf_pq = b.addRunArtifact(bench_ivf_pq_exe);
    run_bench_ivf_pq.has_side_effects = true;
    const bench_ivf_pq_step = b.step("bench-ivf-pq", "IVFPQ bench at N=10000 D=128 M=8 nlist=100 nprobe=10");
    bench_ivf_pq_step.dependOn(&run_bench_ivf_pq.step);
}
