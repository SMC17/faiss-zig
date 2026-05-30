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

    // HNSW recall-vs-QPS Pareto sweep across ef_search ∈ {50,100,200,400,800}.
    const bench_hnsw_sweep_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_hnsw_sweep.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_hnsw_sweep_mod.addImport("faiss", mod);
    const bench_hnsw_sweep_exe = b.addExecutable(.{
        .name = "bench-hnsw-sweep",
        .root_module = bench_hnsw_sweep_mod,
    });
    const run_bench_hnsw_sweep = b.addRunArtifact(bench_hnsw_sweep_exe);
    run_bench_hnsw_sweep.has_side_effects = true;
    const bench_hnsw_sweep_step = b.step("bench-hnsw-sweep", "HNSW recall-vs-QPS Pareto sweep over ef_search");
    bench_hnsw_sweep_step.dependOn(&run_bench_hnsw_sweep.step);

    // Flat search with comptime-specialized D=128 distance.
    const bench_ct128_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_flat_comptime_d128.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_ct128_mod.addImport("faiss", mod);
    const bench_ct128_exe = b.addExecutable(.{
        .name = "bench-flat-comptime-d128",
        .root_module = bench_ct128_mod,
    });
    const run_bench_ct128 = b.addRunArtifact(bench_ct128_exe);
    run_bench_ct128.has_side_effects = true;
    const bench_ct128_step = b.step("bench-flat-comptime-d128", "Flat search with comptime-specialized D=128 distance vs library general-D");
    bench_ct128_step.dependOn(&run_bench_ct128.step);

    // Flat search with explicit prefetch + TILE
    const bench_pref_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_flat_prefetch_tile.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_pref_mod.addImport("faiss", mod);
    const bench_pref_exe = b.addExecutable(.{
        .name = "bench-flat-prefetch-tile",
        .root_module = bench_pref_mod,
    });
    const run_bench_pref = b.addRunArtifact(bench_pref_exe);
    run_bench_pref.has_side_effects = true;
    const bench_pref_step = b.step("bench-flat-prefetch-tile", "Flat search with @prefetch + tiled corpus access");
    bench_pref_step.dependOn(&run_bench_pref.step);

    // Hand-rolled @Vector(8, f16) half-precision matmul Flat L2.
    const bench_f16_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_flat_f16_handrolled.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_f16_exe = b.addExecutable(.{
        .name = "bench-flat-f16-handrolled",
        .root_module = bench_f16_mod,
    });
    const run_bench_f16 = b.addRunArtifact(bench_f16_exe);
    if (b.args) |args| run_bench_f16.addArgs(args);
    run_bench_f16.has_side_effects = true;
    const bench_f16_step = b.step("bench-flat-f16-handrolled", "Hand-rolled @Vector(8, f16) half-precision Flat L2 matmul");
    bench_f16_step.dependOn(&run_bench_f16.step);

    // HNSW ef_construction sweep
    const bench_ef_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_hnsw_ef_sweep.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_ef_mod.addImport("faiss", mod);
    const bench_ef_exe = b.addExecutable(.{
        .name = "bench-hnsw-ef-sweep",
        .root_module = bench_ef_mod,
    });
    const run_bench_ef = b.addRunArtifact(bench_ef_exe);
    run_bench_ef.has_side_effects = true;
    const bench_ef_step = b.step("bench-hnsw-ef-sweep", "HNSW ef_construction sweep (recall + throughput)");
    bench_ef_step.dependOn(&run_bench_ef.step);

    // HNSW ef_search sweep
    const bench_efs_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_hnsw_efsearch_sweep.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_efs_mod.addImport("faiss", mod);
    const bench_efs_exe = b.addExecutable(.{
        .name = "bench-hnsw-efsearch-sweep",
        .root_module = bench_efs_mod,
    });
    const run_bench_efs = b.addRunArtifact(bench_efs_exe);
    run_bench_efs.has_side_effects = true;
    const bench_efs_step = b.step("bench-hnsw-efsearch-sweep", "HNSW ef_search recall/throughput Pareto");
    bench_efs_step.dependOn(&run_bench_efs.step);

    // HNSW CSR A/B bench
    const bench_csr_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_hnsw_csr.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_csr_mod.addImport("faiss", mod);
    const bench_csr_exe = b.addExecutable(.{
        .name = "bench-hnsw-csr",
        .root_module = bench_csr_mod,
    });
    const run_bench_csr = b.addRunArtifact(bench_csr_exe);
    run_bench_csr.has_side_effects = true;
    const bench_csr_step = b.step("bench-hnsw-csr", "HNSW CSR A/B (nested ArrayList vs flat offsets+edges)");
    bench_csr_step.dependOn(&run_bench_csr.step);

    // Batched @Vector(8, f16) tiled corpus + Q reuse.
    const bench_f16b_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_flat_f16_batched.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_f16b_exe = b.addExecutable(.{
        .name = "bench-flat-f16-batched",
        .root_module = bench_f16b_mod,
    });
    const run_bench_f16b = b.addRunArtifact(bench_f16b_exe);
    if (b.args) |args| run_bench_f16b.addArgs(args);
    run_bench_f16b.has_side_effects = true;
    const bench_f16b_step = b.step("bench-flat-f16-batched", "Batched @Vector(8, f16) tiled corpus L1-resident x Q queries");
    bench_f16b_step.dependOn(&run_bench_f16b.step);
}
