# Fixtures

Every benchmark claim in this repository is bounded by the test fixtures used to produce it. The fixtures and the bench harnesses are part of the published contribution.

## `bench/bench_simd.zig` — SIMD vs scalar speedup

Inner-loop distance throughput. For each dimension (16 / 64 / 128 / 256 / 512 / 1024), measure ns/distance for `@Vector(8, f32)` vs scalar-reduction loops on N=100,000 vectors per dim, median of 3 trials, ReleaseFast.

The dimensions are chosen because each is a real production-workload class:

- 16 — low-D sanity check
- 64 / 128 — BERT-class embedding space
- 256 / 512 — image embedding (CLIP), small-model retrieval
- 1024 — sentence-transformers all-MiniLM-L6-v2 class

Curation choices:

- N=100K per dim. Large enough that per-call overhead amortises, small enough that the working set fits in L2/L3 on commodity hardware. Larger N bottlenecks on memory bandwidth and obscures the SIMD-vs-scalar lowering difference.
- Median of 3 trials. Variance under 5% per trial; median resists outliers without consuming bench time.
- ReleaseFast, x86_64-linux. Single-platform calibration. Apple M-series, ARM64, embedded targets will have different SIMD lowering ratios — those are separate benches against the same harness.

## `bench/bench_faiss_cpu.py` — end-to-end vs `facebookresearch/faiss`

Head-to-head. N=10,000 vectors, D=128, K=10 nearest, Q=100 queries, L2 distance. Compares the faiss-zig FlatIndex path to the `faiss-cpu` Python binding of canonical FAISS C++.

N=10K D=128 is the small end of "thousands to low-millions of vectors, sub-thousand-dim embeddings, latency-sensitive single-query retrieval" — the common production embedding-search workload. The Flat bench at N=10K isolates inner-loop performance from index-strategy choice. N=1M+ is the regime where approximate indexes become mandatory; that is a different bench.

Curation choices:

- K=10. K=1 bottlenecks on argmin; K=100 bottlenecks on heap maintenance. K=10 is typical RAG retrieval count and balances both costs realistically.
- L2 distance, not cosine or IP. L2 is the canonical reference, simplest hot path, dtype-agnostic. Cosine and IP are separate benches.
- `faiss-cpu` 1.13.2 specifically pinned. FAISS has shipped meaningful inner-loop changes between major versions; the comparison numbers are bound to this version.

## `bench/results.jsonl` — committed measurements

JSONL with per-bench results (ns/op, throughput, config). Committed so that the README and BENCH.md claims point at specific dated runs, not at floating assertions. A clean-room reimplementation can compare its results against ours quantitatively.

## Synthetic data for HNSW / IVFFlat / IVFPQ

Approximate-index benchmarks use synthetic 10,000-vector D=128 unit-cube random data, deterministically seeded.

Synthetic, not real embeddings, because recall numbers vary hugely between random data and real clustered embeddings. The synthetic numbers are documented as the *worst case* for PQ-style approximate indexes: uniform random data has no exploitable cluster structure. Real production embeddings recover toward FAISS's published 0.6–0.8 R@10. The synthetic floor is named explicitly so readers do not extrapolate the recall floor to their workload without caveat.

## Why reproducibility binds to the fixture

A port reproducing these numbers needs to:

1. Use the same fixture shapes — N=10K, D=128, K=10, L2 — when comparing against `facebookresearch/faiss`. Different shapes produce different ratios.
2. Pin against `faiss-cpu` 1.13.2 specifically. Different FAISS versions run different inner loops.
3. Use the same SIMD-vs-scalar harness at `bench/bench_simd.zig`. Different harness shapes — N per dim, trial counts, optimisation level — produce different ratios.
4. Use synthetic uniform-random data for the recall floor, with the explicit non-claim that real clustered embeddings will recall higher. Reporting recall on different data is a different experiment.

## Provenance, licence

Bench harnesses (`bench/bench_simd.zig`, `bench/bench_faiss_cpu.py`, per-index benches): original. AGPL-3.0-or-later.

Synthetic fixtures: deterministically generated, seed and generation code committed.

`faiss-cpu` comparison: runs against the published `faiss-cpu` Python wheel (Meta, MIT). Not redistributed; the bench harness `pip install faiss-cpu==1.13.2` at run time.

Cite the fixture and version: `faiss-zig v0.7 vs faiss-cpu 1.13.2, N=10K D=128 K=10 L2 synthetic uniform, bench/bench_faiss_cpu.py`.
