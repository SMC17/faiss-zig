# Test fixtures — what they are, why they bound the claims

Every benchmark claim in this repository is bounded by the test
fixtures used to produce it. The fixtures and the bench harnesses
are part of the published contribution.

## Fixture inventory

### `bench/bench_simd.zig` — SIMD vs scalar speedup table

**What:** Inner-loop distance-throughput bench. For each dimension
(16 / 64 / 128 / 256 / 512 / 1024), measure ns/distance for
`@Vector(8, f32)` vs scalar-reduction loops on N=100,000 vectors per
dim, median of 3 trials, ReleaseFast.

**Why these specific dimensions:** 16 is a useful low-D sanity check;
64-128 cover the BERT-class embedding space; 256-512 cover image
embedding (CLIP) and small-model retrieval; 1024 covers
sentence-transformers all-MiniLM-L6-v2-class. Each dim was chosen
because it's a real production-workload class, not because it
flatters the bench number.

**Taste decisions:**

1. **N=100K vectors per dim.** Large enough that per-call overhead
   amortises; small enough that the working set fits in L2/L3 on
   commodity hardware. Larger N (1M+) bottlenecks on memory bandwidth
   and obscures the SIMD vs scalar lowering difference.
2. **Median of 3 trials.** Trades repeatability for compactness. The
   variance between trials is small (<5%); median resists outliers
   without consuming bench time.
3. **ReleaseFast, x86_64-linux.** Single-platform calibration. Apple
   M-series, ARM64, and embedded targets will have different SIMD
   lowering ratios — those would be separate benches against the
   same harness.

### `bench/bench_faiss_cpu.py` — end-to-end vs `facebookresearch/faiss`

**What:** Head-to-head comparison: N=10,000 vectors, D=128, K=10
nearest, Q=100 queries, L2 distance. Compares the faiss-zig FlatIndex
path to the `faiss-cpu` Python binding of the canonical FAISS C++.

**Why this specific shape (N=10K, D=128):** the most common production
workload for embedding-search is "thousands to low-millions of vectors,
sub-thousand-dim embeddings, latency-sensitive single-query retrieval."
N=10K is the small end of that range where Flat (brute-force) is still
practical; N=1M+ is the regime where approximate indexes (HNSW, IVF)
become mandatory. The Flat bench at N=10K isolates the inner-loop
performance from the index-strategy choice.

**Taste decisions:**

1. **K=10 nearest, not K=1 or K=100.** K=1 bottlenecks on argmin; K=100
   bottlenecks on heap maintenance. K=10 is the typical RAG retrieval
   count and balances both costs realistically.
2. **L2 distance, not cosine or IP.** L2 is the canonical reference;
   it has the simplest hot path and is dtype-agnostic. Cosine and IP
   are separate benches against the same harness.
3. **`faiss-cpu` 1.13.2 specifically.** Pinning the canonical version
   matters because FAISS has shipped meaningful inner-loop changes
   between major versions. The "~1.5x slower at v0.2 / ~4.5x slower at
   v0.1" numbers are bound to this specific FAISS version.

### `bench/results.jsonl` — committed bench outputs

**What:** JSONL file with per-bench measurements (ns/op, throughput,
config). Committed so that the bench claims in the README + BENCH.md
are traceable to a specific committed result file.

**Why committed:** the README claims "~5x SIMD speedup at D=128" point
at a specific dated bench run. Committing the JSONL means a clean-room
reimplementation can compare its bench results against ours
quantitatively (not just "we got similar numbers").

### IndexHNSW, IVFFlat, IVFPQ bench fixtures (synthetic)

**What:** Synthetic 10,000-vector D=128 unit-cube random data,
deterministically seeded, used for the approximate-index benchmarks.

**Why synthetic, not real embeddings:** the recall numbers vary
hugely between random data and real clustered embeddings. The
synthetic numbers are documented as a *worst case* for PQ-style
approximate indexes (because uniform random data has no exploitable
cluster structure). Real production embeddings recover toward FAISS's
published 0.6-0.8 R@10. The synthetic floor is named explicitly in
the README so readers don't extrapolate the recall floor to their
real workload without caveat.

## Why the fixtures matter for reproducibility

A port to another language reproducing the bench numbers needs to:

1. **Use the same fixture shapes** (N=10K, D=128, K=10, L2) when
   comparing against `facebookresearch/faiss`. Different shapes
   produce different ratios.
2. **Pin against `faiss-cpu` 1.13.2 specifically.** Comparison against
   different FAISS versions runs different inner loops.
3. **Use the same SIMD-vs-scalar bench harness** at `bench/bench_simd.zig`.
   Other harness shapes (different N per dim, different trial counts,
   different optimisation level) will produce different ratios.
4. **Use synthetic uniform-random data for the recall floor**, with
   the explicit non-claim that real clustered embeddings will recall
   higher. Reporting recall on different data is a different
   experiment.

## Fixture provenance + license

- The bench harnesses (`bench/bench_simd.zig`, `bench/bench_faiss_cpu.py`,
  per-index benches) are original to this project. AGPL-3.0-or-later.
- The synthetic fixtures are deterministically generated; the seed and
  generation code are committed.
- The `faiss-cpu` comparison runs against the published `faiss-cpu`
  Python wheel (Meta, MIT). We do not redistribute it; the bench
  harness `pip install faiss-cpu==1.13.2` at run time.

When citing benchmark numbers, cite the fixture: `faiss-zig v0.7 vs
faiss-cpu 1.13.2 on N=10K D=128 K=10 L2 synthetic uniform; bench at
bench/bench_faiss_cpu.py`.
