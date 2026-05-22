#!/usr/bin/env python3
"""
Head-to-head HNSW bench: faiss-zig v0.3 IndexHNSW vs faiss-cpu IndexHNSWFlat.

Same workload as bench_flat: N=10000 vectors of D=128 dims, Q=100
queries, K=10 top results, L2 metric. HNSW params M=16,
ef_construction=200, ef_search=100 (matching defaults on both sides).

Reports:
- Build time (insert N vectors)
- Query throughput (ns/query, queries/sec)
- Top-10 recall vs FlatIndex brute force baseline

Usage:
    LD_LIBRARY_PATH=/usr/lib python3 bench/bench_hnsw_vs_faiss_cpu.py
"""

import time
import numpy as np
import faiss

N = 10_000
D = 128
K = 10
Q = 100

M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 100

rng = np.random.default_rng(seed=42)
vectors = rng.random((N, D), dtype=np.float32)
queries = rng.random((Q, D), dtype=np.float32)

# --- Build: Flat baseline (for recall reference) ---
flat = faiss.IndexFlatL2(D)
flat.add(vectors)

# --- Build: HNSW ---
build_start = time.monotonic_ns()
hnsw = faiss.IndexHNSWFlat(D, M)
hnsw.hnsw.efConstruction = EF_CONSTRUCTION
hnsw.hnsw.efSearch = EF_SEARCH
hnsw.add(vectors)
build_ns = time.monotonic_ns() - build_start

# --- Warmup ---
_, _ = hnsw.search(queries[:1], K)
_, _ = flat.search(queries[:1], K)

# --- Query: HNSW ---
hnsw_start = time.monotonic_ns()
hnsw_d, hnsw_i = hnsw.search(queries, K)
hnsw_ns = time.monotonic_ns() - hnsw_start

# --- Query: Flat ---
flat_start = time.monotonic_ns()
flat_d, flat_i = flat.search(queries, K)
flat_ns = time.monotonic_ns() - flat_start

# --- Recall vs Flat ---
total_recall = 0.0
for q_idx in range(Q):
    hnsw_set = set(hnsw_i[q_idx].tolist())
    flat_set = set(flat_i[q_idx].tolist())
    total_recall += len(hnsw_set & flat_set) / K
avg_recall = total_recall / Q

print(f"=== faiss-cpu HNSW vs Flat (N={N}, D={D}, K={K}, Q={Q}) ===")
print()
print(f"Build phase:")
print(f"  HNSW build_ms        = {build_ns // 1_000_000}")
print(f"  HNSW ns_per_add      = {build_ns // N}")
print()
print(f"Query phase:")
print(f"  HNSW ns_per_query    = {hnsw_ns // Q}")
print(f"  HNSW us_per_query    = {hnsw_ns // Q // 1000}")
print(f"  HNSW queries_per_sec = {Q * 1_000_000_000 // hnsw_ns}")
print()
print(f"  Flat ns_per_query    = {flat_ns // Q}")
print(f"  Flat us_per_query    = {flat_ns // Q // 1000}")
print(f"  Flat queries_per_sec = {Q * 1_000_000_000 // flat_ns}")
print()
print(f"  HNSW speedup vs Flat = {flat_ns / hnsw_ns:.2f}x")
print(f"  HNSW top-{K} recall  = {avg_recall:.3f}")
