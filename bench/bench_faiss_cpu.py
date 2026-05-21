#!/usr/bin/env python3
"""
Head-to-head reference bench using facebookresearch/faiss via faiss-cpu.

Same workload as faiss-zig's bench-flat:
    N=10000 vectors of D=128 dims
    Q=100 queries
    K=10 top results per query
    L2-squared metric

Reports ns/query and queries/sec, on the same scale as the Zig bench
output, for fair head-to-head comparison.

Usage:
    LD_LIBRARY_PATH=/usr/lib bench/bench_faiss_cpu.py
"""

import time
import numpy as np
import faiss

N = 10_000
D = 128
K = 10
Q = 100

# Deterministic RNG so both Zig and Python see the same statistical
# distribution (not the SAME vectors — we're comparing distance-loop
# throughput, not exact-match output).
rng = np.random.default_rng(seed=42)

vectors = rng.random((N, D), dtype=np.float32)
queries = rng.random((Q, D), dtype=np.float32)

# Build
build_start = time.monotonic_ns()
index = faiss.IndexFlatL2(D)
index.add(vectors)
build_end = time.monotonic_ns()
build_ns = build_end - build_start

# Warmup
distances, ids = index.search(queries[:1], K)

# Query phase — N=100 separate single-query searches, matching the
# Zig bench's per-query loop. faiss is internally batched; this exposes
# per-query latency rather than batch throughput.
query_start = time.monotonic_ns()
for i in range(Q):
    distances, ids = index.search(queries[i:i+1], K)
query_end = time.monotonic_ns()
query_ns = query_end - query_start

ns_per_query = query_ns // Q
queries_per_sec = Q * 1_000_000_000 // query_ns

# Match Zig bench output format
print(f"config")
print(f"  N      = {N}")
print(f"  D      = {D}")
print(f"  K      = {K}")
print(f"  Q      = {Q}")
print(f"build")
print(f"  total_ms = {build_ns // 1_000_000}")
print(f"  ns_per_add = {build_ns // N}")
print(f"query")
print(f"  ns_per_query    = {ns_per_query}")
print(f"  us_per_query    = {ns_per_query // 1000}")
print(f"  queries_per_sec = {queries_per_sec}")
print(f"  vectors_per_sec = {queries_per_sec * N}")
