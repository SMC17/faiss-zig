# Ideas carried forward

## Removed 2026-11-12: "Phase Omega" (commit f0a4d02, June 2026)

An autonomous checkpoint added topological hashing, hybrid HNSW search, three
"morphogenetic" index variants, an Erlang port worker and four QPS benches.
None of it was sound: the topological hash read 128 floats from every vector
(out of bounds when dim < 128) and derived "Betti numbers" from rank digits;
`topologicalSearch` ignored its inputs; the port worker never read its
payload. It is removed; the code stays in history at f0a4d02.

Worth doing properly later:
- Topological features for retrieval: see SMC17/qsr-zig IDEAS.md.
- An Erlang/Elixir port or NIF in front of FlatIndex/IVFPQ (Zig hot path,
  BEAM control plane), with a framed stdin reader and a round-trip test.
