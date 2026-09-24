# Exact-search performance

Use the repository's `benchmarks/exact/` tool to measure insertion, embedding
fetches, and KNN separately against a preserved baseline extension. It supports
in-memory and warm file-backed databases and records effective SQLite settings.

Float16 columns halve the vector payload and can improve scan throughput;
distance arithmetic still accumulates in float32. Evaluate ranking changes on
your own embeddings before converting an existing corpus. See
[float16 embeddings](../features/vec0.md#float16-embeddings).

The benchmark can compare float32 and float16 lookup latency on the same
workload. Its ranking-overlap results help quantify the accuracy change from
half-precision storage.
