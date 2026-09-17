# Exact-search performance

Use the repository's `benchmarks/exact/` tool to measure insertion, embedding
fetches, and KNN separately against a preserved baseline extension. It supports
in-memory and warm file-backed databases and records effective SQLite settings.

Float16 columns halve the vector payload and can improve scan throughput;
distance arithmetic still accumulates in float32. Evaluate ranking changes on
your own embeddings before converting an existing corpus. See
[float16 embeddings](../features/vec0.md#float16-embeddings).

For file-backed databases, try an explicit memory-mapping limit and verify the
value returned by your SQLite build:

```sql
pragma mmap_size=1073741824;
pragma mmap_size;
```

Measure chunk size together with mmap and cache size; larger chunks are not
always faster. Keep durability settings identical when comparing insertion
performance. Warm-cache measurements do not predict cold-disk latency.

Metadata and rowid filters are applied before vector chunks are read. Filters
that exclude whole chunks can save substantial work. Scattered matches may
still require most chunks, even at low selectivity. Partition keys can help
when application queries naturally select a partition.

Cosine search computes the query norm once and uses SIMD on supported builds.
Pre-normalizing and switching to L2 should remain an explicit application
choice: it changes stored values, adds preparation work, and half-precision
rounding affects unit norms. Benchmark it rather than assuming it is faster.
