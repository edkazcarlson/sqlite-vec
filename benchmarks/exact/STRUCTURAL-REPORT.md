# Structural exact-search experiments

This round adds an opt-in `exact_va()` L2 index for fp32 and fp16. It scans
compact int8 reconstructions with conservative distance bounds and reads the
original vector whenever the bound cannot rule it out. It uses no ANN candidate
budget, probe limit, or approximate stopping rule. Returned scores and embeddings
use the original stored values.

The final 100k-vector fp32 file-backed confirmation improved from **27.803 to
9.183 ms (3.03x)**. At one million vectors, the prototype improved from
**227.418 to 78.841 ms (2.88x)**. The 100k fp16 file-backed comparison improved
from **17.300 to 9.857 ms (1.75x)**. All timed queries matched their isolated
baseline. The cost is **2.57x fp32 / 2.81x fp16 database size**, with roughly
1.6x / 2.1x insertion time in the initial 100k file runs.

Sparse candidate reads are useful for memory-only selective workloads: a 1%
metadata filter fell from **14.853 to 2.658 ms (5.59x)**. The same change slowed
the file-backed case from 25.794 to 38.200 ms. It remains an explicit build
option, not a default. BLOB-handle reuse, rowid routing and a global heap did
not establish a broad end-to-end improvement worth enabling by default.

The comparison baseline is the already optimized implementation at `c399333`,
including fp16 support, cosine SIMD, filter-first execution and chunk top-k
selection. These gains are additional to the earlier [fp16 report](REPORT.md).
See [build instructions, SQL usage and bound derivation](STRUCTURAL.md).

## Measurement method

AMD Ryzen 9 9900X, approximately 30 GiB RAM, Linux x86-64, GCC 15.2, SQLite
3.50.4; CPU only. Real embeddings come from the Cohere benchmark parquet files
in the [Zilliz benchmark mirror](https://assets.zilliz.com/benchmark/cohere_medium_1m/train.parquet).
Reports preserve source and array hashes. The main screen uses the first 100,000
768-dimensional training vectors and 100 held-out test queries after five
warmups, k=10. Synthetic controls deliberately cover clustered, Gaussian and
anisotropic distributions.

Tables report the median of repetition medians; p95 columns report the median
of each repetition's p95. Real-data comparisons use five repetitions with
alternating variant order. Each SQL build/repetition runs in a fresh process.
Every timed SQL query must return exactly the baseline IDs and distance bits.
Native exact methods additionally check against float64 distances over the
stored dtype, allowing near ties from the production fp32 arithmetic.

File tests use 4 KiB pages, 64 MiB SQLite caches, WAL/FULL, mmap disabled, and
warm OS caches. Insertions commit every 1,000 rows. They include index maintenance
and per-row BLOB materialization, but exclude dtype conversion and the subsequent
checkpoint. Mutation
tests update or delete 1,000 rows. These are single-query latency measurements,
not concurrent throughput or cold-disk tests. Background desktop activity can
still affect results; small changes are not sufficient evidence for promotion.

Initial SQL measurements that loaded several extension builds into one process
were invalidated: exported C symbols can interpose across builds. Those reports
are excluded. The replacement runner isolates workers and checks cross-process
answer hashes. Interrupted runs retain completed workers and validate inputs
and binary hashes before resuming. Frozen build manifests, rather than the
older generated `vec_debug()` commit string, identify the tested sources.

## Why this index won

Coordinate-wise interval bounds eliminated almost every full distance calculation
but spent too long on bound evaluation. Replacing those lookups with contiguous
signed int8 dot products made the bounds cheap enough. On the 100k real-data
native screen, residual bounds required about 845 full scores per query instead
of 100,000: fp32 fell from 6.092 to 2.975 ms and fp16 from 3.341 to 2.937 ms.
The SQLite benefit can be larger because it also avoids reading most original
vector bytes.

The [one-million-vector native check](results/structural-residual-million.json)
also improved from 67.143 to 27.483 ms (2.44x), with exact results. These native
times exclude SQLite and should not be compared directly with SQL latency.

Each vector has an independent scale and reconstruction-error radius. Inserts
and updates do not retrain a codebook. The persistent format stores compact
chunk summaries plus a row-addressable original-vector copy, retaining the
existing chunk vectors for compatibility. This duplicates vector payloads and
makes writes slower. The index stays experimental and disabled by default.

SQLite stores large payloads across linked overflow pages. That makes repeated
small ranges inside large vector chunks a plausible source of overhead; the
benchmarks establish the regression, but do not independently attribute every
cycle to overflow traversal. The row-addressable copy avoids that access pattern.
See SQLite's [overflow-page format](https://www.sqlite.org/fileformat.html#cell_payload_overflow_pages)
and [BLOB-handle reuse API](https://www.sqlite.org/c3ref/blob_reopen.html).

The separate [instrumented run](results/isolated-instrumentation.json) supports
the access-pattern explanation. For a 1% metadata filter, sparse reads reduced
requested vector bytes from 308.3 MB to 3.1 MB, but increased read calls from
98 to 1,000 and measured read-stage time from 25.64 to 35.72 ms. For unfiltered
exact-index search, only 871 of 100,000 vectors were fully scored on average;
requested index-plus-vector bytes were 82.2 MB instead of 308.3 MB.

Those counters cover vector/index BLOB requests, not physical disk traffic or
metadata reads. The index's distance-stage time includes bounds and survivor
fetches, while its heap work is fused into that stage. Instrumentation uses 20
queries and one repetition and adds clock overhead; its timings are diagnostic,
not the latency numbers used to select a winner.

## Other approaches tested

| Native method, 100k Cohere / L2 | FP32 ms | FP16 ms | Decision |
|---|---:|---:|---|
| Production kernels + separate top-k | 6.695 | 3.410 | Reference for this screen |
| Partial L2, early abandon | 44.450 | 71.994 | Bound work costs more than saved scoring |
| Coordinate intervals + AVX2 gathers | 33.115 | 32.915 | Strong pruning, expensive evaluation |
| Exact cell/radius traversal | 16.670 | 14.895 | Almost no pruning on real embeddings |
| Eight-vector transposed tiles | 6.176 | 6.172 | Kernel only; not an exact end-to-end result |
| Fused global top-k heap | 6.170 | 3.277 | Modest native gain; SQL ablation required |

These numbers are a separate run from the residual-bound experiment above;
compare methods against the baseline in their own raw report. Transposed tiles
widen fp16 to fp32 and change reduction order, so they do not qualify for exact
production integration on this evidence.

The exact cell/radius method was excellent on deliberately clustered synthetic
vectors: 6.337 to 0.494 ms, about 12.8x faster. It scored approximately 7,500 rows
there, but nearly all 100,000 real embeddings. Gaussian and anisotropic controls
also lost. No persistent cell index is added on the strength of the synthetic win.

Cached cosine norms preserve the original reduction order and measured
6.292 to 6.074 ms for fp32 and 3.523 to 3.046 ms for fp16. The latter is a 15.7%
speedup, but this is a native result without SQLite storage/maintenance
costs. A persistent cosine format is not justified yet. Pre-normalizing inputs
is a separate user-visible numerical contract; it does not preserve the original
embedding bytes and cosine score bits, so it is not silently substituted.

Raw screens: [fp32 L2](results/structural-native-f32-l2.json),
[fp16 L2](results/structural-native-f16-l2.json),
[fp32 cosine](results/structural-native-f32-cosine.json),
[fp16 cosine](results/structural-native-f16-cosine.json),
[clustered](results/structural-native-clustered.json),
[Gaussian](results/structural-native-gaussian.json),
[anisotropic](results/structural-native-anisotropic.json),
[fp32 residual](results/structural-residual-f32.json),
[fp16 residual](results/structural-residual-f16.json).

## Validation and limits

The full SQL suite with IVF, the exact index and all flat-path experiment switches enabled passed
386 tests (four
unrelated skips); the C unit suite passed. The native research suite passed 17
tests. Native and SQLite lifecycle drivers passed AddressSanitizer,
UndefinedBehaviorSanitizer and leak checking. All 19 index-specific tests also
passed against a portable scalar amalgamation. Tests cover fp16/fp32 tails,
zeros, duplicates, adjacent values, extreme finite values and subnormals, plus
filters, distance restrictions, mutation, rollback, reconnect, rename/drop,
text primary keys, attached databases, quoted names and malformed BLOB lengths.

The index currently supports L2, fp16/fp32 and at most 8192 dimensions. Fast-math
builds are rejected. Bounds use conservative rounding margins and strict
comparisons; ambiguous candidates always receive the original full scorer.
Quantized codes are used only for rejection, never returned as scores or
embeddings. This is still an experimental format: the test evidence is not a
formal floating-point proof or a guarantee of a win on other distributions.

The final shadow-table names are `*_exactvachunksNN` and
`*_exactvavectorsNN`, registered with SQLite as shadow tables. Early frozen
`structural-va-v2` builds used different internal names; they are benchmark
prototypes, not an upgrade format. The final confirmation build uses the names
in the current source. Existing flat tables require explicit copy into a new
indexed table; there is no implicit migration.

## Next useful experiments

1. Replace duplicated chunk and row payloads with a single layout that supports
   both sequential reads and indexed survivor fetches. The measured storage and
   write amplification make this the highest-value follow-up; it requires an
   explicit format/migration design.
2. Order bound evaluation by a cheap coarse score to establish a useful top-k
   threshold earlier, while still visiting every candidate not safely excluded.
   Measure the ordering overhead and preserve tie semantics.
3. Split reconstruction error across a small number of subspaces for tighter
   bounds on difficult distributions. Benchmark extra code/scale traffic against
   fewer survivor reads; the failed coordinate-wise method shows why tighter
   bounds alone are insufficient.
4. Consider cached cosine norms only with an end-to-end SQLite prototype and
   mutation costs. The native fp16 gain is promising but not yet a measured SQL
   improvement.

<!-- GENERATED MEASUREMENTS -->
## End-to-end exact-index results

All values below are milliseconds unless specified. `100%` means unfiltered;
other percentages select scattered rows. Each row compares builds within the
same run. Early `v2` and final confirmation runs are identified separately.

| Run | Restriction | Flat median / p95 | Index median / p95 | Speedup |
|---|---|---:|---:|---:|
| [100k fp32 file k=10](results/isolated-va-f32-file.json) | metadata 1% | 25.127 / 25.856 | 6.488 / 6.688 | 3.87x |
| [100k fp32 file k=10](results/isolated-va-f32-file.json) | rowid 1% | 26.616 / 27.497 | 7.900 / 8.106 | 3.37x |
| [100k fp32 file k=10](results/isolated-va-f32-file.json) | metadata 10% | 25.500 / 26.182 | 6.969 / 7.217 | 3.66x |
| [100k fp32 file k=10](results/isolated-va-f32-file.json) | rowid 10% | 28.347 / 29.023 | 9.642 / 10.053 | 2.94x |
| [100k fp32 file k=10](results/isolated-va-f32-file.json) | metadata 100% | 28.314 / 29.032 | 9.444 / 10.192 | 3.00x |
| [100k fp16 file k=10](results/isolated-va-f16-file.json) | metadata 1% | 13.007 / 13.734 | 6.518 / 7.163 | 2.00x |
| [100k fp16 file k=10](results/isolated-va-f16-file.json) | rowid 1% | 14.354 / 14.920 | 8.074 / 8.949 | 1.78x |
| [100k fp16 file k=10](results/isolated-va-f16-file.json) | metadata 10% | 13.210 / 13.699 | 7.044 / 7.428 | 1.88x |
| [100k fp16 file k=10](results/isolated-va-f16-file.json) | rowid 10% | 16.305 / 17.984 | 9.963 / 10.638 | 1.64x |
| [100k fp16 file k=10](results/isolated-va-f16-file.json) | metadata 100% | 17.300 / 17.984 | 9.857 / 10.794 | 1.75x |
| [100k fp32 memory k=10](results/isolated-va-f32-memory.json) | metadata 1% | 14.224 / 14.355 | 4.082 / 4.217 | 3.48x |
| [100k fp32 memory k=10](results/isolated-va-f32-memory.json) | rowid 1% | 15.620 / 15.700 | 5.480 / 5.589 | 2.85x |
| [100k fp32 memory k=10](results/isolated-va-f32-memory.json) | metadata 10% | 14.533 / 14.637 | 4.579 / 4.765 | 3.17x |
| [100k fp32 memory k=10](results/isolated-va-f32-memory.json) | rowid 10% | 17.252 / 17.399 | 7.404 / 7.658 | 2.33x |
| [100k fp32 memory k=10](results/isolated-va-f32-memory.json) | metadata 100% | 17.266 / 17.423 | 7.273 / 7.882 | 2.37x |
| [100k fp16 memory k=10](results/isolated-va-f16-memory.json) | metadata 1% | 7.372 / 7.435 | 4.077 / 4.164 | 1.81x |
| [100k fp16 memory k=10](results/isolated-va-f16-memory.json) | rowid 1% | 8.775 / 8.866 | 5.443 / 5.530 | 1.61x |
| [100k fp16 memory k=10](results/isolated-va-f16-memory.json) | metadata 10% | 7.708 / 7.793 | 4.463 / 4.633 | 1.73x |
| [100k fp16 memory k=10](results/isolated-va-f16-memory.json) | rowid 10% | 10.324 / 10.416 | 7.292 / 7.476 | 1.42x |
| [100k fp16 memory k=10](results/isolated-va-f16-memory.json) | metadata 100% | 10.294 / 10.409 | 7.091 / 7.597 | 1.45x |
| [1M fp32 file k=10](results/isolated-va-million.json) | metadata 100% | 227.418 / 228.478 | 78.841 / 81.895 | 2.88x |
| [100k fp32 file k=1](results/isolated-va-k1.json) | metadata 100% | 27.417 / 27.590 | 8.804 / 9.039 | 3.11x |
| [100k fp32 file k=100](results/isolated-va-k100.json) | metadata 100% | 29.153 / 29.298 | 11.637 / 12.646 | 2.51x |
| [100k fp32 file k=10 final](results/final-va-f32-file.json) | metadata 1% | 24.517 / 24.725 | 6.647 / 6.774 | 3.69x |
| [100k fp32 file k=10 final](results/final-va-f32-file.json) | rowid 1% | 25.946 / 26.169 | 8.131 / 8.227 | 3.19x |
| [100k fp32 file k=10 final](results/final-va-f32-file.json) | metadata 10% | 24.816 / 25.079 | 7.121 / 7.293 | 3.48x |
| [100k fp32 file k=10 final](results/final-va-f32-file.json) | rowid 10% | 27.901 / 29.868 | 9.927 / 10.178 | 2.81x |
| [100k fp32 file k=10 final](results/final-va-f32-file.json) | metadata 100% | 27.803 / 28.099 | 9.183 / 9.602 | 3.03x |

### Storage and maintenance

| Run | Index | Insert seconds | Rows/s | DB MiB | Update 1k ms | Delete 1k ms |
|---|---|---:|---:|---:|---:|---:|
| 100k fp32 file k=10 | flat | 2.357 | 42,422 | 297.37 | 21.93 | 23.96 |
| 100k fp32 file k=10 | exact_va | 3.779 | 26,465 | 764.84 | 43.94 | 27.44 |
| 100k fp16 file k=10 | flat | 1.615 | 61,934 | 150.18 | 17.73 | 12.66 |
| 100k fp16 file k=10 | exact_va | 3.314 | 30,175 | 421.85 | 36.49 | 21.52 |
| 100k fp32 memory k=10 | flat | 0.621 | 161,092 | 297.37 | 9.67 | 10.65 |
| 100k fp32 memory k=10 | exact_va | 1.353 | 73,910 | 764.84 | 18.05 | 13.95 |
| 100k fp16 memory k=10 | flat | 0.454 | 220,250 | 150.18 | 6.49 | 7.72 |
| 100k fp16 memory k=10 | exact_va | 1.220 | 81,991 | 421.85 | 14.83 | 10.01 |
| 1M fp32 file k=10 | flat | 27.519 | 36,339 | 2965.25 | 27.97 | 29.65 |
| 1M fp32 file k=10 | exact_va | 47.528 | 21,040 | 7637.74 | 56.82 | 32.42 |
| 100k fp32 file k=10 final | flat | 2.318 | 43,134 | 297.37 | 21.75 | 23.66 |
| 100k fp32 file k=10 final | exact_va | 3.711 | 26,947 | 764.84 | 43.93 | 27.10 |

DB size is measured after checkpoint, before mutations. Memory databases report
their SQLite page allocation. For the initial 100k file runs, extra ingest cost
is amortized after approximately 75 unfiltered fp32 queries or 228 fp16 queries,
using only measured insert time and query savings. This excludes storage cost,
later writes, cache effects and workload changes.

## Flat-path ablations

These switches remain off by default. In-memory gains do not establish a win
for file-backed databases. The table includes each isolated change and their
combination, on 100k fp32 vectors, k=10.

| Storage / method | Metadata 1% | Metadata 10% | Rowid 1% | Rowid 10% | Unfiltered |
|---|---:|---:|---:|---:|---:|
| [file / control](results/isolated-ablation-file.json) | 25.794 | 26.191 | 28.068 | 29.377 | 29.034 |
| [file / reopen](results/isolated-ablation-file.json) | 26.035 | 25.801 | 26.887 | 29.415 | 28.839 |
| [file / gather](results/isolated-ablation-file.json) | 38.200 | 41.773 | 41.758 | 51.663 | 28.464 |
| [file / route](results/isolated-ablation-file.json) | 25.040 | 25.335 | 27.295 | 28.307 | 28.633 |
| [file / heap](results/isolated-ablation-file.json) | 24.932 | 25.536 | 26.355 | 28.196 | 28.487 |
| [file / combined](results/isolated-ablation-file.json) | 37.935 | 41.307 | 43.019 | 52.091 | 28.489 |
| [memory / control](results/isolated-ablation-memory.json) | 14.853 | 14.839 | 15.941 | 17.856 | 17.615 |
| [memory / reopen](results/isolated-ablation-memory.json) | 14.523 | 14.829 | 15.887 | 17.615 | 17.570 |
| [memory / gather](results/isolated-ablation-memory.json) | 2.658 | 5.980 | 4.147 | 11.445 | 17.697 |
| [memory / route](results/isolated-ablation-memory.json) | 14.527 | 14.875 | 16.589 | 17.675 | 17.673 |
| [memory / heap](results/isolated-ablation-memory.json) | 14.275 | 14.605 | 15.707 | 17.336 | 17.810 |
| [memory / combined](results/isolated-ablation-memory.json) | 2.346 | 5.753 | 4.416 | 11.317 | 17.776 |

`reopen` reuses BLOB handles; `gather` reads coalesced candidate ranges (and
closes each handle); `route` restricts chunk traversal using rowid B-tree lookups;
`heap` fuses global selection; `combined` enables all four. Scattered IN lists
touch most chunks, limiting the routing opportunity.
