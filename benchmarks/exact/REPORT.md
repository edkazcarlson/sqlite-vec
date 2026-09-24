# Exact-search benchmark results

Collected September 16–17, 2026 on this machine: AMD Ryzen 9 9900X, 30 GiB RAM,
Linux x86-64. CPU only; no ANN index, GPU, or external embedding model. Raw
reports contain the full hardware, compiler, SQLite/Python/NumPy, build, and
workload metadata. See [method and commands](README.md).

## Results to use

Native float16 storage is implemented, with float32 distance accumulation.
On this CPU the half-distance kernel uses runtime-checked F16C conversion.
For **100k vectors × 768 dimensions**, k=10, the final build measured:

| Storage / metric | FP32 insert rows/s | FP16 insert rows/s | FP32 KNN ms | FP16 KNN ms | Search speedup |
|---|---:|---:|---:|---:|---:|
| Memory / L2 | 164,161 | 221,454 | 17.283 | 10.155 | 1.70× |
| Memory / cosine | 164,046 | 221,265 | 17.506 | 10.286 | 1.70× |
| File / L2 | 42,478 | 61,743 | 26.849 | 15.602 | 1.72× |
| File / cosine | 43,127 | 62,799 | 26.533 | 15.246 | 1.74× |

Database size was **297.37 MiB for fp32 versus 150.18 MiB for fp16**. This includes
SQLite, chunk, rowid, and metadata overhead, so it is slightly more than half.
For the same L2 workload, median point-fetch latency was 5.99 → 4.06 µs in memory
and 116.29 → 28.02 µs from the warm file-backed database. File-backed gains depend
on SQLite cache fit and page reads, not just the vector byte count.

Across the final representative suite, fp16 KNN was **1.47–2.71× faster** than fp32
in the same build. Insertion was typically **1.20–1.60× faster**. The recorded
2.68× insertion result for 100k × 384 file-backed cosine is anomalous: that fp32
case was unusually slow, so do not generalize it. Durable insertion timings are
particularly sensitive to other filesystem activity.

These same-build comparisons isolate the dtype paths better than comparing new
fp16 directly against the original extension, which also lacks cosine SIMD.

## Baseline and reproducibility

The unchanged baseline is commit `04d28bd21773981e2d266bbf6aa4efbd011eb4f6`.
All extensions were built with `-O3 -fPIC -shared`, AVX/AVX2 enabled, and the
same compiler. Default options retain the existing rescore/DiskANN code, but
every benchmark table uses **flat exact search**.

| Artifact | Contents |
|---|---|
| [baseline.json](results/baseline.json) | 72 original fp32 runs |
| [fp16-initial.json](results/fp16-initial.json) | 72 fp16 runs before the other search optimizations |
| [experiments.json](results/experiments.json) | 174 alternating ablation and tuning runs |
| [final.json](results/final.json) | 144 final runs, alternating fp32/fp16 order |
| [verification.json](results/verification.json) | 30 alternating checks of apparent L2 regressions |

Complete ratio tables:

- [Original fp32 versus initial fp16](results/baseline-vs-fp16-initial.md)
- [Original versus final fp32](results/baseline-vs-final-f32.md)
- [Final fp32 versus final fp16](results/final-fp32-vs-fp16.md)

Ratios are baseline time / candidate time, using the median of three run
medians. Raw samples support alternative analyses and include p95 summaries in
the later reports. Baseline/initial reports predate the summary fields; their
raw samples contain the same measurements. Build binaries and source snapshots
remain locally in the ignored `results/builds/` directory. The final benchmark
snapshot precedes two subsequent correctness-only edits: freeing metadata names
on table teardown and rendering nonfinite half values as JSON null. Neither
edit changes the timed paths. Current source and the resulting `dist/vec0.so`
include both fixes.

The initial ablations used these flags in addition to the common build flags:

| Build | Additional flags |
|---|---|
| `simd` | `-DSQLITE_VEC_EXPERIMENTAL_EXACT_SIMD` |
| `heap` | `-DSQLITE_VEC_EXPERIMENTAL_MIN_IDX` |
| `filter` | `-DSQLITE_VEC_EXPERIMENTAL_FILTER_FIRST` |
| `combined` | All three flags above |
| `fma` | All three flags above, plus `-mfma` |

The measured final build promoted these to the default `SQLITE_VEC_EXACT_SIMD`,
`SQLITE_VEC_EXACT_HEAP`, and `SQLITE_VEC_FILTER_FIRST` switches. It retained
the original aligned L2 kernel because replacing it did not produce a reliable
benefit. The current source removes those unrelated float32 search changes;
use the preserved builds and recorded binary hashes to reproduce or interpret
these historical measurements.

### Apparent regression checked

Some cross-day comparisons suggested a 15–20% file-backed L2 regression.
Alternating fresh baseline/heap/filter/final runs did not reproduce it:

| Workload, fp32 L2, 768 dimensions | Baseline median ms | Final median ms | Speedup |
|---|---:|---:|---:|
| 10k rows, file | 2.8767 | 2.8806 | 0.999× |
| 100k rows, file | 26.4581 | 26.0786 | 1.015× |
| 10k rows, memory | 1.4807 | 1.4735 | 1.005× |

The unchanged baseline also slowed relative to the first collection. The
paired evidence supports timing/environment variation rather than a code
regression. Do not interpret the unpaired historical table as proving an L2
regression or speedup. Hardware-counter profiling was unavailable because
`perf_event_paranoid=4`; no cycle-count or bandwidth-saturation claim is made.

## Optimizations retained and experiments rejected

| Experiment | Paired observation | Decision |
|---|---|---|
| AVX cosine, independent accumulators, one query norm per search | 2.56–3.57× faster at 10k × 384/768/769 | Enabled on AVX builds |
| SIMD tails for L2 dimensions not divisible by 16 | 2.48× faster at 10k × 769 | Enabled |
| Replace the existing aligned L2 accumulation loop | About 0.98–1.01× at 384/768 | Keep the original aligned kernel |
| Heap top-k, preserving existing tie order | 1.16× at k=100; approximately neutral at k=10 | Enabled for k > 1; k=1 retains the original selection path |
| Read vector chunks after metadata/rowid filters | 60.97× at 1% clustered selectivity; 7.60× at 10% | Enabled; skip only chunks with no valid candidates |
| Same filter change, scattered matches | About 1.00–1.02× in the isolated filter build | No broad claim: nearly every chunk still needs reading |
| Global `-mfma` | Mixed, up to about 8% over combined on one cosine case | Not enabled globally; requires separate CPU compatibility and wider validation |
| Pre-normalize then use L2 | Approximately neutral after cosine SIMD | Application choice, not an automatic rewrite |

The normalization experiment at 100k × 768 measured fp32 cosine at 17.624 ms
versus normalized L2 at 17.803 ms; fp16 measured 10.375 versus 10.266 ms.
Total preparation/serialization increased from 85.7 to 137.0 ms for fp32 and
129.1 to 185.9 ms for fp16. Unit-vector L2 and cosine rankings are mathematically
equivalent, but rounding stored halves can alter norms and rankings.
[Metric relationship](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)

### Useful application tuning

For warm file-backed 100k × 768 fp32 L2, the experimental combined build measured:

| Chunk size | mmap disabled, ms | mmap limit 1 GiB, ms |
|---|---:|---:|
| 256 | 26.641 | 16.603 |
| 1024 | 27.963 | 15.556 |
| 4096 | 29.040 | 15.762 |

`PRAGMA mmap_size=1073741824` is worth testing: at the existing 1024 chunk size
it improved this workload by **1.80×**. Verify the returned pragma because SQLite
builds can limit or omit mmap. Keep it explicit rather than changing connection
defaults. The chunk-size winner changed with mmap, so increasing chunk size
alone is not a general optimization. [SQLite mmap documentation](https://www.sqlite.org/mmap.html)

## Accuracy, tests, and remaining opportunities

All checked final KNN results matched the exact top-k over the **stored values**.
The fp16 results had **99.72% mean top-10 overlap** with the original fp32 values.
This checks three reference queries per workload, repeated across storage modes
and repetitions; it is not an application-level recall guarantee. The reference
uses float64 and tolerates fp32 rounding near ties. No approximate candidate
pruning is used.

Validation includes all 65,536 half bit patterns; positive/negative rounding
boundaries and subnormals; scalar/F16C distance checks through 8192 dimensions;
invalid inputs; raw/typed insertion and update; rollback/reopen; vector fetches;
metadata/rowid/partition constraints; top-k ties and NaNs; and benchmark output
and comparison failures. The existing C harness was repaired to separate
interleaved test functions, close missing conditional blocks, and match current
internal struct layouts and quantizer semantics. Sanitizers also found and
verified the fix for the pre-existing metadata-name leak.

Final checks: **286 passed / 66 skipped** in the default Python suite (IVF is
disabled); **348 passed / 4 skipped** with IVF enabled; **45 passed** against the
scalar amalgamated extension; C unit suites passed in default and IVF builds;
AddressSanitizer, UndefinedBehaviorSanitizer, and LeakSanitizer passed. Static
library and amalgamation builds also passed. The remaining skips belong to
existing optional/platform-specific tests.

Follow-up ideas, ranked by the measured workload bottlenecks:

1. **Sparse exact gathers for selective scattered filters.** At 1% scattered
   selectivity, the current scan still reads almost every vector chunk. Read
   selected vector ranges instead, choosing a crossover against full-chunk
   reads. Coalesce adjacent rows to avoid one SQLite call per vector.
2. **Avoid fp32 input copies on insertion.** Its parser allocates and copies raw
   blobs before validation. A validated borrowed-buffer path could reduce
   allocator traffic, with explicit alignment and SQLite-value lifetime rules.
3. **Batch exact queries.** Reuse each loaded chunk for several queries or use a
   blocked matrix kernel. This can amortize memory reads and SQLite overhead,
   but needs a batch API and has a latency/throughput tradeoff.
4. **Cache norms computed from stored values.** Cosine currently recomputes each
   candidate norm. A norm sidecar could remove that work while preserving raw
   embeddings. Updates, rollback, migration, and fp16 rounding must remain
   consistent; benchmark the extra reads after SIMD before committing to it.
5. **Defer L2 square roots until after selection.** Use squared distances
   internally, transform distance constraints carefully, and return ordinary
   L2 distances. Test ties, overflow, and negative thresholds. Larger gains are
   more likely at low dimensions than in the measured large-vector scans.
6. **Runtime-selected FMA/AVX-512 and alternative blocking.** Compare independent
   accumulation chains and layouts across dimensions and CPU models. The first
   aligned-L2 experiment shows why a wider or more unrolled kernel should not
   be assumed faster. [Instruction and optimization reference](https://www.intel.com/content/www/us/en/developer/articles/technical/intel64-and-ia32-architectures-optimization.html)

These are proposals, not measured gains or implemented ANN alternatives. No
automatic normalization, cached-norm storage format, batch API, or global FMA
requirement was introduced.
