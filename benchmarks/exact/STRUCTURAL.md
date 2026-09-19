# Structural exact-search experiments

These experiments compare with the already optimized fp16/fp32 implementation
at commit `c399333`. They do not use approximate candidate budgets or ANN search.
Results and final decisions are recorded in [STRUCTURAL-REPORT.md](STRUCTURAL-REPORT.md).

## Reproduce

Use `uv sync --project benchmarks/exact`. NumPy and PyArrow are required imports;
missing data, dependencies, binaries and requested diagnostics fail explicitly.
The native research runner currently targets Linux AVX2/F16C machines. It is not
a portable application binary or a replacement for the extension's dispatch.

The real-data source is the existing Cohere 1M benchmark mirror:
`https://assets.zilliz.com/benchmark/cohere_medium_1m/`. Download `train.parquet`
and `test.parquet` into `results/work/cohere/`, then run:

```sh
uv run --project benchmarks/exact benchmarks/exact/structural.py prepare
uv run --project benchmarks/exact benchmarks/exact/build_structural.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --project benchmarks/exact \
  benchmarks/exact/structural.py native --modes 0,1,2,3,4,6,7 \
  --output benchmarks/exact/results/new-native.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --project benchmarks/exact \
  benchmarks/exact/structural.py sql \
  --extensions benchmarks/exact/results/builds/structural-control/vec0.so \
               benchmarks/exact/results/builds/structural-gather/vec0.so \
  --output benchmarks/exact/results/new-sql.json
```

Preparation streams bounded Arrow batches into memory-mapped arrays; manifests
record source URLs, source/array hashes, dimensions and row counts. The default
screen uses the first 100,000 train rows and 100 held-out test queries after five
warmups. This is an explicit deterministic subset, not a random sample of the
million rows. Reference neighbors are recomputed for that subset.

For cosine use `--metric cosine --modes 0,4,5,6`. Other options include
`--dtype float16`, `--rows 1000000`, `--storage memory`, `--k 100`, and
`--dataset gaussian|clustered|anisotropic`. The SQL runner measures scattered
metadata and rowid restrictions; the original `bench.py` covers clustered
metadata. SQL results compare identical IDs and distance values against the
baseline on every timed query. Native runs also independently check every query
against float64 distances over the actual stored values, permitting fp32
rounding near ties. Native modes use row-number tie breaking; SQL keeps vec0's
existing chunk/offset tie breaking.

Five repetitions alternate variant order, with one worker at a time. Run long
jobs with `setsid nohup`, save their logs under `results/`, verify PID/PGID/SID,
and use `kill -- -PGID` to stop them. Do not run competing benchmarks together.
File runs use 4 KiB pages, 64 MiB SQLite caches, WAL/FULL, mmap disabled and warm
OS caches. They do not measure cold-disk latency.

## Native methods

| Mode | Method |
|---|---|
| 0 | Production distance kernels, followed by a separate global top-k pass |
| 1 | Conservative partial L2 bounds, followed by the original scorer for survivors |
| 2 | Eight-bit coordinate intervals; query-specific lower-bound lookup tables and AVX2 gathers |
| 3 | 32 sampled k-means cells, enclosing radii, lower-bound visit order, no probe limit |
| 4 | Eight-vector transposed tiles; **kernel-only**, not an exact end-to-end query |
| 5 | Cached cosine norms with the original scorer's reduction order |
| 6 | Production kernels with fused global heap selection |
| 7 | Signed int8 codes, integer dot products and reconstruction-error radii |

Mode 7 bounds the distance between reconstructed vectors, subtracts both
reconstruction-error radii using the triangle inequality, and adds a conservative
floating-point envelope before rejecting anything. Full scores still use the
production kernels. The prototypes support dimensions through 8192. They are
research implementations: bounds and mutation handling must be reviewed and
validated before any persistent index is promoted.

Mode 4 preserves original data separately and widens fp16 tiles to fp32. Its
reported top-k overlap and score error are observations, not exactness guarantees.
A favorable mode-4 kernel time alone cannot qualify it for production integration.
Native timings exclude SQLite I/O; do not compare their speedups directly with
SQL timings. `bytes` counts full-vector scorer input bytes, not measured memory
traffic; it excludes index reads, page amplification and cache effects.

Build time includes clustering, quantization, transposition or norm computation.
Query preparation, compact-code lookup tables and heap selection are included in
query timings. Stage counters are collected separately with instrumentation;
clock reads inside those measurements add overhead and must not be used as
latency results.

## Extension switches

`build_structural.py` freezes sources, compiler command and binary hashes for
independent control, reopen, gather, route, heap and combined builds. Existing
build manifests are not overwritten.

- `SQLITE_VEC_SPARSE_READS`: coalesced vector ranges, using a page-based read-cost estimate.
- `SQLITE_VEC_BLOB_REOPEN`: reuse a read-only vector BLOB handle across chunks.
- `SQLITE_VEC_ROWID_ROUTING`: use existing rowid lookups to restrict the chunk query for small IN lists.
- `SQLITE_VEC_GLOBAL_HEAP`: fused selection for finite fp16/fp32 L2; other metrics retain their original path.

`--instrument` on the build helper enables `SQLITE_VEC_BENCHMARK`. Pass
`--instrument` to the SQL runner to require and collect `vec_bench_stats()`.
That diagnostic describes the most recent flat scan on the current thread,
not connection-wide metrics. It is absent from normal builds. Instrumented
latencies are not used for promotion decisions.

## Correctness checks

```sh
uv run --project tests pytest -q tests/test-structural-search.py
uv run --project benchmarks/exact --with pytest pytest -q benchmarks/exact/test_structural.py
cc -O1 -g -mavx2 -mf16c -DSQLITE_VEC_ENABLE_AVX -Ivendor \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  benchmarks/exact/test_structural.c -lm -o /tmp/vec-structural-test
/tmp/vec-structural-test
```

`VEC_TEST_EXTENSION` selects an experimental extension for the SQL tests.

## Optional persistent exact index

Build a separate extension with:

```sh
make loadable prefix=dist/exact \
  CFLAGS='-lm -mavx -mavx2 -DSQLITE_VEC_ENABLE_AVX -DSQLITE_VEC_EXPERIMENTAL_EXACT_VA=1'
```

After loading that extension:

```sql
CREATE VIRTUAL TABLE documents USING vec0(
  embedding float[768] indexed by exact_va(),
  category integer
);
-- float16[768] is supported too. The index is L2-only.
SELECT rowid, distance FROM documents
WHERE embedding MATCH :query AND k = 10
ORDER BY distance;
```

Existing flat tables are unchanged. To add the index to existing data, create a
new table with the same columns and `indexed by exact_va()`, then copy rows with
`INSERT INTO new_table SELECT ... FROM old_table` inside a transaction. Keep the
old table until the new table has been checked. There is no implicit migration
or rebuild command. Unsupported metrics and vector types fail at table creation;
a build without the experimental feature rejects the index clause explicitly.
Fast-math builds are rejected when this index is enabled.

The index scans signed int8 reconstructions and refines every candidate whose
lower bound can improve the current top-k. It has no oversampling, probe budget,
or approximate stopping rule. Filtering happens before scoring, and ties follow
the original chunk/offset order. Returned embeddings retain their original
bytes. Each row has its own scale, so inserts and updates need no global training
or rebuild. Deletes, replacements, savepoints, rollback, reconnect and rename
maintain the summaries through ordinary SQLite shadow-table transactions.

This first storage format deliberately keeps both the original chunk vectors and
a row-addressable copy for survivor fetches. That costs disk space and writes;
it avoids random reads deep into SQLite overflow chains. It is opt-in and
experimental, not a new default or a promise of improvements on every dataset.
See measured size, insertion, mutation and latency costs in the report.

### Bounds and floating-point behavior

Let `x_hat = scale_x * codes_x`, with each signed code in [-127,127]. Store an
upper bound `radius_x >= ||x - x_hat||`, including double-precision construction
rounding. Quantize the query the same way. Compute the reconstructed squared
L2 distance from exact integer code norms and an exact integer dot product,
with a downward error allowance for the double-precision scaling operations.
The triangle inequality then gives:

```
||x - q|| >= max(0, ||x_hat - q_hat|| - radius_x - radius_q)
```

Before comparing with the production scorer's fp32 distance, reduce this lower
bound by `16*(dimensions+1)*FLT_EPSILON` proportionally and
`sqrt(dimensions*FLT_MIN)` absolutely. This intentionally loose envelope covers
subtraction, positive squared-term accumulation, square root rounding and
subnormal/flush-to-zero loss. At most 8192 dimensions are supported; intermediate
integer SIMD lanes remain far below int32 overflow. Reject only on strict `>`;
borderline candidates always use the original distance kernel. Infinite final
L2 scores cannot cause finite-bound pruning. The adversarial tests include
zeros, adjacent representable values, subnormals and extreme finite inputs.

### Process isolation and interrupted runs

**Never load different extension builds in the same process.** Their exported C
symbols can interpose, even when loaded into different SQLite connections. The
SQL orchestrator launches a fresh process for every build and repetition,
alternates order, and compares hashes of all exact result IDs and distance values
with the first isolated baseline worker. The initial same-process SQL results
were invalidated and moved to ignored `results/work/invalid/`; they are not used
in the report. Native single-library results are unaffected.

Use `--resume` with the identical SQL command to resume an interrupted run.
Workload settings, source dataset manifests and completed workers' binary hashes
must match; completed workers are retained. A complete report is left unchanged.
The output's `resumes` field records the orchestrator sources used on resumption.
