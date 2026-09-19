# Exact-search benchmark

The follow-up [structural experiments](STRUCTURAL.md) test exact pruning indexes,
alternative memory layouts and SQLite access paths on real embeddings. See their
[results and tradeoffs](STRUCTURAL-REPORT.md). That runner also requires PyArrow,
which is declared in this project's dependencies.

Run from the repository root. The tool needs only NumPy and Python's SQLite
bindings; `uv` installs the declared dependencies. Missing extensions, packages,
or requested capabilities fail explicitly. No ANN indexes or GPU are used.

```bash
make loadable
uv run --project benchmarks/exact benchmarks/exact/bench.py run \
  --extension dist/vec0.so --label candidate --dtypes f32 f16 \
  --build-flags='-O3 -mavx -mavx2 -DSQLITE_VEC_ENABLE_AVX' \
  --output benchmarks/exact/results/candidate.json

uv run --project benchmarks/exact benchmarks/exact/bench.py compare \
  benchmarks/exact/results/baseline.json benchmarks/exact/results/candidate.json \
  --candidate-dtype f32

uv run --project benchmarks/exact benchmarks/exact/bench.py compare \
  benchmarks/exact/results/candidate.json benchmarks/exact/results/candidate.json \
  --baseline-dtype f32 --candidate-dtype f16 --cross-dtype
```

Use the actual compiler flags from your build. `--source-dir` can point at a
frozen copy of the build's sources; otherwise source hashes describe the current
checkout. Each report records source/binary hashes, commit, CPU, compiler,
Python/NumPy/SQLite versions, effective SQLite settings, and `vec_debug()`.
Preserve baseline binaries before changing the extension. Force a rebuild or use
a fresh `prefix` when changing compiler flags: make does not track flag changes.

For a quick check, add `--rows 1000 --dims 33 --queries 5 --repetitions 1
--metrics l2 --storage memory`. Individual dimensions, row counts, metrics,
storage modes, chunk size, mmap size, k, normalization, and synthetic metadata
filters are configurable; see `--help`. Comparisons reject mismatched workloads
or repetition counts. Dtype comparisons must be requested explicitly.

## Method

- Default: seed 20260916, independent Gaussian base/query vectors, 10k/100k rows,
  384/768/1536 dimensions, L2/cosine, k=10, memory/file, three repetitions.
- Each case runs in a fresh worker. FP32/FP16 order alternates by repetition when
  both types are requested. Workers run sequentially; NumPy uses one thread.
- SQLite uses 4 KiB pages, 1,024-row chunks, a 64 MiB cache, and mmap disabled.
  File databases use WAL and `synchronous=FULL`. Insertion timing includes commits
  every 1,000 rows; checkpoint time is separate. Results include actual pragmas.
- Generate and serialize vectors before timing. Report conversion/normalization
  time separately. No embedding model, network download, or GPU is involved.
- Time 100 KNN queries after five warmups, separately with and without embedding
  projection; 1,000 random point fetches; and a streamed full embedding scan.
  Fetch assertions check returned bytes. KNN consumes all result rows.
- Reads are warm at the OS-cache level; datasets larger than the SQLite cache
  still exercise SQLite page-cache misses. These are not cold-disk benchmarks.
- Reference distances use float64, computed in bounded chunks from the actual
  stored values. Check three queries per case, allowing fp32 rounding near ties.
  Separately report top-k overlap with the original float32 values. Synthetic
  accuracy results do not substitute for evaluation on application embeddings.
- Raw timings, median/p95 latency, insertion throughput, scan throughput, database
  size, reference error, and peak process RSS are retained. RSS includes the
  Python harness, arrays, and reference calculation, not just the extension.
- Speedup means baseline time / candidate time. Ratios use the median of the
  three per-run medians. Larger is faster; database-size ratios use the same
  direction. Shared-machine frequency/cache variation still affects timings.

Reports are saved after each completed case. A failed run raises an error and
leaves its partial results; new runs refuse to overwrite existing output.
Local binaries, frozen build sources, temporary databases, and logs are ignored;
JSON measurements and the report are retained in the repository.

## Detached runs on Linux

Long runs should survive terminal and agent-session restarts:

```bash
setsid nohup uv run --project benchmarks/exact benchmarks/exact/bench.py run \
  --extension dist/vec0.so --label candidate --dtypes f32 f16 \
  --build-flags='-O3 -mavx -mavx2 -DSQLITE_VEC_ENABLE_AVX' \
  --output benchmarks/exact/results/candidate.json \
  > benchmarks/exact/results/candidate.log 2>&1 < /dev/null &
bench_pid=$!
ps -o pid,pgid,sid,args -p "$bench_pid"
```

Poll the log; stop with `kill -- -<PGID>`. A sandbox that destroys its process
namespace on exit requires launching the detached job in a persistent host
session. A read-only uv cache can be redirected with `UV_CACHE_DIR=/tmp/vec-uv`.

## Optimization experiments

`experiments.py` runs the recorded one-change-at-a-time experiments, reversing
build order on alternating repetitions. It expects preserved builds named
`baseline`, `fp16-initial`, `simd`, `heap`, `filter`, `combined`, and `fma` under
`results/builds/`. The initial measurements used the experimental compile flags
recorded in [the results report](REPORT.md); those frozen sources are preserved
locally alongside the binaries.

For new ablations with the current source, disable or enable these independent
compile-time switches (all default to 1):

| Switch | Behavior |
|---|---|
| `SQLITE_VEC_EXACT_SIMD` | AVX cosine and L2 tail kernels; existing aligned L2 remains |
| `SQLITE_VEC_EXACT_HEAP` | Heap top-k selection for k > 1, preserving ties |
| `SQLITE_VEC_FILTER_FIRST` | Apply candidate filters before reading vector chunks |

These switches do not disable fp16 support. A portable scalar build uses
`make loadable OMIT_SIMD=1 prefix=dist/scalar`; `vec_debug()` identifies the
active half conversion kernel. The FMA experiment additionally used `-mfma` and
requires compatible hardware; it is not a default build requirement.
