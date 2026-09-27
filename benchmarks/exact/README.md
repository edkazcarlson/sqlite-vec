# Exact-search validation benchmark

This harness measures insertion, exact KNN, KNN with embedding projection, point
fetch, and full scan separately. Inputs are serialized before insertion timing;
conversion time is reported separately. File reads are warm, WAL/synchronous FULL,
with commits every 1,000 rows, 4 KiB pages, 64 MiB cache, and mmap disabled.

```powershell
$env:VEC_BENCH_COMPILER = 'MSVC 19.50 /O2 /fp:precise'
python benchmarks/exact/bench.py run --extension dist/vec0.dll --label fp16 --build-flags='/O2 /fp:precise /DSQLITE_VEC_ENABLE_F16C' --rows 2000 --dims 384 768 769 1536 --metrics l2 cosine --storage memory file --dtypes f32 f16 --queries 20 --repetitions 3 --output dist/bench/fp16.json
python benchmarks/exact/bench.py compare dist/bench/fp16.json dist/bench/fp16.json --baseline-dtype f32 --candidate-dtype f16 --cross-dtype
```

Use `uv run --project benchmarks/exact` before the script on systems without
NumPy. Supply actual compiler metadata and flags. GCC/Clang builds may enable
`SQLITE_VEC_ENABLE_F16C` without global AVX flags. Each case runs in a fresh
worker with five warmups; dtype order alternates between repetitions. Reference
calculations check stored-value distances/top-k and original-FP32 overlap.

Preserve parent binaries and use identical workloads for comparisons. For paired
parent/candidate measurements, alternate their workers per case/repetition;
do not interpret cross-day runs as isolated kernel improvements. `worker` accepts
a JSON case and explicit extension/work directory for this purpose.

Reports retain raw samples, source/binary hashes, settings, median/p95, database
size and conversion time. Windows peak RSS is unavailable (`null`), not zero.
MacOS peak RSS is converted from bytes to KiB. Never overwrite baseline reports.
Generated binaries and full benchmark reports belong in ignored `dist/`.

For the fixed lightweight matrix above, `paired.py` alternates preserved parent
and candidate DLLs and writes individual reports accepted by `bench.py compare`:

```powershell
python benchmarks/exact/paired.py --baseline-extension dist/parent/vec0.dll --candidate-extension dist/vec0.dll --baseline-dtypes f32 --candidate-dtypes f32 f16 --build-flags='/O2 /fp:precise; candidate F16C' --output dist/bench/paired.json
```

Omit dtype options to compare both FP32 and FP16 in successive FP16 branches.
Baseline binary hashes identify preserved artifacts; candidate source hashes
describe only the current checkout.
