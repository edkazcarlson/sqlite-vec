# Store FP16 vectors with portable and hardware-assisted exact search

PR base: `codex/windows-simd-build`. FP32 embeddings consume four bytes per
component. This adds an explicit half-precision storage type using two bytes,
while widening values for FP32 distance accumulation. Existing columns keep their
types and require no migration. Users opt in with a new table and explicit conversion.

## API and behavior

Adds `float16[N]` / `f16[N]`, `vec_f16`, explicit FP16-to-FP32 widening, flat
L1/L2/cosine search, and raw half-blob retrieval. Constructors preserve IEEE half
encodings; insertion/update/distance calls reject nonfinite values, and cosine
rejects zero vectors. Rounding can change rankings. ANN/rescore is unsupported.
Untyped blobs in FP16 columns are interpreted as half bytes; JSON/FP32 input needs
explicit conversion. Public conversion remains general, while search validates
finite inputs. Existing FP32 JSON and general type-prefix behavior are preserved,
except that malformed `float16...` declarations are rejected rather than mistaken
for FP32. `vec_debug()` gains a fifth line naming the FP16 backend.

The query is widened once, its norm is computed once, and its metric/backend is
selected once. GCC/Clang function targets and native MSVC isolated intrinsics
supply F16C acceleration. CPU AVX/F16C/OSXSAVE and OS XMM/YMM state are required.
Otherwise the portable decoder is used. No global mutable dispatch state or cache
is introduced. There is no SSE2/FMA/AVX-512 backend or bulk-conversion optimization.

## Build conventions

`SQLITE_VEC_ENABLE_F16C` is a compile-time inclusion option with local runtime
selection: a deliberate exception to existing compile-time-only SIMD dispatch,
needed to avoid raising the Windows DLL's CPU requirements. Windows default
builds enable it without `/arch:AVX*`; `-Simd scalar` omits it. Existing GCC/Clang
AVX builds retain F16C support; opt-in AVX2 builds still require AVX2 hardware.

Included source dependencies and Rust source generation are updated. The release
job now inlines split sources before packaging. C-test repairs reconcile existing
interleaved functions and internal layouts so the FP16 feature can be validated;
no production ANN behavior is changed. Benchmarks and short documentation are
included, but historical measurement datasets are not.

## Validation

Native MSVC 19.50, Windows x64, `/O2 /fp:precise`:

- Default F16C, scalar, AVX2, and forced fallback: **287 passed / 66 skipped**.
- IVF-enabled Python suite: **349 passed / 4 skipped**.
- Default and IVF C unit suites pass.
- CPU/OS dispatch and unaligned/tail checks pass, including forced fallback.
- General conversion tests cover all 65,536 half encodings and FP32 rounding edges.
- A two-file amalgamation archive was extracted, compiled and passed **46 FP16 tests**.
- Static library creation and an executable linked against it pass a SQL smoke test.
- Assembly audit: VEX instructions occur only in the three guarded half kernels.

Reproduction after generating headers and vendoring SQLite:

```powershell
./scripts/build-windows.ps1
./scripts/test-fp16-windows.ps1
python -m pytest -q (Get-ChildItem tests/test-*.py).FullName --basetemp tmp/pytest-fp16
./scripts/build-windows.ps1 -Simd scalar -OutputDirectory dist/scalar
./scripts/build-windows.ps1 -Simd avx2 -OutputDirectory dist/avx2
python benchmarks/exact/paired.py --baseline-extension dist/default/vec0.dll --candidate-extension dist/storage-f16c/vec0.dll --baseline-dtypes f32 --candidate-dtypes f32 f16 --build-flags='/O2 /fp:precise; candidate F16C' --output dist/bench/storage-pre.json
```

Preserve the parent default DLL in `dist/default/` and the candidate in
`dist/storage-f16c/` before comparison. Local auxiliary validation commands are
in `tmp/units.cmd`, `tmp/matrix.cmd`, and `tmp/static-ivf.cmd`; logs and instruction
inspection are retained in `tmp/`. On GCC/Clang, use `make test-unit` and
`make test-fp16-kernels` with the desired CFLAGS. GCC/Clang, ARM, older physical
CPUs, hosted CI, and the sqlite-dist publisher were not executed locally. The
local package test validates the same inlined two-file input, not publication.

## Measurements

2,000 vectors; 384/768/769/1536 dimensions; L2/cosine; memory/warm-file storage;
k=10; three repetitions, five warmups and twenty queries. Parent/candidate order
alternates and each case uses a fresh process. Database settings and raw samples
are in `dist/bench/storage-pre.json`. Input serialization is outside insertion timing.

Geometric-mean same-build FP32/FP16 ratios: **3.31x KNN**, **1.63x insertion**,
**1.66x point fetch**, **1.70x full scan**, and **1.97x database size**. FP32 uses
its existing scalar Windows kernels while FP16 uses F16C: these are not isolated
storage-format speedups. Point-fetch gains also reflect reduced blob traffic.

At 768 dimensions (milliseconds):

| Metric / storage | FP32 KNN | FP16 KNN | Insert speedup |
|---|---:|---:|---:|
| l2 / memory | 0.937 | 0.293 | 1.42x |
| l2 / file | 3.013 | 1.340 | 2.21x |
| cosine / memory | 1.065 | 0.299 | 1.38x |
| cosine / file | 3.200 | 1.359 | 2.29x |

The 768-dimensional database is 3,248,128 bytes for FP16 versus 6,402,048 for FP32.
All benchmark answers pass stored-value distance/top-k checks. Synthetic ranking
checks are not an application-level recall guarantee.

Two apparent FP32 file-backed regressions were investigated with five alternating
reruns: parent/candidate KNN ratios became **1.006x** (769 L2) and **0.999x**
(1536 cosine), rather than reproducing the initial outliers. Evidence:
`dist/bench/storage-investigation.json`. No broad FP32 speedup is claimed.

Relative to the parent default DLL, FP16+F16C adds **8,848 .text bytes**, **1,040
.rdata bytes**, and **9,728 file bytes**. FP16 scalar-only adds 6,832 .text bytes;
the F16C backend itself adds about 2 KiB of executable code. No LUT or persistent
expanded-vector cache is allocated. Sizes are compiler-specific.

Tested source hashes:
- `sqlite-vec.c`: `71a179324f41763098d4f15e17e04a82c5ba58c51ac4643a044ae648df599162`
- `sqlite-vec-fp16.c`: `8592f5961d09eaf3f1b3071bd14e01065d6f25def89f74d28a2734b26d4a43d6`

Final-commit verification is recorded in the implementation handoff.
