# Faster portable FP16 distance decoding

PR base: `codex/fp16-storage`. The portable distance path previously repeated
full IEEE half decoding for every candidate component, including exceptional
values that search validation has already excluded. This change uses a compact
finite-only decoder for scalar distance kernels and hardware-kernel tails.

Normal values use exponent-bias/bit shifts; subnormals use bit construction and
exact subtraction. The result sign is restored explicitly, including under
round-toward-negative, which can make zero subtraction negative. Public general
conversion still handles all IEEE encodings. Query expansion, insertion, storage,
SQL interfaces, CPU dispatch, and build flags are unchanged.

There is no lookup table, expanded-vector cache, candidate batching, new ISA
backend, or new dependency. This adopts the laptop experiment's `magic_branch_b1`
approach as ordinary portable C. It is a small follow-up optimization rather than
a build-policy change. Remaining scalar results need not beat FP32 in every case.

## Correctness and builds

- Default F16C, scalar, AVX2, and forced fallback suites: **287 passed / 66 skipped**.
- IVF-enabled suite: **349 passed / 4 skipped**.
- Default/IVF C suites, static linking smoke test, and CPU/OS dispatch checks pass.
- All **63,488 finite encodings** compare bit-for-bit to the general converter
  with unaligned input under all **16 x86 rounding/FTZ/DAZ combinations**.
- Public tests retain coverage of all 65,536 general encodings, rounding,
  validation, updates, rollback/reopen, filters, ties, tails and zero-result cases.
- Extracted two-file amalgamation builds and passes **46 FP16 tests**.
- Assembly audit keeps VEX instructions confined to the three guarded F16C kernels.

Commands, from an x64 MSVC developer shell:

```powershell
./scripts/build-windows.ps1 -Simd scalar -OutputDirectory dist/portable-scalar
./scripts/build-windows.ps1 -OutputDirectory dist/portable-f16c
./scripts/test-fp16-windows.ps1
python -m pytest -q (Get-ChildItem tests/test-*.py).FullName --basetemp tmp/pytest-portable
python benchmarks/exact/paired.py --baseline-extension dist/storage-scalar/vec0.dll --candidate-extension dist/portable-scalar/vec0.dll --build-flags='/O2 /fp:precise; scalar both' --output dist/bench/portable-complete.json
```

For full-suite variant testing, place the selected DLL at `dist/vec0.dll`; preserve
parent and candidate DLLs in the named directories before benchmarking. Local
extended validation commands and logs are in `tmp/quickmatrix.cmd` and `tmp/`.
GCC/Clang, ARM, old physical CPUs, and hosted CI were not run locally.

## Measurements

Native MSVC 19.50, Windows x64, `/O2 /fp:precise`. Same-machine alternating workers,
2,000 vectors, four dimension sizes, L2/cosine, memory/warm-file storage, k=10,
three repetitions, five warmups, twenty timed queries. Input conversion is outside
insertion timing. Values below are medians of run medians in milliseconds.

Portable FP16 KNN improves **1.43x** geometrically across the 16 cases;
range **1.06-2.10x**.

| Dimensions / metric / storage | Parent FP16 | Candidate FP16 | Speedup |
|---|---:|---:|---:|
| 384 / cosine / file | 1.280 | 0.782 | 1.64x |
| 384 / cosine / memory | 0.894 | 0.426 | 2.10x |
| 384 / l2 / file | 0.930 | 0.789 | 1.18x |
| 384 / l2 / memory | 0.544 | 0.421 | 1.29x |
| 768 / cosine / file | 2.971 | 2.047 | 1.45x |
| 768 / cosine / memory | 2.001 | 0.996 | 2.01x |
| 768 / l2 / file | 2.273 | 1.997 | 1.14x |
| 768 / l2 / memory | 1.249 | 1.024 | 1.22x |
| 769 / cosine / file | 3.040 | 2.066 | 1.47x |
| 769 / cosine / memory | 1.955 | 1.014 | 1.93x |
| 769 / l2 / file | 2.240 | 2.057 | 1.09x |
| 769 / l2 / memory | 1.233 | 1.003 | 1.23x |
| 1536 / cosine / file | 5.822 | 3.977 | 1.46x |
| 1536 / cosine / memory | 4.054 | 1.978 | 2.05x |
| 1536 / l2 / file | 4.650 | 4.380 | 1.06x |
| 1536 / l2 / memory | 2.416 | 2.044 | 1.18x |

Raw samples and insertion/fetch/scan/database-size results are preserved in
`dist/bench/portable-complete.json`. This optimization targets distance arithmetic;
it makes no claim of improving insertion or point fetching. Database sizes are
unchanged. All benchmark answers pass stored-value reference checks.

Initial FP32 control outliers disappeared on five alternating reruns (0.998x and
1.000x at 768/769 L2 in memory). Initial F16C parent/candidate checks ranged from
0.970x to 1.042x, with no material regression. These preliminary controls preceded
the signed-zero refinement; final-commit benchmarks and targeted checks are
recorded in the handoff. No application-level recall guarantee is implied.

Compared with the storage branch, scalar `.text` grows by **1,056 bytes** and
`.rdata` by **16 bytes**; its DLL grows **1,024 bytes**. The default F16C build
adds **2,208 .text bytes**, **208 .rdata bytes**, and **3,072 file bytes**. Across
both FP16 branches the default DLL grows 12,800 bytes over the Windows-build
parent. These are compiler-specific measurements, not universal budgets.

Tested source hashes:
- `sqlite-vec-fp16.c`: `f5603051c3e619565119a0600f77889a6b5912a757529177fc58c6cccf4f2525`
- `tests/test-fp16-kernels.c`: `12c1fca52d30780026284a6681f81fd51d55e3d9fdad535a86c623bbf7696947`

Final-commit verification is recorded in the implementation handoff.
