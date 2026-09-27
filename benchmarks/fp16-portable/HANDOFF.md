# Float16 KNN experiment handoff

The measured prototypes are isolated under this directory. **The stock extension
has no speedup yet:** production `sqlite-vec.c` and `sqlite-vec-fp16.c` were not
edited. [kernels.py](kernels.py) is the source of the experimental C converters,
distance loops, and candidate batching; [run.py](run.py) generates a copy of the
extension with a benchmark-only hook in its filtered chunk scan. Generated C and
DLLs are in ignored `build/`. There is no lookup table, expanded-vector cache,
storage-format change, or public SQL/API change.

## What was tried

| Change | Purpose and outcome |
|---|---|
| Finite-only scalar conversion | For normal halves, combine exponent/mantissa shifts with a bias; for subnormals, compare integer-to-float scaling with float-bit construction followed by subtraction. The branched latter form (`magic_branch_b1`) was the best portable one-candidate option. Both preserve signed zero. Existing insert/update/query validation rejects nonfinite values; the public general converter is unchanged. |
| Branchless selection and 4/8-value normal-block checks | Tested whether predictable normal inputs could amortize case handling. Neither led the Gaussian distance ranking; no embedding-range assumption was made. |
| SSE2 software widening | Convert four halves with integer/SSE2 operations and immediately use the floats for L2, cosine, or L1. This was the fastest distance family on this machine. |
| Across-candidate overlap | Interleave 1, 2, 4, or 8 candidate distances, plus a separate four-candidate SSE2 lane layout. Matching float32 versions distinguish a general batching benefit from half decoding. Two and eight candidates tied in end-to-end SQL; the lane layout was slower in the distance screen. |
| Controls | Current kernels, original converter forced inline, and compiler loop-vectorization-disabled builds. The forced-inline control matters because the large microbenchmark DLL calls the original converter, while the baseline SQL DLL inlines it. Microbenchmark speedups should not be used as SQL predictions. |

## Measured SQL result

Geometric-mean latency speedup; above 1 is faster. Float32 controls store the
**same half-rounded values and queries** and use equivalent batching/SIMD.

| Float16 variant | Versus current float16 | Versus matched float32 | Scope |
|---|---:|---:|---|
| `magic_branch_b1` | 1.33x | 1.52x | 8 primary cases, 768 dimensions |
| `sse2_b2` | 1.86x | 1.47x | 24 primary cases, 384/768/1536 dimensions |
| `sse2_b8` | 1.86x | 1.47x | Same 24 cases |

Primary cases use 1,000/2,000 rows, L2/cosine, memory/warm-file storage, and
`k=10`. Five warmups and twenty timed queries were run in each of three fresh,
sequential repetitions, with variant order rotated. There were also L1 and
10%/50% metadata-filter checks at 2,000 x 768. Every dataset stayed at or below
**2,000 rows**. The 24 SSE2 primary cases all beat current float16 with
non-overlapping ranges of per-run medians. Some memory-backed cases still favor
matched float32. Raw float32 distance kernels are faster than float16, so the
end-to-end float16 advantage is consistent with smaller stored vectors; cache
and copy costs were not separately measured.

`sse2_b2` and `sse2_b8` add respectively **12.25 KiB** and **21.98 KiB** to the
full extension's executable `.text` section versus baseline (including matched
float32 code). Their largest experimental scan frames are **280** and **552**
bytes. No 256 KiB conversion table is present. The 2,000 x 768 float16 database
was **3,248,128 bytes**, versus **6,402,048 bytes** for matched float32.

## Correctness, portability, and next step

Specialized converters passed all **63,488 finite half encodings**; general
converters passed all **65,536**. Checks included signed zero, subnormal edges,
unaligned input, and four FTZ/DAZ modes. Both default and vectorization-disabled
distance builds passed **14,976** cases. Each SQL DLL passed the existing 45
float16 tests plus ties, deletions, filters, partial batches, and top-k edge
cases. Every timed SQL result passed float64 reference/top-k checks.

Builds used native MSVC 19.50 `/O2 /fp:precise`, default x64 SSE2, without AVX
or F16C enablement. Measured float paths were checked for newer instructions;
an unrelated existing Hamming path contains POPCNT, so the whole DLL is not
certified SSE2-only. The Ryzen 7 5800H supports SSE2/AVX/AVX2/FMA/F16C; these
timings do not establish performance on older CPUs.

**Recommendation:** evaluate `magic_branch_b1` as the portable C path and
`sse2_b2` as an optional x64 path on older hardware before considering a
production change. Batch two has the same aggregate SQL gain as batch eight
with less code and stack use.

See the [full report](results/REPORT.md) for per-case median/p95 tables, losses,
figures, CPU survey sources, and caveats. [README.md](README.md) has exact
reproduction commands. Raw samples are in [SQL JSON](results/sql.json),
[distance JSON](results/screen.json), and
[conversion JSON](results/conversion.json); [audit.json](results/audit.json)
records the row cap, hashes, and completed case counts.
