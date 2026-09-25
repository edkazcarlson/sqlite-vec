# Current-source small exact-search comparison

Run September 24, 2026 on AMD Ryzen 9 9900X. Both extensions were compiled from
this checkout at commit `118c138`, with `-O3`; the AVX build used
`-mavx -mavx2 -DSQLITE_VEC_ENABLE_AVX`, and the scalar build used Make's
`OMIT_SIMD=1`. The CPU reports `avx`, `avx2`, and `f16c`; `vec_debug()` selected
`f16c` for the AVX build and `scalar` for the other. Both accumulate distances in
float32. See [raw AVX](current-small-avx.json) and
[raw scalar](current-small-scalar.json) reports for source/binary hashes,
environment, SQLite settings, and all timing samples.

The benchmark used seeded Gaussian vectors, 768 dimensions, k=10, 30 timed
queries, two repetitions, and both in-memory and file-backed databases. Each
dtype ran in a fresh worker; order alternated between repetitions. Ratios below
are median fp32 time divided by median fp16 time; above 1 means fp16 was faster.

| Rows | Metric | Storage | AVX/F16C KNN | Scalar KNN |
|---:|---|---|---:|---:|
| 1,000 | L2 | memory | 1.45× | 0.58× |
| 1,000 | L2 | file | 1.69× | 0.64× |
| 1,000 | cosine | memory | 5.67× | 0.63× |
| 1,000 | cosine | file | 4.06× | 0.73× |
| 5,000 | L2 | memory | 2.14× | 0.57× |
| 5,000 | L2 | file | 2.12× | 0.69× |
| 5,000 | cosine | memory | 6.78× | 0.69× |
| 5,000 | cosine | file | 4.57× | 0.80× |

At 5,000 rows the databases occupied 15.199 MiB for fp32 and 7.691 MiB for
fp16 (1.976× smaller). Across these cases, fp16 insertion was about 1.29–1.65×
faster; the slower scalar fp16 search is consistent with conversion costs
outweighing the smaller read footprint at this size. In the AVX build, fp32
cosine has no explicit AVX kernel, while fp16 cosine uses AVX/F16C, so its
large ratio includes
a kernel difference. At 5,000 rows in memory, for example, median KNN latency
was 2.212 ms versus 0.326 ms for cosine, and 0.662 ms versus 0.310 ms for L2.

The checked KNN results matched the exact top-k over each stored dtype. For the
three reference queries per run, fp16 top-10 overlap with the original fp32
vectors was 100% except for 1,000-row cosine, where it was 96.7%. These are
small synthetic samples and two timing repetitions, not an application recall
or stable hardware throughput estimate. Reads were warm, not cold-disk reads.
