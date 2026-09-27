# Enable opt-in Windows AVX2 builds with MSVC

PR base: `main`. The Windows workflow previously bypassed the repository's SIMD
options, and enabling its AVX option with native MSVC encountered GCC alignment
syntax. This change supplies compatible alignment declarations and a shared
PowerShell build entrypoint, with default and AVX2 CI configurations.

The published DLL retains its existing compiler target. `-Simd avx2` opts into
both AVX floating-point and AVX2 Hamming code, requiring compatible hardware.
There is no new runtime dispatcher, dependency, SQL API, FP16 code, or package.
This follows the existing compile-time SIMD convention. The workflow artifact
names distinguish the test configurations; release artifact names are unchanged.

## Validation

Native MSVC 19.50, x64 Windows, `/O2 /fp:precise`. Both variants: **239 passed,
66 skipped**, with 204 snapshots passing. Skips are existing optional tests.
Generated instructions include AVX arithmetic and AVX2 Hamming operations.
Existing compiler warnings were not addressed in this focused change.

From an x64 MSVC developer shell, after generating the header and vendoring SQLite:

```powershell
./scripts/build-windows.ps1
python -m pytest -q (Get-ChildItem tests/test-*.py).FullName --basetemp tmp/pytest-default
./scripts/build-windows.ps1 -Simd avx2
python -m pytest -q (Get-ChildItem tests/test-*.py).FullName --basetemp tmp/pytest-avx2
```

The local external harness is `tmp/windows-bench.py`; reproduce with
`python tmp/windows-bench.py pre`. It compares preserved main/default/AVX2 DLLs
in fresh subprocesses, rotating variant order: 2,000 vectors, memory-backed
exact top-10 search, five warmups, twenty queries, three repetitions. Values are
medians of run medians in milliseconds. Raw data: `tmp/windows-bench-pre.json`.
The harness is intentionally not part of this build-focused PR.

| Metric / dimensions | Main | Default | AVX2 | Default / AVX2 |
|---|---:|---:|---:|---:|
| l2 / 384 | 0.4829 | 0.4950 | 0.2698 | 1.83x |
| l2 / 768 | 0.9754 | 0.9858 | 0.4851 | 2.03x |
| l2 / 769 | 0.9633 | 0.9761 | 1.0069 | 0.97x |
| hamming / 384 | 0.0205 | 0.0205 | 0.0428 | 0.48x |
| hamming / 768 | 0.0243 | 0.0245 | 0.0227 | 1.08x |
| hamming / 776 | 0.0604 | 0.0602 | 0.0236 | 2.55x |

The default's executable section is identical in size to main (208,144 bytes).
AVX2 adds 1,024 executable bytes and 224 read-only bytes; DLL size grows from
297,472 to 299,008 bytes. This is a compiler-specific measurement.

AVX2 is beneficial for aligned L2 and some Hamming workloads, not universally.
The 384-bit Hamming regression is repeatable across the three runs: the existing
SIMD implementation handles a 32-byte block plus a 16-byte bytewise tail, whereas
the baseline uses six 64-bit popcounts. This PR exposes the existing kernel
without retuning it. Dimension 769 uses the existing scalar L2 path. These
results justify an opt-in option rather than changing the release default.

No Linux/ARM performance claim or older-CPU compatibility test is made.
Final-commit verification is recorded in the implementation handoff.

Tested sqlite-vec.c SHA256: `2853349074585497f60ad96778a934045b5e7a2b4b595ee0ed8a212f8b2a7ea4`.
