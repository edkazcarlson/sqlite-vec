# Portable float16 KNN experiments

Isolated experimental kernels and SQL-extension copies. **No production source
or defaults are modified.** Every inserted or scanned dataset has at most 2,000
vectors. Conversion exhaustiveness tests enumerate encodings without insertion.

Run from the repository root with 64-bit Python 3.12+, NumPy, Windows, and native
MSVC installed. The script defaults to VS 18 Community; override `VEC_VCVARS`
with the full path to another `vcvars64.bat` when necessary.

```powershell
# Use your Python executable; this session used the bundled Python runtime.
$python = 'C:\Users\Edward\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python -m pip install --target benchmarks/fp16-portable/build/packages --no-deps matplotlib pytest packaging pluggy iniconfig colorama contourpy cycler fonttools kiwisolver pyparsing python-dateutil six pygments
& $python benchmarks/fp16-portable/run.py prepare
& $python benchmarks/fp16-portable/run.py build
& $python benchmarks/fp16-portable/run.py verify
& $python benchmarks/fp16-portable/run.py screen
& $python benchmarks/fp16-portable/run.py sql
& $python benchmarks/fp16-portable/run.py report
& $python benchmarks/fp16-portable/audit.py
```

If the SQL stage is interrupted, use `run.py sql --resume`. This checks native
binary hashes and resumes the last valid JSON checkpoint without rebuilding or
repeating completed cases. Windows checkpoint renames retry transient locks.

`all` runs every stage. `prepare` downloads the repository's pinned SQLite 3.45.3
archive, copies sources, and expands the version header inside ignored `build/`.
If Python cannot validate the archive server's certificate, download it using
Windows' trust store without disabling certificate checks:

```powershell
Invoke-WebRequest 'https://sqlite.org/2024/sqlite-amalgamation-3450300.zip' -OutFile benchmarks/fp16-portable/build/sqlite-amalgamation-3450300.zip
```

Stages overwrite their own result files: copy `results/` before rerunning if you
need to preserve an earlier run. Build logs, assembly, frozen input sources, DLLs
and generated C are in ignored `build/`. Raw JSON/CSV, correctness summaries,
build provenance, SVG/PNG figures and the report are retained in `results/`.

## Experimental design

- 32 configurations: historical baseline; five converters with 1/2/4/8
  candidates; normal-block specialization at width 4/8; SSE2 software conversion
  with 1/2/4/8 candidates; SSE2 lanes across four candidates; and an original-
  arithmetic forced-inline ablation with 1/2/4/8 candidates.
- Scalar batching maintains dimension order and one sum per candidate per
  reduction. Within-vector SSE2 uses one vector sum, with no extra unrolled
  accumulation chains. Across-candidate SSE2 keeps dimension order in each lane.
- All specialized conversions require finite input. Existing table validation
  establishes that condition; public conversion helpers are unchanged.
- `kernels.py` emits the same functions for standalone screening and selected
  SQL builds. Benchmark integration is inserted only into a generated copy of
  the filtered chunk scan. Filtered/deleted slots never enter a batch.
- MSVC makes different inlining decisions in the large multi-variant library
  and small SQL builds. The forced-inline original-converter control exposes
  that effect; microbenchmark ratios do not predict SQL speedups. Use measured
  SQL results for recommendations.
- Float32 controls use the same arithmetic/batching strategy and half-rounded
  values. Historical float32 cosine recomputes the query norm per candidate;
  matched experimental controls hoist it out of the candidate loop (once per
  native scan, once per SQL chunk). Float16 retains its once-per-query norm.
  The SQL runs use one or two chunks, a documented outer-loop asymmetry.
- SSE2 variants are x86-specific; scalar kernel expressions are portable C.
  The build/measurement runner is intentionally Windows-specific. It does not
  change repository support requirements or add architecture dependencies to
  production builds.
- All builds use `/O2 /fp:precise` and default x64 SSE2, plus consistent runtime,
  C-language and linker flags. MSVC's documented `loop(no_vector)` pragma on
  measured loops supplies a second screening control; `/Qvec-` is not supported
  by this compiler. Explicit intrinsics remain SIMD in the control.
- The two fastest screened configurations receive the full SQL suite. The best
  one-candidate portable finite converter additionally receives the 768-dimension
  cases, so the half-specific algorithm is checked independently of explicit SIMD.
- No AVX/F16C macro or compiler target is enabled. Audits reject advanced SIMD
  instructions in the binary and inspect floating-point functions for bit-scan
  feature dependencies. Existing unrelated Hamming POPCNT is documented rather
  than patched. Saved disassembly supports manual review.

## Reading the output

Start with [the report](results/REPORT.md). `screen.json` and `conversion.json`
contain all repetitions and internal repeat counts. `ranking.json` documents the
objective finalist selection. `sql.json` contains fresh-worker latency samples,
insertion times, SQLite settings, sizes, correctness errors and original-float32
top-k overlap. `sql-complete.json` distinguishes a complete SQL suite from a
partial file left by interruption. Process peak RSS includes Python/NumPy and
reference calculations, and must not be interpreted as kernel scratch memory.
The production `vec_debug()` text is unchanged and does not describe the
experimental hook; use the build label, selected kernel ID and source hash.

Research references and attribution are in `kernels.py` and the report. No
external conversion implementation, lookup table or persistent widened-vector
cache is linked into the experiments.
