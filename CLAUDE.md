# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

Prerequisites: run `./scripts/vendor.sh` to download vendored SQLite source into `vendor/`.

```bash
make loadable       # Build loadable extension → dist/vec0.{so,dll,dylib}
make static         # Build static library → dist/libsqlite_vec0.a
make cli            # Build SQLite CLI with vec0 built-in → dist/sqlite3
make all            # Build all three above
```

The header `sqlite-vec.h` is generated from `sqlite-vec.h.tmpl` via envsubst using the `VERSION` file. It is regenerated automatically by Make targets that depend on it.

## Test Commands

```bash
make test-loadable                  # Full pytest suite (builds loadable first, uses uv)
make test-unit                      # C unit tests (assert-based, in tests/test-unit.c)
make test                           # Basic SQL smoke test via sqlite3 CLI
make fuzz-quick                     # Libfuzzer targets, 30s each
make fuzz-long                      # Libfuzzer targets, 5min each
```

Run a single Python test:
```bash
uv run --managed-python --project tests pytest -vv -s -x -k test_name tests/test-loadable.py
```

## Windows Build Commands (MSVC)

On Windows without `make`/`gcc`, use the PowerShell scripts that invoke MSVC (`cl.exe`) via `vcvars64.bat`:

```powershell
.\scripts\test-unit-msvc.ps1          # C unit tests (scalar, no SIMD)
.\scripts\test-unit-msvc.ps1 -Avx     # C unit tests with AVX2 enabled
.\scripts\bench-insert-query-msvc.ps1          # Benchmark (scalar)
.\scripts\bench-insert-query-msvc.ps1 -Avx     # Benchmark with AVX2
```

Requires Visual Studio 2022 Community (specifically `vcvars64.bat`). The scripts auto-generate `sqlite-vec.h` from the template if needed.

## Formatting and Linting

```bash
make format         # clang-format on C files, black on Python tests
make lint           # Check C formatting (diff-based, fails on mismatch)
```

## Architecture

**Single-file C extension**: The entire implementation lives in `sqlite-vec.c` (~10K lines). It registers a `vec0` virtual table module and 40+ scalar SQL functions (distance, quantization, vector constructors, etc.) via `sqlite3_vec_init`.

**Vector types**: `float[N]` (32-bit float), `int8[N]` (signed 8-bit int), `bit[N]` (1-bit packed into bytes).

**Distance metrics**: L2 (Euclidean), L1 (Manhattan), cosine, hamming. Optional SIMD acceleration via compile flags: `-DSQLITE_VEC_ENABLE_AVX` (x86_64) and `-DSQLITE_VEC_ENABLE_NEON` (arm64).

**Storage model**: Vectors are stored in chunks (configurable `chunk_size`, default ~1000) across shadow tables:
- `_rowids` — maps user rowid to chunk_id + chunk_offset
- `_chunks` — chunk metadata (size, validity bitmap, rowids)
- `_vector_chunksNN` — one per vector column, stores packed vector data
- `_metadatachunksNN` / `_metadatatextNN` — one per metadata column
- `_auxiliary` — auxiliary column values

**Query plans** (encoded in idxStr, see `ARCHITECTURE.md`):
- Fullscan (`'1'`), point lookup by rowid (`'2'`), KNN search (`'3'`)

**Key compile-time defines**:
- `SQLITE_CORE` + `SQLITE_VEC_STATIC` — for static linking (no extension loading)
- `SQLITE_VEC_ENABLE_AVX` / `SQLITE_VEC_ENABLE_NEON` — SIMD paths
- `SQLITE_VEC_TEST` — enables test hooks (used by `test-unit.c`)

**Dependencies**: Pure C, no external dependencies. Requires SQLite 3.38+ (for `sqlite3_vtab_in`). Vendored SQLite in `vendor/`.

## Testing Architecture

- `tests/conftest.py` — pytest fixture that loads `dist/vec0` extension into an in-memory SQLite database
- `tests/test-loadable.py` — main comprehensive test suite
- `tests/test-*.py` — feature-specific test modules (insert-delete, metadata, auxiliary, partition-keys, knn-distance-constraints, general)
- `tests/test-unit.c` — C unit tests for vector tokenization, parsing, and low-level operations

## Language Bindings

Bindings exist in `bindings/` for Python, Rust, and Go. Each has its own build system. The Python binding adds serialization helpers; the Rust binding wraps `sqlite3_vec_init` via FFI with a build script that compiles `sqlite-vec.c`.
