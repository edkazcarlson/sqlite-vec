"""Reproducible SQLite exact-search benchmarks. Run with uv run --project ... ."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import sqlite3
import subprocess
import sys
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def connect(extension, path, case):
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension(str(Path(extension).resolve()), entrypoint="sqlite3_vec_init")
    db.enable_load_extension(False)
    db.execute("PRAGMA page_size=4096")
    db.execute("PRAGMA cache_size=-65536")
    db.execute(f"PRAGMA mmap_size={case['mmap']}")
    db.execute("PRAGMA synchronous=FULL")
    if path != ":memory:":
        db.execute("PRAGMA journal_mode=WAL")
    return db


def timed(call):
    start = time.perf_counter_ns()
    result = call()
    return (time.perf_counter_ns() - start) / 1e6, result


def worker(args):
    case = json.loads(args.case)
    n, d = case["rows"], case["dims"]
    rng = np.random.default_rng(case["seed"])
    base = rng.standard_normal((n, d), dtype=np.float32)
    queries = rng.standard_normal((case["queries"] + 5, d), dtype=np.float32)
    conversion_start = time.perf_counter_ns()
    if case["normalize"]:
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    dtype = np.float16 if case["dtype"] == "f16" else np.float32
    stored = base.astype(dtype)
    query_values = queries.astype(dtype)
    blobs = [row.tobytes() for row in stored]
    query_blobs = [row.tobytes() for row in query_values]
    conversion_ms = (time.perf_counter_ns() - conversion_start) / 1e6
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    path = ":memory:" if case["storage"] == "memory" else str(work / f"{os.getpid()}.db")
    db = connect(args.extension, path, case)
    wrapper = "vec_f16(?)" if case["dtype"] == "f16" else "?"
    column = "float16" if case["dtype"] == "f16" else "float"
    db.execute(f"CREATE VIRTUAL TABLE v USING vec0(e {column}[{d}] distance_metric={case['metric']}, category integer, chunk_size={case['chunk']})")
    settings = {p: db.execute(f"PRAGMA {p}").fetchall() for p in
                ("page_size", "cache_size", "mmap_size", "journal_mode", "synchronous")}
    insert_sql = f"INSERT INTO v(rowid,e,category) VALUES (?,{wrapper},?)"
    # Metadata is identical across builds; clustered/scattered filters exercise chunk reads.
    categories = (np.arange(n) * 100 // n if case["filter_layout"] == "clustered"
                  else np.arange(n) % 100)
    def insert():
        for start in range(0, n, 1000):
            with db:
                db.executemany(insert_sql, ((i + 1, blobs[i], int(categories[i]))
                                            for i in range(start, min(n, start + 1000))))
    insert_ms, _ = timed(insert)
    checkpoint_ms, _ = timed(lambda: db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall())
    assert db.execute("SELECT count(*) FROM v").fetchone()[0] == n
    filter_sql = "" if case["selectivity"] == 100 else f" AND category < {case['selectivity']}"
    knn = f"SELECT rowid,distance FROM v WHERE e MATCH {wrapper} AND k=?{filter_sql} ORDER BY distance"
    projected = knn.replace("rowid,distance", "rowid,distance,e")
    samples = {"knn_ms": [], "knn_embedding_ms": [], "fetch_ms": []}
    for q in query_blobs[:5]:
        db.execute(projected, (q, case["k"])).fetchall()
    answers = []
    for q in query_blobs[5:]:
        ms, rows = timed(lambda: db.execute(knn, (q, case["k"])).fetchall())
        samples["knn_ms"].append(ms)
        answers.append(rows)
        ms, _ = timed(lambda: db.execute(projected, (q, case["k"])).fetchall())
        samples["knn_embedding_ms"].append(ms)
    for rowid in rng.integers(1, n + 1, size=1000):
        ms, result = timed(lambda: db.execute("SELECT e FROM v WHERE rowid=?", (int(rowid),)).fetchone())
        assert result[0] == blobs[rowid - 1]
        samples["fetch_ms"].append(ms)
    def scan():
        count = 0
        for (blob,) in db.execute("SELECT e FROM v"):
            count += len(blob)
        return count
    scan_ms, scanned_bytes = timed(scan)
    assert scanned_bytes == n * d * np.dtype(dtype).itemsize
    # Reference checks are outside all timing and bounded in memory.
    allowed = np.flatnonzero(categories < case["selectivity"])
    exact_overlap, fp32_overlap, max_errors = [], [], []
    def distances(matrix, query):
        out = np.empty(len(matrix), dtype=np.float64)
        for start in range(0, len(matrix), 2048):
            x = matrix[start:start + 2048].astype(np.float64)
            q = query.astype(np.float64)
            if case["metric"] == "l2":
                out[start:start + len(x)] = np.sqrt(np.sum((x - q) ** 2, axis=1))
            else:
                out[start:start + len(x)] = 1 - (x @ q) / (np.linalg.norm(x, axis=1) * np.linalg.norm(q))
        return out
    for i, rows in enumerate(answers[:3]):
        dist = distances(stored, query_values[i + 5])
        original_dist = distances(base, queries[i + 5])
        k = min(case["k"], len(allowed))
        expected = set(allowed[np.argsort(dist[allowed])[:k]] + 1)
        original = set(allowed[np.argsort(original_dist[allowed])[:k]] + 1)
        actual = {r[0] for r in rows}
        exact_overlap.append(len(actual & expected) / max(k, 1))
        fp32_overlap.append(len(actual & original) / max(k, 1))
        errors = [abs(distance - dist[rowid - 1]) for rowid, distance in rows]
        max_errors.append(max(errors, default=0))
        # Near ties can reorder within fp32 reduction error, but a farther neighbor cannot win.
        if k:
            threshold = np.sort(dist[allowed])[k - 1]
            assert len(rows) == k and len(actual) == k and actual <= set(allowed + 1)
            assert all(dist[rowid - 1] <= threshold + 2e-5 * max(1, threshold) for rowid, _ in rows)
            assert np.allclose([r[1] for r in rows], [dist[r[0] - 1] for r in rows], rtol=2e-5, atol=2e-5)
    logical_bytes = db.execute("PRAGMA page_count").fetchone()[0] * 4096
    result = dict(case=case, insert_ms=insert_ms, checkpoint_ms=checkpoint_ms,
                  conversion_ms=conversion_ms, scan_ms=scan_ms, scanned_bytes=scanned_bytes,
                  logical_bytes=logical_bytes, settings=settings, samples=samples,
                  exact_overlap=exact_overlap, fp32_overlap=fp32_overlap, max_distance_error=max_errors,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  debug=db.execute("SELECT vec_debug()").fetchone()[0])
    result["summary"] = {
        name: {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
        for name, times in samples.items()
    }
    result["summary"].update(insert_rows_s=n * 1000 / insert_ms,
                             scan_mib_s=scanned_bytes / 1048576 * 1000 / scan_ms)
    db.close()
    if path != ":memory:":
        result["file_bytes"] = Path(path).stat().st_size
        Path(path).unlink()
    print(json.dumps(result))


def run(args):
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    source_root = Path(args.source_dir) if args.source_dir else root
    sources = [source_root / "sqlite-vec.c", *sorted(source_root.glob("sqlite-vec-*.c")), source_root / "Makefile", source_root / "sqlite-vec.h"]
    report = {"format": 1, "label": args.label, "extension": str(Path(args.extension).resolve()),
              "binary_sha256": digest(args.extension), "build_flags": args.build_flags,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_hashes": {p.name: digest(p) for p in sources},
              "cpu": subprocess.check_output(["lscpu"], text=True),
              "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
              "python": sys.version, "sqlite": sqlite3.sqlite_version, "numpy": np.__version__,
              "platform": platform.platform(), "results": []}
    report["complete"] = False
    report["command"] = sys.argv
    report["expected_results"] = len(args.rows) * len(args.dims) * len(args.metrics) * len(args.storage) * len(args.dtypes) * args.repetitions
    for n, d, metric, storage in itertools.product(args.rows, args.dims, args.metrics, args.storage):
        for repetition in range(args.repetitions):
            for dtype in (args.dtypes if repetition % 2 == 0 else args.dtypes[::-1]):
                case = dict(rows=n, dims=d, metric=metric, storage=storage, dtype=dtype,
                            seed=args.seed, k=args.k, queries=args.queries, repetition=repetition,
                            normalize=args.normalize, chunk=args.chunk, mmap=args.mmap,
                            selectivity=args.selectivity, filter_layout=args.filter_layout)
                print(f"{args.label}: {case}", flush=True)
                env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
                result = subprocess.run([sys.executable, __file__, "worker", "--extension", args.extension,
                                         "--work", str(output.parent / "work"), "--case", json.dumps(case)],
                                        text=True, capture_output=True, env=env)
                if result.returncode:
                    raise RuntimeError(f"Benchmark failed: {case}\n{result.stderr}\n{result.stdout}")
                report["results"].append(json.loads(result.stdout))
                temp = output.with_suffix(".tmp")
                temp.write_text(json.dumps(report, indent=2) + "\n")
                temp.replace(output)
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")


def compare(args):
    left, right = [json.loads(Path(p).read_text()) for p in (args.baseline, args.candidate)]
    for report, dtype in ((left, args.baseline_dtype), (right, args.candidate_dtype)):
        if dtype:
            report["results"] = [r for r in report["results"] if r["case"]["dtype"] == dtype]
        if not report["results"]:
            raise SystemExit("No benchmark results for requested dtype")
        if report.get("complete") is False:
            raise SystemExit("Cannot compare an incomplete benchmark run")
        if args.cross_dtype and len({r["case"]["dtype"] for r in report["results"]}) != 1:
            raise SystemExit("--cross-dtype requires one dtype per report")
    def indexed(report):
        groups = {}
        for result in report["results"]:
            case = dict(result["case"])
            case.pop("repetition")
            if args.cross_dtype:
                case.pop("dtype")
            key = json.dumps(case, sort_keys=True)
            groups.setdefault(key, []).append(result)
        return groups
    a, b = indexed(left), indexed(right)
    if a.keys() != b.keys():
        raise SystemExit("Workload mismatch; compare matching suites (use --cross-dtype for f32/f16)")
    if any(sorted(r["case"]["repetition"] for r in a[key]) != sorted(r["case"]["repetition"] for r in b[key]) for key in a):
        raise SystemExit("Repetition mismatch; compare matching suites")
    left_types = "/".join(sorted({r["case"]["dtype"] for r in left["results"]}))
    right_types = "/".join(sorted({r["case"]["dtype"] for r in right["results"]}))
    print(f"# {left['label']} ({left_types}) vs {right['label']} ({right_types})\n\nSpeedup = baseline time / candidate time.\n")
    print("| Rows × dims / metric / storage | Insert | KNN | KNN + embedding | Fetch | Scan | Bytes ratio |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for key in a:
        x, y = a[key], b[key]
        c = json.loads(key)
        def med(results, field):
            return float(np.median([r[field] if field in r else np.median(r['samples'][field]) for r in results]))
        ratios = [med(x, f) / med(y, f) for f in ("insert_ms", "knn_ms", "knn_embedding_ms", "fetch_ms", "scan_ms", "logical_bytes")]
        print(f"| {c['rows']} × {c['dims']} / {c['metric']} / {c['storage']} | " + " | ".join(f"{v:.3f}×" for v in ratios) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("run")
    p.add_argument("--extension", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--build-flags", required=True)
    p.add_argument("--source-dir", help="Frozen build sources for provenance (defaults to checkout)")
    p.add_argument("--rows", type=int, nargs="+", default=[10000, 100000])
    p.add_argument("--dims", type=int, nargs="+", default=[384, 768, 1536])
    p.add_argument("--metrics", choices=["l2", "cosine"], nargs="+", default=["l2", "cosine"])
    p.add_argument("--storage", choices=["memory", "file"], nargs="+", default=["memory", "file"])
    p.add_argument("--dtypes", choices=["f32", "f16"], nargs="+", default=["f32"])
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--mmap", type=int, default=0)
    p.add_argument("--selectivity", type=int, choices=[1, 10, 100], default=100)
    p.add_argument("--filter-layout", choices=["clustered", "scattered"], default="clustered")
    p.add_argument("--normalize", action="store_true")
    p = subs.add_parser("worker")
    p.add_argument("--extension", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--work", required=True)
    p = subs.add_parser("compare")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--cross-dtype", action="store_true")
    p.add_argument("--baseline-dtype", choices=["f32", "f16"])
    p.add_argument("--candidate-dtype", choices=["f32", "f16"])
    args = parser.parse_args()
    if args.command == "run" and (min(args.rows + args.dims) < 1 or args.k < 1 or args.queries < 1 or args.repetitions < 1 or args.chunk < 8 or args.chunk % 8):
        parser.error("rows, dims, k, queries, repetitions must be positive; chunk must be a positive multiple of 8")
    {"run": run, "worker": worker, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
