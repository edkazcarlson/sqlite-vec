"""Run paired exact-search ablations using prebuilt extensions, one worker at a time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builds", default="benchmarks/exact/results/builds")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-final", action="store_true",
                        help="Recheck file-backed L2 changes against alternating baseline runs")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    builds = Path(args.builds)
    variants = ["baseline", "fp16-initial", "simd", "heap", "filter", "combined", "fma"]
    if args.verify_final:
        variants = ["baseline", "heap", "filter", "final"]
    report = {"format": 1, "builds": {}, "results": []}
    for variant in variants:
        path = builds / variant / "vec0.so"
        report["builds"][variant] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    profiles = []
    # Isolate SIMD and selection changes, including a non-multiple-of-16 dimension.
    for dims in (384, 768, 769):
        for metric in ("l2", "cosine"):
            profiles.append((dict(rows=10000, dims=dims, metric=metric), ["baseline", "simd", "heap", "combined", "fma"]))
    for k in (1, 100):
        profiles.append((dict(rows=10000, dims=768, k=k), ["baseline", "heap", "combined"]))
    # Empty-chunk elimination vs scattered filters; one percent and ten percent.
    for layout in ("clustered", "scattered"):
        for selectivity in (1, 10):
            profiles.append((dict(rows=100000, dims=768, storage="file", filter_layout=layout, selectivity=selectivity), ["baseline", "filter", "combined"]))
    # Normalization cost/quality and storage tuning are application choices.
    for dtype in ("f32", "f16"):
        profiles.append((dict(rows=100000, dims=768, metric="cosine", dtype=dtype), ["combined"]))
        profiles.append((dict(rows=100000, dims=768, normalize=True, dtype=dtype), ["combined"]))
    for chunk in (256, 1024, 4096):
        for mmap in (0, 1073741824):
            profiles.append((dict(rows=100000, dims=768, storage="file", chunk=chunk, mmap=mmap), ["combined"]))
    if args.verify_final:
        profiles = [(dict(rows=n, dims=768, storage="file"), variants) for n in (10000, 100000)]
        profiles.append((dict(rows=10000, dims=768), ["baseline", "final"]))
    for overrides, names in profiles:
        for repetition in range(3):
            for variant in (names if repetition % 2 == 0 else names[::-1]):
                case = dict(rows=10000, dims=768, metric="l2", storage="memory", dtype="f32",
                            seed=20260916, k=10, queries=30, repetition=repetition,
                            normalize=False, chunk=1024, mmap=0, selectivity=100, filter_layout="clustered")
                case.update(overrides)
                print(variant, case, flush=True)
                extension = builds / variant / "vec0.so"
                result = subprocess.run([sys.executable, str(Path(__file__).with_name("bench.py")), "worker",
                                         "--extension", str(extension), "--work", str(output.parent / "work"),
                                         "--case", json.dumps(case)], capture_output=True, text=True,
                                        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
                if result.returncode:
                    raise RuntimeError(result.stderr + result.stdout)
                entry = json.loads(result.stdout)
                entry["variant"] = variant
                report["results"].append(entry)
                tmp = output.with_suffix(".tmp")
                tmp.write_text(json.dumps(report, indent=2) + "\n")
                tmp.replace(output)


if __name__ == "__main__":
    main()
