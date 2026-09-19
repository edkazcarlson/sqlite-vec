import os
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/exact/bench.py"
EXTENSION = Path(os.environ.get("VEC_TEST_EXTENSION", str(SCRIPT.parents[2] / "dist/vec0.so"))).resolve()


def cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


def test_benchmark_roundtrip(tmp_path):
    output = tmp_path / "results.json"
    command = ["run", "--extension", EXTENSION, "--output", output,
               "--label", "test", "--build-flags=test", "--rows", "48", "--dims", "17",
               "--queries", "2", "--repetitions", "1", "--metrics", "cosine",
               "--storage", "file", "--dtypes", "f32", "f16"]
    result = cli(*command)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["complete"] and report["expected_results"] == 2
    for row in report["results"]:
        assert row["exact_overlap"] == [1.0, 1.0]
        assert row["settings"]["journal_mode"] == [["wal"]]
        assert row["settings"]["synchronous"] == [[2]]
        assert len(row["samples"]["knn_ms"]) == 2
        assert len(row["samples"]["fetch_ms"]) == 1000
        assert row["summary"]["knn_ms"]["p95_ms"] > 0
    assert not list((tmp_path / "work").glob("*.db"))
    result = cli("compare", output, output, "--baseline-dtype", "f32",
                 "--candidate-dtype", "f16", "--cross-dtype")
    assert result.returncode == 0, result.stderr
    assert "test (f32) vs test (f16)" in result.stdout
    assert cli(*command).returncode != 0  # Never silently overwrite a baseline.
    assert cli("compare", output, output, "--cross-dtype").returncode != 0
    altered = tmp_path / "altered.json"
    report["results"][0]["case"]["seed"] += 1
    altered.write_text(json.dumps(report))
    result = cli("compare", output, altered)
    assert result.returncode != 0 and "Workload mismatch" in result.stderr
    report["complete"] = False
    altered.write_text(json.dumps(report))
    result = cli("compare", altered, altered)
    assert result.returncode != 0 and "incomplete" in result.stderr
