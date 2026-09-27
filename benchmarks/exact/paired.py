"""Alternate parent/candidate workers; keep raw results and provenance."""
import argparse
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from bench import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-extension', required=True)
    p.add_argument('--candidate-extension', required=True)
    p.add_argument('--baseline-dtypes', nargs='+', default=['f32', 'f16'], choices=['f32', 'f16'])
    p.add_argument('--candidate-dtypes', nargs='+', default=['f32', 'f16'], choices=['f32', 'f16'])
    p.add_argument('--output', required=True)
    p.add_argument('--build-flags', required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        p.error('Output already exists; preserve previous evidence.')
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    reports = {}
    for label, extension in [('baseline', a.baseline_extension), ('candidate', a.candidate_extension)]:
        reports[label] = dict(format=1, label=label, complete=False, results=[],
            extension=str(Path(extension).resolve()), binary_sha256=digest(extension),
            build_flags=a.build_flags, platform=platform.platform(),
            compiler=os.environ.get('VEC_BENCH_COMPILER', 'unavailable'),
            cpu=os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
            checkout_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
    # Checkout hashes identify candidate sources only; the baseline binary is
    # separately preserved and identified by its hash (not by this checkout).
    reports['candidate']['source_hashes'] = {f.name: digest(f) for f in root.glob('sqlite-vec*.c')}
    variants = [('baseline', d) for d in a.baseline_dtypes] + [('candidate', d) for d in a.candidate_dtypes]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for dims, metric, storage in itertools.product([384, 768, 769, 1536], ['l2', 'cosine'], ['memory', 'file']):
        for rep in range(3):
            for label, dtype in (variants if rep % 2 == 0 else variants[::-1]):
                case = dict(rows=2000, dims=dims, metric=metric, storage=storage,
                    dtype=dtype, seed=20260916, k=10, queries=20, repetition=rep,
                    normalize=False, chunk=1024, mmap=0, selectivity=100, filter_layout='clustered')
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('bench.py')),
                    'worker', '--extension', reports[label]['extension'], '--case', json.dumps(case),
                    '--work', str(output.parent/'work')], capture_output=True, text=True, env=env)
                if result.returncode:
                    raise RuntimeError(result.stderr + result.stdout)
                reports[label]['results'].append(json.loads(result.stdout))
        print(dims, metric, storage, flush=True)
        output.write_text(json.dumps(reports, indent=2))
    for report in reports.values():
        report['complete'] = True
        report['expected_results'] = len(report['results'])
    output.write_text(json.dumps(reports, indent=2))
    for label, report in reports.items():
        output.with_name(output.stem + '-' + label + '.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
