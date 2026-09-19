"""Structural exact-search experiments. Native prototypes are not SQLite indexes."""
import argparse
import ctypes as ct
import hashlib
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
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAMES = {0: 'two-pass', 1: 'partial-l2', 2: 'va8', 3: 'cells', 4: 'tile8', 5: 'cached-norm', 6: 'fused', 7: 'residual8'}


class Stats(ct.Structure):
    _fields_ = [(x, ct.c_ulonglong) for x in ('full', 'coordinates', 'bounds', 'cells', 'bytes')] + [(x, ct.c_double) for x in ('bound_ms', 'score_ms', 'select_ms')]


def ptr(a):
    return None if a is None else a.ctypes.data_as(ct.c_void_p)


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, report):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(path)


def prepare(directory):
    for split in ('train', 'test'):
        source = directory / f'{split}.parquet'
        dest = directory / f'{split}.npy'
        f = pq.ParquetFile(source)
        first = next(f.iter_batches(batch_size=1, columns=['emb'])).column(0)
        d = len(first[0])
        out = np.lib.format.open_memmap(dest, mode='w+', dtype='float32', shape=(f.metadata.num_rows, d))
        offset = 0
        for batch in f.iter_batches(batch_size=4096, columns=['emb']):
            emb = batch.column(0)
            lengths = np.diff(emb.offsets.to_numpy())
            if not np.all(lengths == d):
                raise ValueError('Embedding dimensions vary')
            values = emb.flatten().to_numpy().reshape(-1, d)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite dataset values')
            out[offset:offset + len(values)] = values
            offset += len(values)
        out.flush()
        print(split, out.shape, flush=True)
        (directory / f'{split}.manifest.json').write_text(json.dumps(dict(
            source=f'https://assets.zilliz.com/benchmark/cohere_medium_1m/{split}.parquet',
            source_sha256=sha(source), array_sha256=sha(dest), shape=list(out.shape)), indent=2))


def data(args):
    rng = np.random.default_rng(20260916)
    if args.dataset == 'cohere':
        base = np.load(args.data / 'train.npy', mmap_mode='r')
        query = np.load(args.data / 'test.npy', mmap_mode='r')
        if args.rows > len(base) or args.queries + 5 > len(query):
            raise ValueError('Requested more rows/queries than dataset contains')
        return np.ascontiguousarray(base[:args.rows]), np.ascontiguousarray(query[:args.queries + 5])
    n, d = args.rows, args.dims
    base = rng.standard_normal((n, d), dtype=np.float32)
    query = rng.standard_normal((args.queries + 5, d), dtype=np.float32)
    if args.dataset == 'clustered':
        centers = rng.standard_normal((32, d), dtype=np.float32) * 5
        base = base * .15 + centers[np.arange(n) % 32]
        query = query * .15 + centers[np.arange(len(query)) % 32]
    elif args.dataset == 'anisotropic':
        scale = np.geomspace(1, .01, d).astype('float32')
        base *= scale
        query *= scale
    return base, query


def cluster(base):
    rng = np.random.default_rng(19)
    sample = base[rng.choice(len(base), min(4096, len(base)), replace=False)].astype('float32')
    nc = min(32, len(sample))
    centers = sample[rng.choice(len(sample), nc, replace=False)].copy()
    def assign(x):
        return np.argmin(np.sum(x*x, axis=1)[:, None] + np.sum(centers*centers, axis=1)[None, :] - 2*x@centers.T, axis=1)
    for _ in range(8):
        labels = assign(sample)
        for c in range(nc):
            members = sample[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    labels = np.empty(len(base), dtype='int32')
    for start in range(0, len(base), 4096):
        labels[start:start+4096] = assign(base[start:start+4096].astype('float32'))
    return centers, labels


def library():
    out = HERE / 'results/builds/structural.so'
    out.parent.mkdir(parents=True, exist_ok=True)
    command = ['cc', '-O3', '-fPIC', '-shared', '-mavx2', '-mf16c', '-DSQLITE_VEC_ENABLE_AVX', '-Ivendor', 'benchmarks/exact/structural.c', '-lm', '-o', str(out)]
    subprocess.run(command, cwd=ROOT, check=True)
    lib = ct.CDLL(str(out))
    lib.experiment_build.argtypes = [ct.c_void_p] + [ct.c_int]*5 + [ct.c_void_p, ct.c_void_p, ct.c_int]
    lib.experiment_build.restype = ct.c_void_p
    lib.experiment_query.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p, ct.POINTER(Stats), ct.c_int]
    lib.experiment_query.restype = ct.c_int
    lib.experiment_kernel.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
    lib.experiment_free.argtypes = [ct.c_void_p]
    return lib, command, sha(out)


def summary(samples):
    return dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)))


def native(args, report):
    base, queries = data(args)
    raw = base.astype(args.dtype)
    queries = queries.astype(args.dtype).astype('float32')
    n, d = raw.shape
    lib, command, binary = library()
    report.update(command=command, binary_sha256=binary, dimensions=d)
    modes = [int(m) for m in args.modes.split(',')]
    if args.metric == 'cosine':
        modes = [m for m in modes if m not in (1, 2, 3, 7)]
    modes = list(dict.fromkeys([0] + modes))
    indexes = {}
    keepalive = []
    for mode in modes:
        start = time.perf_counter()
        centers, labels = cluster(raw) if mode == 3 else (None, None)
        keepalive.extend([centers, labels])
        index = lib.experiment_build(ptr(raw), n, d, args.dtype == 'float16', args.metric == 'cosine', mode, ptr(centers), ptr(labels), 0 if centers is None else len(centers))
        if not index:
            raise RuntimeError(f'Building {NAMES[mode]} failed')
        indexes[mode] = index
        report.setdefault('builds', {})[NAMES[mode]] = dict(seconds=time.perf_counter()-start)
        print('built', NAMES[mode], report['builds'][NAMES[mode]], flush=True)
    ids = np.empty(args.k, dtype='int32')
    scores = np.empty(args.k, dtype='float32')
    output = np.empty(n, dtype='float32')
    stats = Stats()
    def query(mode, q, instrument=0):
        rc = lib.experiment_query(indexes[mode], ptr(q), args.k, ptr(ids), ptr(scores), ct.byref(stats), instrument)
        if rc != args.k:
            raise RuntimeError(f'Query failed: {rc}')
        return ids.copy(), scores.copy()
    expected = [query(0, q) for q in queries[5:]]
    # Independent float64 reference for EVERY timed query, outside timing.
    # Near-score ties are checked by distances; native variants must additionally
    # match the exhaustive production scorer's IDs and score bits exactly.
    for qi, q in enumerate(queries[5:]):
        exact = np.empty(n, dtype='float64')
        for start in range(0, n, 2048):
            x = raw[start:start+2048].astype('float64')
            if args.metric == 'l2':
                dist = np.linalg.norm(x-q.astype('float64'), axis=1)
            else:
                dist = 1 - (x@q.astype('float64'))/(np.linalg.norm(x, axis=1)*np.linalg.norm(q.astype('float64')))
            exact[start:start+len(x)] = dist
        want = np.sort(exact)[:args.k]
        np.testing.assert_allclose(exact[expected[qi][0]], want, atol=2e-5, rtol=2e-5)
    for repetition in range(args.repetitions):
        for mode in (modes if repetition % 2 == 0 else modes[::-1]):
            kernel_only = mode == 4
            for q in queries[:5]:
                if kernel_only:
                    lib.experiment_kernel(indexes[mode], ptr(q), ptr(output))
                else:
                    query(mode, q)
            times, max_error, min_overlap = [], 0., 1.
            for qi, q in enumerate(queries[5:]):
                start = time.perf_counter_ns()
                if kernel_only:
                    lib.experiment_kernel(indexes[mode], ptr(q), ptr(output))
                else:
                    result = query(mode, q)
                times.append((time.perf_counter_ns()-start)/1e6)
                if kernel_only:
                    chosen = np.argsort(output, kind='stable')[:args.k]
                    min_overlap = min(min_overlap, len(set(chosen)&set(expected[qi][0]))/args.k)
                    max_error = max(max_error, float(np.max(np.abs(output[expected[qi][0]]-expected[qi][1]))))
                else:
                    np.testing.assert_array_equal(result[0], expected[qi][0])
                    np.testing.assert_array_equal(result[1], expected[qi][1])
            counters = []
            if not kernel_only:
                for q in queries[5:]:
                    query(mode, q, 1)
                    counters.append({name: getattr(stats, name) for name, _ in stats._fields_})
            entry = dict(variant=NAMES[mode], repetition=repetition, kernel_only=kernel_only,
                         samples_ms=times, **summary(times), counters=counters,
                         min_overlap=min_overlap, max_score_error=max_error)
            report.setdefault('results', []).append(entry)
            save(args.output, report)
            print(NAMES[mode], repetition, summary(times), 'overlap', min_overlap, flush=True)
    for index in indexes.values():
        lib.experiment_free(index)


def sql(args, report):
    if len(args.extensions) != 1:
        raise ValueError('SQL workers must load exactly one extension build')
    base, queries = data(args)
    raw = base.astype(args.dtype)
    queries = queries.astype(args.dtype)
    n, d = raw.shape
    wrapper = 'vec_f16(?)' if args.dtype == 'float16' else '?'
    kind = 'float16' if args.dtype == 'float16' else 'float'
    paths = [Path(p).resolve() for p in args.extensions]
    indexes = args.indexes or ['flat'] * len(paths)
    if len(indexes) != len(paths):
        raise ValueError('--indexes must have one entry per extension')
    names = [p.parent.name + ('' if index == 'flat' else ':' + index)
             for p, index in zip(paths, indexes)]
    databases = []
    for index, extension in enumerate(paths):
        path = ':memory:' if args.storage == 'memory' else str(HERE / f'results/work/structural-{os.getpid()}-{index}.db')
        db = sqlite3.connect(path)
        db.enable_load_extension(True)
        db.load_extension(str(extension), entrypoint='sqlite3_vec_init')
        db.enable_load_extension(False)
        db.executescript('PRAGMA page_size=4096; PRAGMA cache_size=-65536; PRAGMA mmap_size=0; PRAGMA synchronous=FULL;')
        if path != ':memory:':
            db.execute('PRAGMA journal_mode=WAL')
        index_clause = '' if indexes[index] == 'flat' else ' indexed by exact_va()'
        db.execute(f'CREATE VIRTUAL TABLE v USING vec0(e {kind}[{d}] distance_metric={args.metric}{index_clause}, category integer)')
        start = time.perf_counter()
        for lo in range(0, n, 1000):
            with db:
                db.executemany(f'INSERT INTO v(rowid,e,category) VALUES (?,{wrapper},?)',
                               ((i+1, raw[i].tobytes(), i%100) for i in range(lo, min(n,lo+1000))))
        insert_s = time.perf_counter()-start
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchall()
        report.setdefault('builds', {})[names[index]] = dict(insert_seconds=insert_s,
            bytes=db.execute('PRAGMA page_count').fetchone()[0]*4096,
            binary_sha256=sha(extension), debug=db.execute('SELECT vec_debug()').fetchone()[0])
        manifest = extension.parent / 'manifest.json'
        if manifest.exists():
            report['builds'][names[index]]['manifest'] = json.loads(manifest.read_text())
        databases.append((db, path))
        print('inserted', extension.parent.name, insert_s, flush=True)
    for selectivity in args.selectivities:
        for restriction in ('metadata', 'rowid') if selectivity != 100 else ('metadata',):
            predicate = f' AND category < {selectivity}' if restriction == 'metadata' else ' AND rowid IN (' + ','.join(str(i+1) for i in range(0,n,max(1,100//selectivity))) + ')'
            statement = f'SELECT rowid,distance FROM v WHERE e MATCH {wrapper} AND k={args.k}{predicate} ORDER BY distance'
            expected = [databases[0][0].execute(statement,(q.tobytes(),)).fetchall() for q in queries[5:]]
            report.setdefault('answers_sha256',{})[f'{restriction}:{selectivity}'] = hashlib.sha256(
                json.dumps(expected, separators=(',',':')).encode()).hexdigest()
            for rep in range(args.repetitions):
                order = range(len(paths)) if rep%2==0 else reversed(range(len(paths)))
                for i in order:
                    db = databases[i][0]
                    for q in queries[:5]: db.execute(statement,(q.tobytes(),)).fetchall()
                    samples=[]
                    for qi,q in enumerate(queries[5:]):
                        start=time.perf_counter_ns()
                        result=db.execute(statement,(q.tobytes(),)).fetchall()
                        samples.append((time.perf_counter_ns()-start)/1e6)
                        if result!=expected[qi]: raise AssertionError((paths[i],qi,result,expected[qi]))
                    entry=dict(variant=names[i], repetition=rep, selectivity=selectivity,
                               restriction=restriction, samples_ms=samples, **summary(samples))
                    report.setdefault('results',[]).append(entry);save(args.output,report)
                    print(entry['variant'], restriction, selectivity, rep, summary(samples),flush=True)
            if args.instrument:
                for i, (db, _) in enumerate(databases):
                    counters=[]
                    for q in queries[5:]:
                        db.execute(statement,(q.tobytes(),)).fetchall()
                        counters.append(json.loads(db.execute('SELECT vec_bench_stats()').fetchone()[0]))
                    report.setdefault('instrumentation',[]).append(dict(
                        variant=names[i], selectivity=selectivity,
                        restriction=restriction, counters=counters))
                save(args.output,report)
    for i, (db,path) in enumerate(databases):
        count=min(n,1000)
        start=time.perf_counter()
        with db:
            db.executemany(f'UPDATE v SET e={wrapper} WHERE rowid=?',
                           [(raw[j].tobytes(),j+1) for j in range(count)])
        update_s=time.perf_counter()-start
        start=time.perf_counter()
        with db:
            db.executemany('DELETE FROM v WHERE rowid=?',[(j+1,) for j in range(count)])
        delete_s=time.perf_counter()-start
        report.setdefault('mutations',{})[names[i]]=dict(rows=count,update_seconds=update_s,delete_seconds=delete_s)
        db.close()
        if path!=':memory:': Path(path).unlink()


def sql_isolated(args, report):
    """Never dlopen two builds into one process: exported symbols may interpose."""
    indexes=args.indexes or ['flat']*len(args.extensions)
    if len(indexes)!=len(args.extensions):
        raise ValueError('--indexes must have one entry per extension')
    names=[Path(p).parent.name+('' if idx=='flat' else ':'+idx)
           for p,idx in zip(args.extensions,indexes)]
    if len(set(names))!=len(names):
        raise ValueError('Variant names must be unique; use separate build directories')
    work=HERE/'results/work';work.mkdir(parents=True,exist_ok=True)
    expected=report.get('workers',[{}])[0].get('answers_sha256')
    done={(w['repetition'],next(iter(w['builds']))) for w in report.get('workers',[])}
    for repetition in range(args.repetitions):
        order=range(len(args.extensions)) if repetition%2==0 else reversed(range(len(args.extensions)))
        for i in order:
            name=Path(args.extensions[i]).parent.name+('' if indexes[i]=='flat' else ':'+indexes[i])
            if (repetition,name) in done:
                continue
            output=work/f'{args.output.stem}-{os.getpid()}-{repetition}-{i}.json'
            command=[sys.executable,str(Path(__file__).resolve()),'sql','--sql-worker',
                     '--extensions',args.extensions[i],'--indexes',indexes[i],
                     '--output',str(output),'--repetitions','1']
            for key in ('dataset','data','rows','dims','queries','k','dtype','metric','storage'):
                command.extend(['--'+key,str(getattr(args,key))])
            command.extend(['--selectivities', *map(str,args.selectivities)])
            if args.instrument:command.append('--instrument')
            print('worker',repetition,Path(args.extensions[i]).parent.name,indexes[i],flush=True)
            subprocess.run(command,check=True)
            result=json.loads(output.read_text())
            if not result.get('complete'):raise RuntimeError(f'Incomplete worker: {output}')
            if expected is None:expected=result['answers_sha256']
            if result['answers_sha256']!=expected:
                raise AssertionError(f'Exact answers differ from baseline; inspect {output}')
            for row in result['results']:
                row['repetition']=repetition
            report.setdefault('results',[]).extend(result['results'])
            report.setdefault('workers',[]).append(dict(repetition=repetition,
                builds=result['builds'], mutations=result['mutations'],
                peak_rss_kib=result['peak_rss_kib'], answers_sha256=result['answers_sha256']))
            report.setdefault('instrumentation',[]).extend(result.get('instrumentation',[]))
            save(args.output,report)
    report['validation']='All timed queries agree bit-for-bit with first isolated baseline worker'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','native','sql'])
    parser.add_argument('--data',type=Path,default=HERE/'results/work/cohere')
    parser.add_argument('--dataset',choices=['cohere','gaussian','clustered','anisotropic'],default='cohere')
    parser.add_argument('--rows',type=int,default=100000)
    parser.add_argument('--dims',type=int,default=768)
    parser.add_argument('--queries',type=int,default=100)
    parser.add_argument('--repetitions',type=int,default=5)
    parser.add_argument('--k',type=int,default=10)
    parser.add_argument('--dtype',choices=['float32','float16'],default='float32')
    parser.add_argument('--metric',choices=['l2','cosine'],default='l2')
    parser.add_argument('--storage',choices=['memory','file'],default='file')
    parser.add_argument('--selectivities',nargs='+',type=int,choices=range(1,101),default=[1,10,100])
    parser.add_argument('--modes',default='0,1,2,3,4,6')
    parser.add_argument('--extensions',nargs='+',default=[str(HERE/'results/builds/structural-baseline/vec0.so'),str(HERE/'results/builds/structural-sparse/vec0.so')])
    parser.add_argument('--indexes',nargs='+',choices=['flat','exact_va'],help='Table index for each extension, default all flat')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--instrument',action='store_true',help='Require diagnostic builds; collect counters separately from timings')
    parser.add_argument('--sql-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--resume',action='store_true',help='Resume an interrupted isolated SQL run after checking workload and binary hashes')
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.data);return
    if not args.output: parser.error('--output required')
    prior=None
    if args.output.exists():
        if not args.resume:raise FileExistsError(args.output)
        if args.action!='sql' or args.sql_worker:parser.error('--resume requires isolated SQL mode')
        prior=json.loads(args.output.read_text())
        if prior.get('invalid_reason'):
            raise ValueError('Cannot resume invalid measurements; use a new output file')
        for key in ('dataset','data','rows','dims','queries','repetitions','k','dtype','metric','storage','extensions','indexes','selectivities','instrument'):
            value=getattr(args,key)
            if isinstance(value,Path):value=str(value)
            old=prior['case'].get(key, [1,10,100] if key=='selectivities' else None)
            if old!=value:raise ValueError(f'Resume workload mismatch: {key}')
        binaries={Path(p).parent.name:sha(p) for p in args.extensions}
        for worker in prior.get('workers',[]):
            for name,build in worker['builds'].items():
                if build['binary_sha256']!=binaries[name.split(':')[0]]:
                    raise ValueError(f'Resume binary mismatch: {name}')
        if prior.get('complete'):
            print('Already complete:',args.output);return
    if min(args.rows,args.dims,args.queries,args.repetitions,args.k)<=0 or args.k>args.rows:
        parser.error('Positive sizes required; k must not exceed rows')
    report=dict(format=1,case={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                machine=platform.uname()._asdict(),sqlite=sqlite3.sqlite_version,
                cpu=Path('/proc/cpuinfo').read_text().split('\n\n',1)[0],
                compiler=subprocess.check_output(['cc','--version'],text=True),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), HERE/'structural.c', *ROOT.glob('sqlite-vec*.c'), ROOT/'sqlite-vec.h']},
                threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']})
    if args.dataset=='cohere': report['dataset']=[json.loads((args.data/f'{s}.manifest.json').read_text()) for s in ('train','test')]
    if prior is not None:
        if prior.get('dataset')!=report.get('dataset'):raise ValueError('Resume dataset mismatch')
        prior.setdefault('resumes',[]).append(dict(source_sha256=report['source_sha256']))
        report=prior
    (native if args.action=='native' else sql if args.sql_worker else sql_isolated)(args,report)
    report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['complete']=True;save(args.output,report)


if __name__=='__main__':
    main()
