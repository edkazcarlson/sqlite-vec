"""Freeze independently controlled structural experiment builds and provenance."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from structural import ROOT, HERE, sha

FLAGS = {
    'control': [],
    'reopen': ['SQLITE_VEC_BLOB_REOPEN=1'],
    'gather': ['SQLITE_VEC_SPARSE_READS=1','SQLITE_VEC_BLOB_REOPEN=0'],
    'route': ['SQLITE_VEC_ROWID_ROUTING=1'],
    'heap': ['SQLITE_VEC_GLOBAL_HEAP=1'],
    'va': ['SQLITE_VEC_EXPERIMENTAL_EXACT_VA=1'],
    'combined': ['SQLITE_VEC_SPARSE_READS=1','SQLITE_VEC_BLOB_REOPEN=1','SQLITE_VEC_ROWID_ROUTING=1','SQLITE_VEC_GLOBAL_HEAP=1'],
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants',nargs='+',choices=list(FLAGS),default=list(FLAGS))
    parser.add_argument('--instrument',action='store_true')
    parser.add_argument('--tag',default='',help='Suffix for a new immutable build set')
    args=parser.parse_args()
    for variant in args.variants:
        name='structural-'+variant+args.tag+('-trace' if args.instrument else '')
        directory=HERE/'results/builds'/name
        if (directory/'manifest.json').exists(): raise FileExistsError(directory)
        sources=directory/'source';sources.mkdir(parents=True,exist_ok=True)
        for source in [*ROOT.glob('sqlite-vec*.c'),ROOT/'sqlite-vec.h']:
            shutil.copyfile(source,sources/source.name)
        flags=['-lm','-mavx','-mavx2','-DSQLITE_VEC_ENABLE_AVX']+['-D'+f for f in FLAGS[variant]]
        if args.instrument: flags+=['-DSQLITE_VEC_BENCHMARK']
        command=['cc','-O3','-fPIC','-shared','-I'+str(ROOT/'vendor'),*flags,str(sources/'sqlite-vec.c'),'-lm','-o',str(directory/'vec0.so')]
        subprocess.run(command,check=True)
        (directory/'manifest.json').write_text(json.dumps(dict(command=command,
            compiler=subprocess.check_output(['cc','--version'],text=True),
            binary_sha256=sha(directory/'vec0.so'),
            sources={p.name:sha(p) for p in sources.iterdir()}),indent=2))
        print(name,flush=True)


if __name__=='__main__':main()
