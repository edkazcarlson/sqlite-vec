"""Native pruning correctness, including adversarial finite inputs."""
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from structural import Stats, cluster, library, ptr


@pytest.fixture(scope='module')
def lib():
    return library()[0]


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('dims', [1,7,8,17,32,33,65,769])
def test_native_bounds(lib,dtype,dims):
    rng=np.random.default_rng(23)
    x=rng.normal(size=(129,dims)).astype(dtype)
    x[::7]=x[0]
    x[1]=np.nextafter(x[0],np.array(np.inf,dtype=dtype))
    x[2]=0
    centers,labels=cluster(x)
    modes=(0,1,2,3,6,7)
    indexes=[lib.experiment_build(ptr(x),len(x),dims,dtype=='float16',0,m,ptr(centers),ptr(labels),len(centers)) for m in modes]
    assert all(indexes)
    for k in (1,10,129):
        for q in (x[0].astype('float32'),x[-1].astype('float32'),np.zeros(dims,dtype='float32')):
            results=[]
            for index in indexes:
                ids=np.empty(k,dtype='int32');ds=np.empty(k,dtype='float32');stats=Stats()
                assert lib.experiment_query(index,ptr(q),k,ptr(ids),ptr(ds),ct.byref(stats),0)==k
                results.append((ids,ds))
            for ids,ds in results[1:]:
                np.testing.assert_array_equal(ids,results[0][0])
                np.testing.assert_array_equal(ds,results[0][1])
    for index in indexes:lib.experiment_free(index)


def test_sql_isolation_and_resume(tmp_path):
    script=Path(__file__).with_name('structural.py')
    root=script.resolve().parents[2]
    output=tmp_path/'result.json'
    command=[sys.executable,str(script),'sql','--dataset','gaussian','--rows','48',
             '--dims','17','--queries','2','--repetitions','2','--storage','memory',
             '--extensions',str(root/'dist/vec0.so'),'--output',str(output)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    result=subprocess.run(command,capture_output=True,text=True,env=env)
    assert result.returncode==0,result.stderr
    report=json.loads(output.read_text())
    assert report['complete'] and len(report['workers'])==2
    assert report['workers'][0]['answers_sha256']==report['workers'][1]['answers_sha256']
    assert subprocess.run(command,capture_output=True,env=env).returncode!=0
    assert subprocess.run(command+['--resume'],capture_output=True,env=env).returncode==0
    assert subprocess.run(command+['--resume','--k','1'],capture_output=True,env=env).returncode!=0
    # Simulate a stopped parent after the first completed child.
    report.pop('complete')
    report['workers']=report['workers'][:1]
    report['results']=[r for r in report['results'] if r['repetition']==0]
    output.write_text(json.dumps(report))
    result=subprocess.run(command+['--resume'],capture_output=True,text=True,env=env)
    assert result.returncode==0,result.stderr
    resumed=json.loads(output.read_text())
    assert resumed['complete'] and len(resumed['workers'])==2
    assert len(resumed['results'])==10
