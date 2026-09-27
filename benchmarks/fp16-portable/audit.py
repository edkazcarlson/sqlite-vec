"""Audit completed measurements, row limits, source integrity and build identity."""
import itertools
import json
from pathlib import Path

import run


def audit():
    checks=0
    for rows in (0,-1,2001,100000):
        for call in (lambda:run.dataset(rows,768),lambda:run.sql_worker({'rows':rows,'dims':768})):
            try:call()
            except ValueError:checks+=1
            else:raise AssertionError(f'Row cap failed for {rows}')
    assert run.read('sql-complete.json')['complete']
    screen=run.read('screen.json');conv=run.read('conversion.json');sql=run.read('sql.json')
    assert len(screen)==len(run.VARIANTS)*2*4*4*2*2
    assert len(conv)==9*4*4*2
    assert len(sql)==600
    keys=[json.dumps(r['case'],sort_keys=True) for r in sql]
    assert len(set(keys))==len(keys)
    assert all(1<=r['case']['rows']<=2000 and len(r['samples_ms'])==20 for r in sql)
    assert all(len(r['samples_ms'])==3 for r in screen+conv)
    env=run.read('environment.json')
    source_ok={p:run.sha(run.ROOT/p)==digest for p,digest in env['sources'].items() if p.startswith('sqlite-vec')}
    assert all(source_ok.values()),'Production sources changed'
    builds=run.read('builds-sql.json')+run.read('builds-micro.json')
    binary_ok={b['label']:run.sha(run.BUILD/b['label']/'vec0.dll')==b['binary_sha256'] for b in builds}
    assert all(binary_ok.values()),'Native binaries changed'
    flags=[b['command'][1:b['command'].index('/Qvec-report:2')+1] for b in run.read('builds-sql.json')]
    assert all(f==flags[0] for f in flags),'SQL compiler settings differ'
    result=dict(status='passed',row_limit_rejection_checks=checks,micro_cases=len(screen),conversion_cases=len(conv),sql_cases=len(sql),timed_sql_queries=sum(len(r['samples_ms']) for r in sql),maximum_rows=max(r['case']['rows'] for r in sql),production_source_hashes_match=source_ok,binary_hashes_match=binary_ok,identical_sql_flags=flags[0],final_harness_hashes={p.name:run.sha(p) for p in run.HERE.glob('*.py')})
    run.save('audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':audit()
