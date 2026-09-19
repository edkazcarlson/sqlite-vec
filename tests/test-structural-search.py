"""Differential tests for sparse reads, BLOB reuse and rowid routing."""
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('dtype', ['float32', 'float16'])
@pytest.mark.parametrize('metric', ['l2', 'l1', 'cosine'])
@pytest.mark.parametrize('dims', [7, 33, 769])
def test_structural_mutations(tmp_path, dtype, metric, dims):
    extension = os.environ.get('VEC_TEST_EXTENSION', 'dist/vec0.so')
    path = tmp_path / 'vectors.db'
    rng = np.random.default_rng(193)
    vectors = rng.normal(size=(193, dims)).astype(dtype)
    # Duplicates deliberately exercise ties within and across chunks.
    vectors[1::13] = vectors[0]
    wrapper = 'vec_f16(?)' if dtype == 'float16' else '?'
    con = sqlite3.connect(path)
    def load(db):
        db.enable_load_extension(True)
        db.load_extension(str(Path(extension).resolve()), entrypoint='sqlite3_vec_init')
        db.enable_load_extension(False)
    load(con)
    con.execute(f'CREATE VIRTUAL TABLE v USING vec0(e {dtype}[{dims}] distance_metric={metric}, category integer, tenant integer partition key, chunk_size=8)')
    con.executemany(f'INSERT INTO v(rowid,e,category,tenant) VALUES (?,{wrapper},?,?)',
                    [(i-70, x.tobytes(), i%9, i%3) for i,x in enumerate(vectors)])
    con.commit()
    def check(db):
        # Scalar production scorer over the live table is an independent access
        # path, including filtering, point fetches and full distance evaluation.
        for predicate in ('category=1', 'category in (1,3,7)', 'rowid in (-69,-69,0,9,30,31,120,900)',
                          'rowid in (900,901)', 'rowid in (-69,0,9,120) and tenant=1',
                          'tenant>=1 and category=3'):
            for q in (vectors[0], vectors[-1]):
                scorer = f'vec_distance_{metric}(e,{wrapper})' if dtype=='float32' else f'vec_distance_{metric}(vec_f16(e),{wrapper})'
                reference = db.execute(f'SELECT rowid,{scorer} AS distance FROM v WHERE {predicate} ORDER BY distance', (q.tobytes(),)).fetchall()
                for k in (1,10,250):
                    actual=db.execute(f'SELECT rowid,distance FROM v WHERE e MATCH {wrapper} AND k=? AND {predicate} ORDER BY distance', (q.tobytes(),k)).fetchall()
                    assert len(actual)==min(k,len(reference))
                    np.testing.assert_allclose([v for _,v in actual], [v for _,v in reference[:k]], rtol=2e-5,atol=2e-5)
                    expected=dict(reference)
                    assert len({i for i,_ in actual})==len(actual)
                    for i,dist in actual:
                        assert dist==pytest.approx(expected[i],rel=2e-5,abs=2e-5)
    check(con)
    con.execute('SAVEPOINT trial')
    con.execute('DELETE FROM v WHERE rowid%4=0')
    con.execute(f'UPDATE v SET e={wrapper} WHERE rowid=31', (vectors[0].tobytes(),))
    check(con)
    con.execute('ROLLBACK TO trial')
    con.execute('RELEASE trial')
    check(con)
    con.close()
    con=sqlite3.connect(path)
    load(con)
    check(con)
    con.close()


def test_structural_ties_and_reopen(tmp_path):
    extension = str(Path(os.environ.get('VEC_TEST_EXTENSION', 'dist/vec0.so')).resolve())
    path = tmp_path / 'ties.db'
    def connect():
        db=sqlite3.connect(path)
        db.enable_load_extension(True)
        db.load_extension(extension, entrypoint='sqlite3_vec_init')
        db.enable_load_extension(False)
        return db
    db=connect()
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE VIRTUAL TABLE v USING vec0(e float[33], category integer, chunk_size=8)')
    zero=np.zeros(33,dtype='float32').tobytes()
    db.executemany('INSERT INTO v(rowid,e,category) VALUES (?,?,?)',[(100-i,zero,i%2) for i in range(25)])
    db.commit()
    other=connect()
    def check():
        expected=db.execute('SELECT rowid FROM v_rowids ORDER BY chunk_id,chunk_offset DESC LIMIT 10').fetchall()
        assert db.execute('SELECT rowid FROM v WHERE e MATCH ? AND k=10 ORDER BY distance',(zero,)).fetchall()==expected
    check()
    other.execute('DELETE FROM v WHERE rowid=100')
    other.execute('INSERT INTO v(rowid,e,category) VALUES (1000,?,1)',(zero,))
    other.commit()
    check()
    other.close()
    # Even a sparse query must validate the whole chunk's stored length.
    db.execute('UPDATE v_vector_chunks00 SET vectors=substr(vectors,1,length(vectors)-1) WHERE rowid=1')
    with pytest.raises(sqlite3.OperationalError,match="vectors blob size doesn't match"):
        db.execute('SELECT rowid FROM v WHERE e MATCH ? AND k=10 AND category=1',(zero,)).fetchall()
    db.close()
