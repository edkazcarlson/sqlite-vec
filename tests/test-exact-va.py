import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def indexed_db(tmp_path):
    extension=str(Path(os.environ.get('VEC_TEST_EXTENSION','dist/vec0.so')).resolve())
    path=tmp_path/'exact.db'
    def connect():
        db=sqlite3.connect(path)
        db.enable_load_extension(True)
        db.load_extension(extension,entrypoint='sqlite3_vec_init')
        db.enable_load_extension(False)
        return db
    db=connect()
    if 'exact_va=1' not in db.execute('select vec_debug()').fetchone()[0]:
        db.close()
        if os.environ.get('VEC_TEST_EXTENSION'):
            pytest.fail('Requested extension does not report exact_va=1')
        pytest.skip('experimental exact_va disabled; build with SQLITE_VEC_EXPERIMENTAL_EXACT_VA=1')
    yield db,connect
    db.close()


@pytest.mark.parametrize('dtype',['float32','float16'])
@pytest.mark.parametrize('dims',[1,7,8,17,33,769])
def test_exact_va_lifecycle(indexed_db,dtype,dims):
    db,connect=indexed_db
    rng=np.random.default_rng(153)
    vectors=rng.normal(size=(137,dims)).astype(dtype)
    vectors[::11]=vectors[0]
    vectors[1]=np.nextafter(vectors[0],np.array(np.inf,dtype=dtype))
    vectors[2]=0
    wrap='vec_f16(?)' if dtype=='float16' else '?'
    for table,index in [('a',''),('b','indexed by exact_va()')]:
        db.execute(f'create virtual table {table} using vec0(e {dtype}[{dims}] {index}, category integer, tenant integer partition key, chunk_size=8)')
        db.executemany(f'insert into {table}(rowid,e,category,tenant) values(?,{wrap},?,?)',[(i-30,x.tobytes(),i%5,i%3) for i,x in enumerate(vectors)])
    db.commit()
    shadows = {row[1]: row[2] for row in db.execute('pragma table_list')}
    assert shadows['b_exactvachunks00'] == 'shadow'
    assert shadows['b_exactvavectors00'] == 'shadow'
    def check(connection):
        for q in (vectors[0],vectors[-1],vectors[2]):
            for predicate in ('','and category=1','and tenant>=1','and rowid in (-30,0,1,5,9,100,999)',
                              'and category=99','and distance>0.5','and distance<=20'):
                for k in (1,10,200):
                    statement=f'select rowid,distance from {{}} where e match {wrap} and k=? {predicate} order by distance'
                    a=connection.execute(statement.format('a'),(q.tobytes(),k)).fetchall()
                    b=connection.execute(statement.format('b'),(q.tobytes(),k)).fetchall()
                    assert a==b
        assert connection.execute('select rowid,e from a order by rowid').fetchall()==connection.execute('select rowid,e from b order by rowid').fetchall()
    check(db)
    db.execute('savepoint changes')
    for table in ('a','b'):
        db.execute(f'update {table} set e={wrap} where rowid=5',(vectors[-1].tobytes(),))
        db.execute(f'delete from {table} where rowid<10')
        db.execute(f'insert into {table}(rowid,e,category,tenant) values(500,{wrap},1,1)',(vectors[0].tobytes(),))
    check(db)
    db.execute('rollback to changes');db.execute('release changes')
    check(db)
    second=connect();check(second);second.close()
    db.execute('alter table b rename to renamed')
    assert db.execute(f'select rowid,distance from renamed where e match {wrap} and k=10 order by distance',(vectors[0].tobytes(),)).fetchall()==db.execute(f'select rowid,distance from a where e match {wrap} and k=10 order by distance',(vectors[0].tobytes(),)).fetchall()
    db.execute('drop table renamed')
    assert not db.execute("select name from sqlite_master where name like 'renamed_exactva%'").fetchall()


def test_exact_va_extremes_and_corruption(indexed_db):
    db,_=indexed_db
    x=np.array([[0,0],[np.finfo('float32').max,0],[-np.finfo('float32').max,0],
                [np.finfo('float32').tiny,0],[1e-40,0],[1,1]],dtype='float32')
    for table,index in [('a',''),('b','indexed by exact_va()')]:
        db.execute(f'create virtual table {table} using vec0(e float[2] {index},chunk_size=8)')
        db.executemany(f'insert into {table}(rowid,e) values(?,?)',[(i+1,v.tobytes()) for i,v in enumerate(x)])
    for q in x:
        for k in (1,3,6):
            sql='select rowid,distance from {} where e match ? and k=? order by distance'
            assert db.execute(sql.format('a'),(q.tobytes(),k)).fetchall()==db.execute(sql.format('b'),(q.tobytes(),k)).fetchall()
    db.execute('update b_exactvachunks00 set vectors=substr(vectors,1,length(vectors)-1)')
    with pytest.raises(sqlite3.DatabaseError):
        db.execute('select rowid from b where e match ? and k=1',(x[0].tobytes(),)).fetchall()


@pytest.mark.parametrize('column',['e int8[8] indexed by exact_va()',
 'e bit[8] indexed by exact_va()','e float[8] distance_metric=cosine indexed by exact_va()',
 'e float[8] indexed by exact_va() distance_metric=l1'])
def test_exact_va_rejects_unsupported(indexed_db,column):
    db,_=indexed_db
    with pytest.raises(sqlite3.OperationalError):
        db.execute(f'create virtual table v using vec0({column})')


def test_exact_va_attached_text_keys_and_failures(indexed_db):
    db,_=indexed_db
    db.execute("attach ':memory:' as extra")
    name='extra."quoted table"'
    db.execute(f'create virtual table {name} using vec0(id text primary key,e float[3] indexed by exact_va(),+note text)')
    db.execute(f"insert into {name}(id,e,note) values('one','[1,2,3]','first'),('two','[2,3,4]','second')")
    db.execute(f"insert or replace into {name}(id,e,note) values('one','[3,4,5]','replacement')")
    assert db.execute(f"select id,note from {name} where e match '[3,4,5]' and k=1").fetchone()==('one','replacement')
    before=db.execute(f'select id,e,note from {name} order by id').fetchall()
    with pytest.raises(sqlite3.OperationalError):
        db.execute(f"update {name} set e='[1,2]'")
    assert db.execute(f'select id,e,note from {name} order by id').fetchall()==before
    db.execute('savepoint empty')
    db.execute(f'delete from {name}')
    assert db.execute('select count(*) from extra."quoted table_exactvachunks00"').fetchone()==(0,)
    assert db.execute('select count(*) from extra."quoted table_exactvavectors00"').fetchone()==(0,)
    db.execute('rollback to empty');db.execute('release empty')
    assert db.execute(f"select id from {name} where e match '[3,4,5]' and k=1").fetchone()==('one',)


def test_exact_va_multiple_columns_and_max_dimensions(indexed_db):
    db, _ = indexed_db
    rng = np.random.default_rng(711)
    vectors = rng.normal(size=(24, 8192)).astype('float32')
    halves = vectors[:, :17].astype('float16')
    for table, index in [('a', ''), ('b', 'indexed by exact_va()')]:
        db.execute(f'create virtual table {table} using vec0('
                   f'x float[8192] {index}, y float16[17] {index}, chunk_size=8)')
        db.executemany(f'insert into {table}(rowid,x,y) values(?,?,vec_f16(?))',
                       [(i + 1, x.tobytes(), y.tobytes())
                        for i, (x, y) in enumerate(zip(vectors, halves))])
    def check():
        for column, query, wrap in [('x', vectors[0], '?'),
                                    ('y', halves[0], 'vec_f16(?)')]:
            sql = (f'select rowid,distance from {{}} where {column} match {wrap}'
                   ' and k=7 order by distance')
            assert (db.execute(sql.format('a'), (query.tobytes(),)).fetchall() ==
                    db.execute(sql.format('b'), (query.tobytes(),)).fetchall())
    check()
    for table in ('a', 'b'):
        db.execute(f'update {table} set x=?, y=vec_f16(?) where rowid=20',
                   (vectors[0].tobytes(), halves[0].tobytes()))
        db.execute(f'delete from {table} where rowid<=8')
    check()
    with pytest.raises(sqlite3.OperationalError):
        db.execute('create virtual table unsupported using vec0('
                   'x float[8193] indexed by exact_va())')
