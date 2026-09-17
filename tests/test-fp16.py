import json
import os
import sqlite3

import numpy as np
import pytest


@pytest.fixture
def halfdb():
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(os.environ.get("VEC_TEST_EXTENSION", "dist/vec0"), entrypoint="sqlite3_vec_init")
    db.enable_load_extension(False)
    yield db
    db.close()


def half(values):
    return np.asarray(values, dtype=np.float16).tobytes()


def test_all_half_bit_patterns(halfdb):
    bits = np.arange(65536, dtype=np.uint16)
    expected = bits.view(np.float16).astype(np.float32)
    blob = halfdb.execute("select vec_f32(vec_f16(?))", (bits.tobytes(),)).fetchone()[0]
    actual = np.frombuffer(blob, dtype=np.float32)
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))
    assert np.isnan(actual[~finite]).all()
    assert halfdb.execute("select vec_f16(?)", (bits.tobytes(),)).fetchone()[0] == bits.tobytes()


def test_rounding_boundaries(halfdb):
    # Both adjacent half values, their midpoint and float32 neighbors of it.
    values = np.arange(0x7bff, dtype=np.uint16).view(np.float16).astype(np.float32)
    next_values = np.arange(1, 0x7c00, dtype=np.uint16).view(np.float16).astype(np.float32)
    mid = (values + next_values) / 2
    data = np.concatenate([values, next_values, mid,
                           np.nextafter(mid, -np.inf), np.nextafter(mid, np.inf),
                           np.array([65504, 65519, 65520, 1e10, 2**-25, 2**-24, 0], dtype=np.float32)])
    data = np.concatenate([data, -data])
    with np.errstate(over="ignore"):
        expected = data.astype(np.float16)
    result = halfdb.execute("select vec_f16(vec_f32(?))", (data.tobytes(),)).fetchone()[0]
    np.testing.assert_array_equal(np.frombuffer(result, dtype=np.uint16), expected.view(np.uint16))


def test_helpers(halfdb):
    row = halfdb.execute("select vec_type(vec_f16('[1,2,3]')), vec_length(vec_f16('[1,2,3]')), vec_to_json(vec_f16('[1,2,3]'))").fetchone()
    assert row == ("float16", 3, "[1.000000,2.000000,3.000000]")
    assert halfdb.execute("select value from vec_each(vec_f16('[1,2,3]'))").fetchall() == [(1.0,), (2.0,), (3.0,)]
    assert halfdb.execute("select vec_to_json(vec_f16(?))", (half([np.inf, -np.inf, np.nan]),)).fetchone()[0] == "[null,null,null]"
    for sql in ["vec_add(vec_f16('[1]'),vec_f16('[1]'))",
                "vec_sub(vec_f16('[1]'),vec_f16('[1]'))", "vec_slice(vec_f16('[1,2]'),0,1)",
                "vec_normalize(vec_f16('[1]'))", "vec_quantize_binary(vec_f16('[1,1,1,1,1,1,1,1]'))",
                "vec_quantize_int8(vec_f16('[1]'),'unit')", "vec_distance_hamming(vec_f16('[1]'),vec_f16('[1]'))"]:
        with pytest.raises(sqlite3.OperationalError):
            halfdb.execute("select " + sql).fetchall()


@pytest.mark.parametrize("metric", ["l2", "l1", "cosine"])
@pytest.mark.parametrize("dims", [1, 7, 8, 15, 16, 17, 31, 32, 33, 384, 769, 1536, 8192])
def test_distances_and_knn(halfdb, metric, dims):
    rng = np.random.default_rng(17)
    base = rng.normal(size=(41, dims)).astype(np.float16)
    query = rng.normal(size=dims).astype(np.float16)
    a, q = base.astype(np.float64), query.astype(np.float64)
    if metric == "l2":
        ref = np.linalg.norm(a - q, axis=1)
    elif metric == "l1":
        ref = np.abs(a - q).sum(axis=1)
    else:
        ref = 1 - (a @ q) / (np.linalg.norm(a, axis=1) * np.linalg.norm(q))
    halfdb.execute(f"create virtual table v using vec0(e float16[{dims}] distance_metric={metric}, category integer, chunk_size=8)")
    halfdb.executemany("insert into v(rowid,e,category) values (?,vec_f16(?),?)", [(i + 1, row.tobytes(), i % 2) for i, row in enumerate(base)])
    scalar = [halfdb.execute(f"select vec_distance_{metric}(vec_f16(?),vec_f16(?))", (row.tobytes(), query.tobytes())).fetchone()[0] for row in base]
    np.testing.assert_allclose(scalar, ref, atol=2e-5, rtol=2e-5)
    for predicate, allowed in [("", np.arange(len(base))), ("and category=1", np.arange(1, len(base), 2)), ("and rowid in (2,4,6)", np.array([1, 3, 5]))]:
        for k in (1, 10, 100):
            rows = halfdb.execute(f"select rowid,distance,e from v where e match vec_f16(?) and k=? {predicate} order by distance", (query.tobytes(), k)).fetchall()
            assert len(rows) == min(k, len(allowed))
            assert set(r[0] - 1 for r in rows) <= set(allowed)
            np.testing.assert_allclose([r[1] for r in rows], np.sort(ref[allowed])[:k], atol=2e-5, rtol=2e-5)
            for rowid, distance, blob in rows:
                assert blob == base[rowid - 1].tobytes()
                assert distance == pytest.approx(ref[rowid - 1], abs=2e-5, rel=2e-5)
    assert halfdb.execute("select rowid from v where e match vec_f16(?) and k=10 and category=99", (query.tobytes(),)).fetchall() == []
    threshold = float(np.median(ref)) + max(1e-4, abs(float(np.median(ref))) * 1e-4)
    rows = halfdb.execute("select rowid,distance from v where e match vec_f16(?) and k=100 and distance > ? order by distance", (query.tobytes(), threshold)).fetchall()
    assert {r[0] for r in rows} == set(np.flatnonzero(ref > threshold) + 1)


def test_storage_mutations_reopen(tmp_path):
    path = tmp_path / "half.db"
    def connect():
        db = sqlite3.connect(path)
        db.enable_load_extension(True)
        db.load_extension(os.environ.get("VEC_TEST_EXTENSION", "dist/vec0"), entrypoint="sqlite3_vec_init")
        return db
    db = connect()
    db.execute("create virtual table v using vec0(id text primary key, grp integer partition key, e f16[3], other float[3], +note text, chunk_size=8)")
    for i in range(20):
        db.execute("insert into v values (?,?,vec_f16(?),?,?)", (str(i), i % 2, half([i, -i, 0]), json.dumps([i, 1, 2]), "note"))
    db.commit()
    db.execute("update v set e=vec_f16(?) where id='0'", (half([2, 3, 4]),))
    db.rollback()
    assert db.execute("select e from v where id='0'").fetchone()[0] == half([0, 0, 0])
    db.execute("update v set e=vec_f16(?) where id='0'", (half([2, 3, 4]),))
    db.execute("delete from v where id='1'")
    db.execute("insert into v values ('new',1,vec_f16('[5,6,7]'),'[1,2,3]','new')")
    db.commit()
    db.close()
    db = connect()
    assert db.execute("select e from v where id='0'").fetchone()[0] == half([2, 3, 4])
    assert len(db.execute("select e from v").fetchall()) == 20
    assert db.execute("select distinct length(vectors) from v_vector_chunks00").fetchall() == [(8 * 3 * 2,)]
    assert db.execute("select id from v where e match vec_f16('[5,6,7]') and k=1 and grp=1").fetchone()[0] == "new"
    db.close()


def test_errors(halfdb):
    for blob in (b"", b"a", b"abc"):
        with pytest.raises(sqlite3.OperationalError, match="even byte length"):
            halfdb.execute("select vec_f16(?)", (blob,))
    halfdb.execute("create virtual table v using vec0(e f16[2] distance_metric=cosine)")
    for blob in (half([np.nan, 1]), half([np.inf, 1]), half([0, -0.0])):
        for sql in ("insert into v(e) values (vec_f16(?))", "select * from v where e match vec_f16(?) and k=1"):
            with pytest.raises(sqlite3.OperationalError, match="finite|nonzero norm"):
                halfdb.execute(sql, (blob,)).fetchall()
    with pytest.raises(sqlite3.OperationalError, match="type"):
        halfdb.execute("insert into v(e) values ('[1,2]')")
    with pytest.raises(sqlite3.OperationalError, match="Dimension mismatch"):
        halfdb.execute("insert into v(e) values (vec_f16('[1]'))")
    for kind in ("diskann()", "rescore(quantizer=int8)", "ivf(nlist=4)"):
        with pytest.raises(sqlite3.OperationalError):
            halfdb.execute(f"create virtual table bad using vec0(e float16[8] indexed by {kind})")
    with pytest.raises(sqlite3.OperationalError):
        halfdb.execute("create virtual table bad using vec0(e float16garbage[8])")


def test_raw_blob_context_and_update_validation(halfdb):
    halfdb.execute("create virtual table v using vec0(e f16[2] distance_metric=cosine)")
    halfdb.execute("insert into v(rowid,e) values (1,?)", (half([1, 2]),))
    halfdb.commit()
    halfdb.execute("update v set e=? where rowid=1", (half([3, 4]),))
    assert halfdb.execute("select e from v where rowid=1").fetchone()[0] == half([3, 4])
    assert halfdb.execute("select rowid from v where e match ? and k=1", (half([3, 4]),)).fetchall() == [(1,)]
    for blob in (half([np.nan, 1]), half([0, 0])):
        with pytest.raises(sqlite3.OperationalError, match="finite|nonzero norm"):
            halfdb.execute("update v set e=vec_f16(?) where rowid=1", (blob,))
    assert halfdb.execute("select e from v where rowid=1").fetchone()[0] == half([3, 4])
    halfdb.rollback()
    assert halfdb.execute("select e from v where rowid=1").fetchone()[0] == half([1, 2])
