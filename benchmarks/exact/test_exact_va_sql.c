/* Run with SQLITE_CORE and the experimental index enabled, under sanitizers. */
#include "../../sqlite-vec.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void exec(sqlite3 *db, const char *sql) {
  char *error = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &error);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "%s: %s\n", sql, error);
    sqlite3_free(error);
    abort();
  }
}
static void compare(sqlite3 *db, const float *q) {
  sqlite3_stmt *a = NULL, *b = NULL;
  const char *sa =
      "SELECT rowid,distance FROM a WHERE e MATCH ? AND k=10 ORDER BY distance";
  const char *sb =
      "SELECT rowid,distance FROM b WHERE e MATCH ? AND k=10 ORDER BY distance";
  assert(sqlite3_prepare_v2(db, sa, -1, &a, NULL) == SQLITE_OK);
  assert(sqlite3_prepare_v2(db, sb, -1, &b, NULL) == SQLITE_OK);
  sqlite3_bind_blob(a, 1, q, 33 * sizeof(float), SQLITE_STATIC);
  sqlite3_bind_blob(b, 1, q, 33 * sizeof(float), SQLITE_STATIC);
  for (;;) {
    int ar = sqlite3_step(a), br = sqlite3_step(b);
    assert(ar == br);
    if (ar == SQLITE_DONE)
      break;
    assert(ar == SQLITE_ROW);
    assert(sqlite3_column_int64(a, 0) == sqlite3_column_int64(b, 0));
    assert(sqlite3_column_double(a, 1) == sqlite3_column_double(b, 1));
  }
  sqlite3_finalize(a);
  sqlite3_finalize(b);
}
int main(void) {
  sqlite3 *db = NULL;
  assert(sqlite3_open(":memory:", &db) == SQLITE_OK);
  assert(sqlite3_vec_init(db, NULL, NULL) == SQLITE_OK);
  exec(db, "CREATE VIRTUAL TABLE a USING vec0(e float[33],chunk_size=8);"
           "CREATE VIRTUAL TABLE b USING vec0(e float[33] indexed by "
           "exact_va(),chunk_size=8);");
  float q[33];
  for (int j = 0; j < 33; j++)
    q[j] = j * .03125f;
  for (int i = 0; i < 200; i++) {
    float x[33];
    for (int j = 0; j < 33; j++)
      x[j] = (i % 17) * .125f + j * .0625f;
    for (int t = 0; t < 2; t++) {
      sqlite3_stmt *stmt = NULL;
      const char *sql = t ? "INSERT INTO b(rowid,e) VALUES (?,?)"
                          : "INSERT INTO a(rowid,e) VALUES (?,?)";
      assert(sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK);
      sqlite3_bind_int(stmt, 1, i + 1);
      sqlite3_bind_blob(stmt, 2, x, sizeof(x), SQLITE_STATIC);
      assert(sqlite3_step(stmt) == SQLITE_DONE);
      sqlite3_finalize(stmt);
    }
  }
  compare(db, q);
  exec(db, "SAVEPOINT trial; DELETE FROM a WHERE rowid<90; DELETE FROM b WHERE "
           "rowid<90;");
  compare(db, q);
  exec(db, "ROLLBACK TO trial; RELEASE trial;");
  compare(db, q);
  exec(db, "ALTER TABLE b RENAME TO renamed; ALTER TABLE renamed RENAME TO b;");
  compare(db, q);
  exec(db, "DROP TABLE a;DROP TABLE b;");
  assert(sqlite3_close(db) == SQLITE_OK);
  puts("exact_va SQLite sanitizer checks passed");
  return 0;
}
