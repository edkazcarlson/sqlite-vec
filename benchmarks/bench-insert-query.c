/*
 * sqlite-vec benchmark: insert 10K vectors, query 1K times.
 * Measures wall clock time and peak memory usage.
 *
 * Compile with MSVC:
 *   cl /nologo /O2 /DSQLITE_CORE bench-insert-query.c sqlite-vec.c
 *       vendor\sqlite3.c /I. /Ivendor /Fe:dist\bench-insert-query.exe
 */

#include "sqlite3.h"
#include "sqlite-vec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

/* Benchmark parameters */
#define BENCH_NUM_VECTORS  100000
#define BENCH_NUM_QUERIES  10000
#define BENCH_DIMENSIONS   512
#define BENCH_K            10

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

/* ---------- PRNG (xorshift32) ---------- */

static unsigned int xorshift32_state = 12345;

static unsigned int xorshift32(void) {
  unsigned int x = xorshift32_state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  xorshift32_state = x;
  return x;
}

static float rand_float(void) {
  return (float)(xorshift32() & 0x7FFFFF) / (float)0x7FFFFF;
}

static void fill_random_vector(float *vec, int dims) {
  int i;
  for (i = 0; i < dims; i++) {
    vec[i] = rand_float();
  }
}

/* ---------- Timer ---------- */

static LARGE_INTEGER perf_freq;

static void init_timer(void) { QueryPerformanceFrequency(&perf_freq); }

static double get_time_ms(void) {
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  return (double)counter.QuadPart / (double)perf_freq.QuadPart * 1000.0;
}

/* ---------- Memory ---------- */

static size_t get_peak_memory_bytes(void) {
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return pmc.PeakWorkingSetSize;
  }
  return 0;
}

/* ---------- Helpers ---------- */

#define CHECK_OK(rc, db, msg)                                                  \
  do {                                                                         \
    if ((rc) != SQLITE_OK) {                                                   \
      fprintf(stderr, "ERROR: %s: %s\n", (msg), sqlite3_errmsg(db));          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_DONE(rc, db, msg)                                                \
  do {                                                                         \
    if ((rc) != SQLITE_DONE) {                                                 \
      fprintf(stderr, "ERROR: %s: %s\n", (msg), sqlite3_errmsg(db));          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/* ---------- Main ---------- */

int main(void) {
  int rc;
  sqlite3 *db;
  sqlite3_stmt *stmt;
  double t0, t1;
  float vec[BENCH_DIMENSIONS];
  int i;

  init_timer();

  /* Register sqlite-vec */
  rc = sqlite3_auto_extension((void (*)(void))sqlite3_vec_init);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "ERROR: sqlite3_auto_extension failed\n");
    return 1;
  }

  rc = sqlite3_open(":memory:", &db);
  CHECK_OK(rc, db, "sqlite3_open");

  /* Print versions */
  rc = sqlite3_prepare_v2(db, "SELECT sqlite_version(), vec_version()", -1,
                          &stmt, NULL);
  CHECK_OK(rc, db, "version query");
  rc = sqlite3_step(stmt);
  printf("sqlite_version=%s, vec_version=%s\n", sqlite3_column_text(stmt, 0),
         sqlite3_column_text(stmt, 1));
  sqlite3_finalize(stmt);

  /* Create vec0 table */
  rc = sqlite3_exec(
      db,
      "CREATE VIRTUAL TABLE bench USING vec0(embedding float[" STRINGIFY(
          BENCH_DIMENSIONS) "])",
      NULL, NULL, NULL);
  CHECK_OK(rc, db, "CREATE VIRTUAL TABLE");

  /* === INSERT PHASE === */
  xorshift32_state = 12345;

  t0 = get_time_ms();

  rc = sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);
  CHECK_OK(rc, db, "BEGIN");

  rc = sqlite3_prepare_v2(db, "INSERT INTO bench(embedding) VALUES (?)", -1,
                          &stmt, NULL);
  CHECK_OK(rc, db, "prepare INSERT");

  for (i = 0; i < BENCH_NUM_VECTORS; i++) {
    fill_random_vector(vec, BENCH_DIMENSIONS);
    sqlite3_bind_blob(stmt, 1, vec, sizeof(float) * BENCH_DIMENSIONS,
                      SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    CHECK_DONE(rc, db, "INSERT step");
    sqlite3_reset(stmt);
  }
  sqlite3_finalize(stmt);

  rc = sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
  CHECK_OK(rc, db, "COMMIT");

  t1 = get_time_ms();
  double insert_time_ms = t1 - t0;

  /* === CORRECTNESS CHECKS === */
  int errors = 0;
  double t_start, t_end;
  double check_rowcount_ms, check_pointlookup_ms, check_knn_self_ms,
      check_knn_order_ms;

  /* Check row count */
  {
    t_start = get_time_ms();
    rc = sqlite3_prepare_v2(db, "SELECT count(*) FROM bench", -1, &stmt, NULL);
    CHECK_OK(rc, db, "prepare COUNT");
    rc = sqlite3_step(stmt);
    int count = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    t_end = get_time_ms();
    check_rowcount_ms = t_end - t_start;
    if (count != BENCH_NUM_VECTORS) {
      fprintf(stderr, "FAIL: row count = %d, expected %d\n", count,
              BENCH_NUM_VECTORS);
      errors++;
    } else {
      printf("PASS: row count = %d (%.2f ms)\n", count, check_rowcount_ms);
    }
  }

  /* Check point lookup: re-generate vector for rowid 1 and retrieve it */
  {
    t_start = get_time_ms();
    xorshift32_state = 12345; /* same seed as insert phase */
    fill_random_vector(vec, BENCH_DIMENSIONS);
    rc = sqlite3_prepare_v2(db, "SELECT embedding FROM bench WHERE rowid = 1",
                            -1, &stmt, NULL);
    CHECK_OK(rc, db, "prepare point lookup");
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
      fprintf(stderr, "FAIL: rowid 1 not found\n");
      errors++;
    } else {
      const float *stored =
          (const float *)sqlite3_column_blob(stmt, 0);
      int blob_bytes = sqlite3_column_bytes(stmt, 0);
      if (blob_bytes != (int)(sizeof(float) * BENCH_DIMENSIONS)) {
        fprintf(stderr, "FAIL: rowid 1 blob size = %d, expected %d\n",
                blob_bytes, (int)(sizeof(float) * BENCH_DIMENSIONS));
        errors++;
      } else {
        int match = 1;
        for (i = 0; i < BENCH_DIMENSIONS; i++) {
          if (stored[i] != vec[i]) {
            match = 0;
            break;
          }
        }
        if (match) {
          printf("PASS: rowid 1 vector matches inserted data\n");
        } else {
          fprintf(stderr, "FAIL: rowid 1 vector mismatch at dim %d "
                          "(got %f, expected %f)\n",
                  i, stored[i], vec[i]);
          errors++;
        }
      }
    }
    sqlite3_finalize(stmt);
    t_end = get_time_ms();
    check_pointlookup_ms = t_end - t_start;
    printf("      point lookup time: %.2f ms\n", check_pointlookup_ms);
  }

  /* Check KNN: query with a stored vector, expect distance ~0 as top result */
  {
    t_start = get_time_ms();
    xorshift32_state = 12345;
    fill_random_vector(vec, BENCH_DIMENSIONS); /* vector for rowid 1 */
    rc = sqlite3_prepare_v2(
        db,
        "SELECT rowid, distance FROM bench "
        "WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
        -1, &stmt, NULL);
    CHECK_OK(rc, db, "prepare KNN self-lookup");
    sqlite3_bind_blob(stmt, 1, vec, sizeof(float) * BENCH_DIMENSIONS,
                      SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
      fprintf(stderr, "FAIL: KNN self-lookup returned no rows\n");
      errors++;
    } else {
      sqlite3_int64 rowid = sqlite3_column_int64(stmt, 0);
      double dist = sqlite3_column_double(stmt, 1);
      if (rowid == 1 && dist < 1e-6) {
        printf("PASS: KNN self-lookup returned rowid=1 distance=%e\n", dist);
      } else {
        fprintf(stderr,
                "FAIL: KNN self-lookup returned rowid=%lld distance=%e "
                "(expected rowid=1 distance~0)\n",
                rowid, dist);
        errors++;
      }
    }
    sqlite3_finalize(stmt);
    t_end = get_time_ms();
    check_knn_self_ms = t_end - t_start;
    printf("      KNN self-lookup time: %.2f ms\n", check_knn_self_ms);
  }

  /* Check KNN returns k results and distances are sorted ascending */
  {
    t_start = get_time_ms();
    xorshift32_state = 99999;
    fill_random_vector(vec, BENCH_DIMENSIONS);
    rc = sqlite3_prepare_v2(
        db,
        "SELECT rowid, distance FROM bench "
        "WHERE embedding MATCH ? AND k = " STRINGIFY(BENCH_K)
        " ORDER BY distance",
        -1, &stmt, NULL);
    CHECK_OK(rc, db, "prepare KNN order check");
    sqlite3_bind_blob(stmt, 1, vec, sizeof(float) * BENCH_DIMENSIONS,
                      SQLITE_TRANSIENT);
    int row_count = 0;
    double prev_dist = -1.0;
    int sorted = 1;
    int all_nonneg = 1;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      double dist = sqlite3_column_double(stmt, 1);
      if (dist < 0.0)
        all_nonneg = 0;
      if (dist < prev_dist - 1e-9)
        sorted = 0;
      prev_dist = dist;
      row_count++;
    }
    CHECK_DONE(rc, db, "KNN order check step");
    sqlite3_finalize(stmt);
    t_end = get_time_ms();
    check_knn_order_ms = t_end - t_start;

    if (row_count != BENCH_K) {
      fprintf(stderr, "FAIL: KNN returned %d rows, expected %d\n", row_count,
              BENCH_K);
      errors++;
    } else {
      printf("PASS: KNN returned %d rows\n", row_count);
    }
    if (!sorted) {
      fprintf(stderr, "FAIL: KNN distances not in ascending order\n");
      errors++;
    } else {
      printf("PASS: KNN distances in ascending order\n");
    }
    if (!all_nonneg) {
      fprintf(stderr, "FAIL: KNN returned negative distances\n");
      errors++;
    } else {
      printf("PASS: all KNN distances >= 0\n");
    }
    printf("      KNN order check time: %.2f ms\n", check_knn_order_ms);
  }

  if (errors > 0) {
    fprintf(stderr, "\nCORRECTNESS: %d check(s) FAILED\n", errors);
  } else {
    printf("\nCORRECTNESS: all checks passed\n");
  }

  /* === QUERY PHASE === */
  xorshift32_state = 99999; /* different seed for queries */

  rc = sqlite3_prepare_v2(
      db,
      "SELECT rowid, distance FROM bench "
      "WHERE embedding MATCH ? AND k = " STRINGIFY(BENCH_K) " ORDER BY distance",
      -1, &stmt, NULL);
  CHECK_OK(rc, db, "prepare SELECT");

  t0 = get_time_ms();

  int total_results = 0;
  for (i = 0; i < BENCH_NUM_QUERIES; i++) {
    fill_random_vector(vec, BENCH_DIMENSIONS);
    sqlite3_bind_blob(stmt, 1, vec, sizeof(float) * BENCH_DIMENSIONS,
                      SQLITE_TRANSIENT);
    int row_count = 0;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      (void)sqlite3_column_int64(stmt, 0);
      (void)sqlite3_column_double(stmt, 1);
      row_count++;
    }
    CHECK_DONE(rc, db, "SELECT step");
    total_results += row_count;
    sqlite3_reset(stmt);
  }
  sqlite3_finalize(stmt);

  t1 = get_time_ms();
  double query_time_ms = t1 - t0;

  /* === RESULTS === */
  size_t peak_mem = get_peak_memory_bytes();

  printf("\n=== sqlite-vec benchmark results ===\n");
  printf("Vectors:     %d x %d dimensions (float32)\n", BENCH_NUM_VECTORS,
         BENCH_DIMENSIONS);
  printf("Queries:     %d (k=%d)\n", BENCH_NUM_QUERIES, BENCH_K);
  printf("\n");
  printf("Insert time:          %10.2f ms (%.2f ms/vector, %.0f vectors/sec)\n",
         insert_time_ms, insert_time_ms / BENCH_NUM_VECTORS,
         BENCH_NUM_VECTORS / (insert_time_ms / 1000.0));
  printf("Row count check:      %10.2f ms\n", check_rowcount_ms);
  printf("Point lookup check:   %10.2f ms\n", check_pointlookup_ms);
  printf("KNN self-lookup check:%10.2f ms\n", check_knn_self_ms);
  printf("KNN order check:      %10.2f ms\n", check_knn_order_ms);
  printf("Query time:           %10.2f ms (%.2f ms/query, %.0f queries/sec)\n",
         query_time_ms, query_time_ms / BENCH_NUM_QUERIES,
         BENCH_NUM_QUERIES / (query_time_ms / 1000.0));
  printf("Total results returned: %d (expected %d)\n", total_results,
         BENCH_NUM_QUERIES * BENCH_K);
  printf("Peak memory: %.2f MB\n",
         (double)peak_mem / (1024.0 * 1024.0));

  sqlite3_close(db);
  return errors > 0 ? 1 : 0;
}
