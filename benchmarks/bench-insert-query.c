/*
 * sqlite-vec benchmark: insert vectors, query with KNN.
 * Tests all distance metrics: L2, cosine, L1, cosine with normalize=unit.
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
#include <math.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

/* Benchmark parameters */
#define BENCH_NUM_VECTORS  100000
#define BENCH_NUM_QUERIES  10000
#define BENCH_DIMENSIONS   768
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
  float norm = 0.0f;
  for (i = 0; i < dims; i++) {
    vec[i] = rand_float() * 2.0f - 1.0f;
    norm += vec[i] * vec[i];
  }
  /* Normalize to simulate CLIP-like embeddings */
  norm = sqrtf(norm);
  if (norm > 0.0f) {
    for (i = 0; i < dims; i++) {
      vec[i] /= norm;
    }
  }
}

static void fill_random_vector_unnormalized(float *vec, int dims) {
  int i;
  for (i = 0; i < dims; i++) {
    vec[i] = rand_float() * 2.0f - 1.0f;
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

/* ---------- Benchmark results for one distance metric ---------- */

struct bench_result {
  const char *metric_name;
  double insert_ms;
  double query_ms;
  int correctness_errors;
};

/* Run the full insert + correctness + query benchmark for one distance metric.
 * metric_name: display name (e.g. "cosine_norm")
 * table_name: unique table name for this metric
 * col_options: column options string (e.g. "distance_metric=cosine normalize=unit")
 * use_unnormalized: if 1, insert unnormalized vectors (to test auto-normalization)
 */
static void run_benchmark(sqlite3 *db, const char *metric_name,
                          const char *table_name, const char *col_options,
                          int use_unnormalized, struct bench_result *result) {
  int rc;
  sqlite3_stmt *stmt;
  double t0, t1;
  float vec[BENCH_DIMENSIONS];
  int i;
  char sql[512];

  result->metric_name = metric_name;
  result->correctness_errors = 0;

  printf("\n--- %s ---\n", metric_name);

  /* Create vec0 table with auxiliary original_idx column */
  snprintf(sql, sizeof(sql),
    "CREATE VIRTUAL TABLE %s USING vec0("
    "+original_idx integer,"
    "embedding float[" STRINGIFY(BENCH_DIMENSIONS) "] %s"
    ")", table_name, col_options);
  rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
  CHECK_OK(rc, db, "CREATE VIRTUAL TABLE");

  /* === INSERT PHASE === */
  xorshift32_state = 12345;
  t0 = get_time_ms();

  rc = sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);
  CHECK_OK(rc, db, "BEGIN");

  snprintf(sql, sizeof(sql),
    "INSERT INTO %s(original_idx, embedding) VALUES (?, ?)", table_name);
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  CHECK_OK(rc, db, "prepare INSERT");

  for (i = 0; i < BENCH_NUM_VECTORS; i++) {
    if (use_unnormalized) {
      fill_random_vector_unnormalized(vec, BENCH_DIMENSIONS);
    } else {
      fill_random_vector(vec, BENCH_DIMENSIONS);
    }
    sqlite3_bind_int(stmt, 1, i + 1);
    sqlite3_bind_blob(stmt, 2, vec, sizeof(float) * BENCH_DIMENSIONS,
                      SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    CHECK_DONE(rc, db, "INSERT step");
    sqlite3_reset(stmt);
  }
  sqlite3_finalize(stmt);

  rc = sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
  CHECK_OK(rc, db, "COMMIT");

  t1 = get_time_ms();
  result->insert_ms = t1 - t0;

  /* === CORRECTNESS CHECKS === */

  /* Check row count */
  {
    snprintf(sql, sizeof(sql), "SELECT count(*) FROM %s", table_name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    CHECK_OK(rc, db, "prepare COUNT");
    rc = sqlite3_step(stmt);
    int count = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    if (count != BENCH_NUM_VECTORS) {
      fprintf(stderr, "FAIL [%s]: row count = %d, expected %d\n",
              metric_name, count, BENCH_NUM_VECTORS);
      result->correctness_errors++;
    } else {
      printf("PASS [%s]: row count = %d\n", metric_name, count);
    }
  }

  /* Check KNN self-lookup: query with rowid=1's vector, expect distance~0
   * and original_idx=1. For normalize=unit with unnormalized input, the query
   * vector is auto-normalized to match the stored unit vector. */
  {
    xorshift32_state = 12345;
    if (use_unnormalized) {
      fill_random_vector_unnormalized(vec, BENCH_DIMENSIONS);
    } else {
      fill_random_vector(vec, BENCH_DIMENSIONS);
    }
    snprintf(sql, sizeof(sql),
      "SELECT rowid, distance, original_idx FROM %s "
      "WHERE embedding MATCH ? AND k = 1 ORDER BY distance", table_name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    CHECK_OK(rc, db, "prepare KNN self-lookup");
    sqlite3_bind_blob(stmt, 1, vec, sizeof(float) * BENCH_DIMENSIONS,
                      SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
      fprintf(stderr, "FAIL [%s]: KNN self-lookup returned no rows\n",
              metric_name);
      result->correctness_errors++;
    } else {
      sqlite3_int64 rowid = sqlite3_column_int64(stmt, 0);
      double dist = sqlite3_column_double(stmt, 1);
      int orig_idx = sqlite3_column_int(stmt, 2);
      if (rowid == 1 && dist < 1e-6 && orig_idx == 1) {
        printf("PASS [%s]: KNN self-lookup rowid=1 original_idx=%d distance=%e\n",
               metric_name, orig_idx, dist);
      } else {
        fprintf(stderr,
                "FAIL [%s]: KNN self-lookup rowid=%lld original_idx=%d distance=%e\n",
                metric_name, rowid, orig_idx, dist);
        result->correctness_errors++;
      }
    }
    sqlite3_finalize(stmt);
  }

  /* Check KNN returns k sorted results */
  {
    xorshift32_state = 99999;
    fill_random_vector(vec, BENCH_DIMENSIONS);
    snprintf(sql, sizeof(sql),
      "SELECT rowid, distance FROM %s "
      "WHERE embedding MATCH ? AND k = " STRINGIFY(BENCH_K)
      " ORDER BY distance", table_name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    CHECK_OK(rc, db, "prepare KNN order check");
    sqlite3_bind_blob(stmt, 1, vec, sizeof(float) * BENCH_DIMENSIONS,
                      SQLITE_TRANSIENT);
    int row_count = 0;
    double prev_dist = -1.0;
    int sorted = 1;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      double dist = sqlite3_column_double(stmt, 1);
      if (dist < prev_dist - 1e-9)
        sorted = 0;
      prev_dist = dist;
      row_count++;
    }
    CHECK_DONE(rc, db, "KNN order check step");
    sqlite3_finalize(stmt);

    if (row_count != BENCH_K) {
      fprintf(stderr, "FAIL [%s]: KNN returned %d rows, expected %d\n",
              metric_name, row_count, BENCH_K);
      result->correctness_errors++;
    } else {
      printf("PASS [%s]: KNN returned %d sorted rows\n",
             metric_name, row_count);
    }
    if (!sorted) {
      fprintf(stderr, "FAIL [%s]: KNN distances not sorted\n", metric_name);
      result->correctness_errors++;
    }
  }

  /* === QUERY PHASE === */
  xorshift32_state = 99999;

  snprintf(sql, sizeof(sql),
    "SELECT rowid, distance FROM %s "
    "WHERE embedding MATCH ? AND k = " STRINGIFY(BENCH_K)
    " ORDER BY distance", table_name);
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
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
  result->query_ms = t1 - t0;
}

/* ---------- Main ---------- */

int main(void) {
  int rc;
  sqlite3 *db;
  sqlite3_stmt *stmt;

  init_timer();

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
  sqlite3_step(stmt);
  printf("sqlite_version=%s, vec_version=%s\n",
         sqlite3_column_text(stmt, 0), sqlite3_column_text(stmt, 1));
  sqlite3_finalize(stmt);

#ifdef SQLITE_VEC_ENABLE_THREADS
#ifndef SQLITE_VEC_KNN_THREADS
#define SQLITE_VEC_KNN_THREADS 4
#endif
  printf("threads:     %d (SQLITE_VEC_KNN_THREADS)\n", SQLITE_VEC_KNN_THREADS);
#else
  printf("threads:     disabled\n");
#endif

  /* Run benchmarks for each distance metric */
  static const char *metric_names[] = {
    "l2", "cosine", "l1", "cosine_norm"
  };
  static const char *table_names[] = {
    "bench_l2", "bench_cosine", "bench_l1", "bench_cosine_norm"
  };
  static const char *col_options[] = {
    "distance_metric=l2",
    "distance_metric=cosine",
    "distance_metric=l1",
    "distance_metric=cosine normalize=unit"
  };
  static const int use_unnormalized[] = {
    0, 0, 0, 1
  };
  #define NUM_METRICS 4

  struct bench_result results[NUM_METRICS];
  int total_errors = 0;
  int m;

  for (m = 0; m < NUM_METRICS; m++) {
    run_benchmark(db, metric_names[m], table_names[m], col_options[m],
                  use_unnormalized[m], &results[m]);
    total_errors += results[m].correctness_errors;
  }

  sqlite3_close(db);

  /* === SUMMARY === */
  printf("\n=== sqlite-vec benchmark ===\n");
  printf("Vectors:     %d x %d dimensions (float32)\n",
         BENCH_NUM_VECTORS, BENCH_DIMENSIONS);
  printf("Queries:     %d (k=%d)\n\n", BENCH_NUM_QUERIES, BENCH_K);

  printf("%-12s %12s %12s %12s %12s\n",
         "Metric", "Insert (ms)", "Query (ms)", "Per query", "Queries/sec");
  printf("%-12s %12s %12s %12s %12s\n",
         "------", "-----------", "----------", "---------", "-----------");

  for (m = 0; m < NUM_METRICS; m++) {
    printf("%-12s %12.2f %12.2f %12.4f %12.0f\n",
           results[m].metric_name,
           results[m].insert_ms,
           results[m].query_ms,
           results[m].query_ms / BENCH_NUM_QUERIES,
           BENCH_NUM_QUERIES / (results[m].query_ms / 1000.0));
  }

  printf("\nPeak memory: %.2f MB\n",
         (double)get_peak_memory_bytes() / (1024.0 * 1024.0));

  if (total_errors > 0) {
    fprintf(stderr, "\n%d correctness error(s)\n", total_errors);
    return 1;
  }
  return 0;
}
