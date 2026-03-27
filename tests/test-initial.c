// Minimal test for initial.c optimizations.
// Tests the distance functions, min_idx (heap), and basic tokenizer/parser.
#include "../sqlite-vec.h"
#include "sqlite-vec-internal.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>

#define countof(x) (sizeof(x) / sizeof((x)[0]))

void test_vec0_token_next() {
  printf("Starting %s...\n", __func__);
  struct Vec0Token token;
  int rc;
  char *input;

  input = "+";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_PLUS);

  input = "hello";
  rc = vec0_token_next(input, input + 5, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);

  input = "42";
  rc = vec0_token_next(input, input + 2, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_DIGIT);

  printf("  PASSED.\n");
}

void test_distance_l2_sqr_float() {
  printf("Starting %s...\n", __func__);
  float a1[] = {1.0f, 2.0f, 3.0f};
  float b1[] = {1.0f, 2.0f, 3.0f};
  float d1 = _test_distance_l2_sqr_float(a1, b1, 3);
  assert(d1 == 0.0f);

  float a2[] = {1.0f, 0.0f};
  float b2[] = {0.0f, 1.0f};
  float d2 = _test_distance_l2_sqr_float(a2, b2, 2);
  // sqrt(1+1) = sqrt(2) ~ 1.4142
  assert(fabsf(d2 - sqrtf(2.0f)) < 1e-5f);

  // Higher dimensions (test unrolled path)
  float a16[16], b16[16];
  for (int i = 0; i < 16; i++) { a16[i] = (float)i; b16[i] = (float)(i + 1); }
  float d16 = _test_distance_l2_sqr_float(a16, b16, 16);
  // Each diff = 1, sum = 16, sqrt(16) = 4
  assert(fabsf(d16 - 4.0f) < 1e-5f);

  // 17 dimensions (tests unroll remainder)
  float a17[17], b17[17];
  for (int i = 0; i < 17; i++) { a17[i] = (float)i; b17[i] = (float)(i + 2); }
  float d17 = _test_distance_l2_sqr_float(a17, b17, 17);
  // Each diff = 2, sum = 17*4=68, sqrt(68)
  assert(fabsf(d17 - sqrtf(68.0f)) < 1e-4f);

  printf("  PASSED.\n");
}

void test_distance_cosine_float() {
  printf("Starting %s...\n", __func__);
  float a1[] = {1.0f, 0.0f};
  float b1[] = {1.0f, 0.0f};
  float d1 = _test_distance_cosine_float(a1, b1, 2);
  assert(fabsf(d1) < 1e-6f); // identical vectors = 0 distance

  float a2[] = {1.0f, 0.0f};
  float b2[] = {0.0f, 1.0f};
  float d2 = _test_distance_cosine_float(a2, b2, 2);
  assert(fabsf(d2 - 1.0f) < 1e-5f); // orthogonal = 1.0

  // Higher dimensions
  float a8[8], b8[8];
  for (int i = 0; i < 8; i++) { a8[i] = (float)(i + 1); b8[i] = (float)(i + 1); }
  float d8 = _test_distance_cosine_float(a8, b8, 8);
  assert(fabsf(d8) < 1e-5f); // identical = 0

  printf("  PASSED.\n");
}

void test_distance_hamming() {
  printf("Starting %s...\n", __func__);
  unsigned char a1[] = {0xFF};
  unsigned char b1[] = {0xFF};
  float d1 = _test_distance_hamming(a1, b1, 8);
  assert(d1 == 0.0f);

  unsigned char a2[] = {0xFF};
  unsigned char b2[] = {0x00};
  float d2 = _test_distance_hamming(a2, b2, 8);
  assert(d2 == 8.0f);

  unsigned char a3[] = {0xAA}; // 10101010
  unsigned char b3[] = {0x55}; // 01010101
  float d3 = _test_distance_hamming(a3, b3, 8);
  assert(d3 == 8.0f);

  printf("  PASSED.\n");
}

// Bitmap helpers we need
uint8_t *bitmap_new(int32_t n);
void bitmap_fill(uint8_t *bitmap, int32_t n);
void bitmap_set(uint8_t *bitmap, int32_t position, int value);
int bitmap_get(uint8_t *bitmap, int32_t position);

void test_min_idx() {
  printf("Starting %s...\n", __func__);

  // Basic: 8 elements, find top 3
  {
    float distances[] = {5.0f, 1.0f, 3.0f, 0.5f, 7.0f, 2.0f, 4.0f, 6.0f};
    int32_t n = 8;
    int32_t k = 3;
    uint8_t *candidates = bitmap_new(n);
    bitmap_fill(candidates, n);
    uint8_t *bTaken = bitmap_new(n);
    int32_t out[3];
    int32_t k_used;
    int rc = min_idx(distances, n, candidates, out, k, bTaken, &k_used);
    assert(rc == 0); // SQLITE_OK
    assert(k_used == 3);
    // Should be indices 3 (0.5), 1 (1.0), 5 (2.0) in sorted order
    assert(out[0] == 3);
    assert(out[1] == 1);
    assert(out[2] == 5);
    sqlite3_free(candidates);
    sqlite3_free(bTaken);
  }

  // With bitmap filtering
  {
    float distances[] = {5.0f, 1.0f, 3.0f, 0.5f, 7.0f, 2.0f, 4.0f, 6.0f};
    int32_t n = 8;
    int32_t k = 3;
    uint8_t *candidates = bitmap_new(n);
    bitmap_fill(candidates, n);
    // Remove indices 1 and 3 (the two smallest)
    bitmap_set(candidates, 1, 0);
    bitmap_set(candidates, 3, 0);
    uint8_t *bTaken = bitmap_new(n);
    int32_t out[3];
    int32_t k_used;
    int rc = min_idx(distances, n, candidates, out, k, bTaken, &k_used);
    assert(rc == 0);
    assert(k_used == 3);
    // Should be indices 5 (2.0), 2 (3.0), 6 (4.0)
    assert(out[0] == 5);
    assert(out[1] == 2);
    assert(out[2] == 6);
    sqlite3_free(candidates);
    sqlite3_free(bTaken);
  }

  // k larger than available candidates
  {
    float distances[] = {5.0f, 1.0f, 3.0f, 0.5f, 7.0f, 2.0f, 4.0f, 6.0f};
    int32_t n = 8;
    int32_t k = 8;
    uint8_t *candidates = bitmap_new(n);
    // Only 3 candidates: indices 0, 2, 4
    bitmap_set(candidates, 0, 1);
    bitmap_set(candidates, 2, 1);
    bitmap_set(candidates, 4, 1);
    uint8_t *bTaken = bitmap_new(n);
    int32_t out[8];
    int32_t k_used;
    int rc = min_idx(distances, n, candidates, out, k, bTaken, &k_used);
    assert(rc == 0);
    assert(k_used == 3);
    // Should be 2 (3.0), 0 (5.0), 4 (7.0)
    assert(out[0] == 2);
    assert(out[1] == 0);
    assert(out[2] == 4);
    sqlite3_free(candidates);
    sqlite3_free(bTaken);
  }

  // Single element
  {
    float distances[] = {42.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int32_t n = 8;
    int32_t k = 1;
    uint8_t *candidates = bitmap_new(n);
    bitmap_set(candidates, 0, 1);
    uint8_t *bTaken = bitmap_new(n);
    int32_t out[1];
    int32_t k_used;
    int rc = min_idx(distances, n, candidates, out, k, bTaken, &k_used);
    assert(rc == 0);
    assert(k_used == 1);
    assert(out[0] == 0);
    sqlite3_free(candidates);
    sqlite3_free(bTaken);
  }

  printf("  PASSED.\n");
}

void test_vec0_parse_vector_column() {
  printf("Starting %s...\n", __func__);
  struct VectorColumnDefinition col;
  int rc;

  rc = vec0_parse_vector_column("embedding float[768]", 20, &col);
  assert(rc == 0); // SQLITE_OK
  assert(col.dimensions == 768);
  assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
  assert(col.distance_metric == VEC0_DISTANCE_METRIC_L2);

  rc = vec0_parse_vector_column("v int8[128] distance_metric=cosine", 35, &col);
  assert(rc == 0);
  assert(col.dimensions == 128);
  assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_INT8);
  assert(col.distance_metric == VEC0_DISTANCE_METRIC_COSINE);

  printf("  PASSED.\n");
}

int main() {
  printf("Starting initial.c unit tests...\n");
#ifdef SQLITE_VEC_ENABLE_AVX
  printf("SQLITE_VEC_ENABLE_AVX=1\n");
#endif
#ifdef SQLITE_VEC_ENABLE_NEON
  printf("SQLITE_VEC_ENABLE_NEON=1\n");
#endif
#if !defined(SQLITE_VEC_ENABLE_AVX) && !defined(SQLITE_VEC_ENABLE_NEON)
  printf("SIMD: none\n");
#endif
  test_vec0_token_next();
  test_distance_l2_sqr_float();
  test_distance_cosine_float();
  test_distance_hamming();
  test_min_idx();
  test_vec0_parse_vector_column();
  printf("All unit tests passed.\n");
  return 0;
}
