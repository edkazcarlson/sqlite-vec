#include "../sqlite-vec.h"
#include "sqlite-vec-internal.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define countof(x) (sizeof(x) / sizeof((x)[0]))

// Transpose an AoS (row-major) array into SoA (column-major) layout.
// aos[i * dims + d] -> soa[d * n + i]
static void transpose_aos_to_soa(const float *aos, float *soa, int n,
                                  int dims) {
  for (int i = 0; i < n; i++)
    for (int d = 0; d < dims; d++)
      soa[d * n + i] = aos[i * dims + d];
}

// Tests vec0_token_next(), the low-level tokenizer that extracts the next
// token from a raw char range. Covers every token type (identifier, digit,
// brackets, plus, equals), whitespace skipping, EOF on empty/whitespace-only
// input, error on unrecognised characters, and boundary behaviour where
// identifiers and digits stop at the next non-matching character.
void test_vec0_token_next() {
  printf("Starting %s...\n", __func__);
  struct Vec0Token token;
  int rc;
  char *input;

  // Single-character tokens
  input = "+";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_PLUS);

  input = "[";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_LBRACKET);

  input = "]";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_RBRACKET);

  input = "=";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_EQ);

  // Identifier
  input = "hello";
  rc = vec0_token_next(input, input + 5, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
  assert(token.start == input);
  assert(token.end == input + 5);

  // Identifier with underscores and digits
  input = "col_1a";
  rc = vec0_token_next(input, input + 6, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
  assert(token.end - token.start == 6);

  // Digit sequence
  input = "1234";
  rc = vec0_token_next(input, input + 4, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_DIGIT);
  assert(token.start == input);
  assert(token.end == input + 4);

  // Leading whitespace is skipped
  input = "  abc";
  rc = vec0_token_next(input, input + 5, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
  assert(token.end - token.start == 3);

  // Tab/newline whitespace
  input = "\t\n\r X";
  rc = vec0_token_next(input, input + 5, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);

  // Empty input
  input = "";
  rc = vec0_token_next(input, input, &token);
  assert(rc == VEC0_TOKEN_RESULT_EOF);

  // Only whitespace
  input = "   ";
  rc = vec0_token_next(input, input + 3, &token);
  assert(rc == VEC0_TOKEN_RESULT_EOF);

  // Unrecognized character
  input = "@";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_ERROR);

  input = "!";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_ERROR);

  // Identifier stops at bracket
  input = "foo[";
  rc = vec0_token_next(input, input + 4, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
  assert(token.end - token.start == 3);

  // Digit stops at non-digit
  input = "42abc";
  rc = vec0_token_next(input, input + 5, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_DIGIT);
  assert(token.end - token.start == 2);

  // Left paren
  input = "(";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_LPAREN);

  // Right paren
  input = ")";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_RPAREN);

  // Comma
  input = ",";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_COMMA);

  printf("  All vec0_token_next tests passed.\n");
}

// Tests Vec0Scanner, the stateful wrapper around vec0_token_next() that
// tracks position and yields successive tokens. Verifies correct tokenisation
// of full sequences like "abc float[128]" and "key=value", empty input,
// whitespace-heavy input, and expressions with operators ("a+b").
void test_vec0_scanner() {
  printf("Starting %s...\n", __func__);
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  int rc;

  // Scan "abc float[128]"
  {
    const char *input = "abc float[128]";
    vec0_scanner_init(&scanner, input, (int)strlen(input));

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(token.end - token.start == 3);
    assert(strncmp(token.start, "abc", 3) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(token.end - token.start == 5);
    assert(strncmp(token.start, "float", 5) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_LBRACKET);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_DIGIT);
    assert(strncmp(token.start, "128", 3) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_RBRACKET);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  // Scan "key=value"
  {
    const char *input = "key=value";
    vec0_scanner_init(&scanner, input, (int)strlen(input));

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "key", 3) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_EQ);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "value", 5) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  // Scan empty string
  {
    const char *input = "";
    vec0_scanner_init(&scanner, input, 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  // Scan with lots of whitespace
  {
    const char *input = "  a   b  ";
    vec0_scanner_init(&scanner, input, (int)strlen(input));

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(token.end - token.start == 1);
    assert(*token.start == 'a');

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(token.end - token.start == 1);
    assert(*token.start == 'b');

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  // Scan "a+b"
  {
    const char *input = "a+b";
    vec0_scanner_init(&scanner, input, (int)strlen(input));

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_PLUS);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  // Scan "diskann(k=v, k2=v2)"
  {
    const char *input = "diskann(k=v, k2=v2)";
    vec0_scanner_init(&scanner, input, (int)strlen(input));

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "diskann", 7) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_LPAREN);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "k", 1) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_EQ);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "v", 1) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_COMMA);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "k2", 2) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_EQ);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "v2", 2) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_RPAREN);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  printf("  All vec0_scanner tests passed.\n");
}

// Tests vec0_parse_vector_column(), which parses a vec0 column definition
// string like "embedding float[768] distance_metric=cosine" into a
// VectorColumnDefinition struct. Covers all element types (float/f32, int8/i8,
// bit), column names with underscores/digits, all distance metrics (L2, L1,
// cosine), the default metric, and error cases: empty input, missing type,
// unknown type, missing dimensions, unknown metric, unknown option key, and
// distance_metric on bit columns.
void test_vec0_parse_vector_column() {
  printf("Starting %s...\n", __func__);
  struct VectorColumnDefinition col;
  int rc;

  // Basic float column
  {
    const char *input = "embedding float[768]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.name_length == 9);
    assert(strncmp(col.name, "embedding", 9) == 0);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    assert(col.dimensions == 768);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_L2);
    sqlite3_free(col.name);
  }

  // f32 alias
  {
    const char *input = "v f32[3]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    assert(col.dimensions == 3);
    sqlite3_free(col.name);
  }

  // int8 column
  {
    const char *input = "quantized int8[256]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_INT8);
    assert(col.dimensions == 256);
    assert(col.name_length == 9);
    assert(strncmp(col.name, "quantized", 9) == 0);
    sqlite3_free(col.name);
  }

  // i8 alias
  {
    const char *input = "q i8[64]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_INT8);
    assert(col.dimensions == 64);
    sqlite3_free(col.name);
  }

  // bit column
  {
    const char *input = "bvec bit[1024]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_BIT);
    assert(col.dimensions == 1024);
    sqlite3_free(col.name);
  }

  // Column name with underscores and digits
  {
    const char *input = "col_name_2 float[10]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.name_length == 10);
    assert(strncmp(col.name, "col_name_2", 10) == 0);
    sqlite3_free(col.name);
  }

  // distance_metric=cosine
  {
    const char *input = "emb float[128] distance_metric=cosine";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_COSINE);
    assert(col.dimensions == 128);
    sqlite3_free(col.name);
  }

  // distance_metric=L2 (explicit)
  {
    const char *input = "emb float[128] distance_metric=L2";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_L2);
    sqlite3_free(col.name);
  }

  // distance_metric=L1
  {
    const char *input = "emb float[128] distance_metric=l1";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_L1);
    sqlite3_free(col.name);
  }

  // SQLITE_EMPTY: empty string
  {
    const char *input = "";
    rc = vec0_parse_vector_column(input, 0, &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: non-vector column (text primary key)
  {
    const char *input = "document_id text primary key";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: non-vector column (partition key)
  {
    const char *input = "user_id integer partition key";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: no type (single identifier)
  {
    const char *input = "emb";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: unknown type
  {
    const char *input = "emb double[128]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: unknown type (unknowntype)
  {
    const char *input = "v unknowntype[128]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: missing brackets entirely
  {
    const char *input = "emb float";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // Error: zero dimensions
  {
    const char *input = "v float[0]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: empty brackets (no dimensions)
  {
    const char *input = "v float[]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: unknown distance metric
  {
    const char *input = "emb float[128] distance_metric=hamming";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: unknown distance metric (foo)
  {
    const char *input = "v float[128] distance_metric=foo";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: unknown option key
  {
    const char *input = "emb float[128] foobar=baz";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: distance_metric on bit type
  {
    const char *input = "emb bit[64] distance_metric=cosine";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // normalize=unit with float column
  {
    const char *input = "emb float[128] normalize=unit";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.normalize == 1);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_L2);
    assert(col.dimensions == 128);
    sqlite3_free(col.name);
  }

  // normalize=unit with distance_metric=cosine (both options)
  {
    const char *input = "emb float[128] distance_metric=cosine normalize=unit";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.normalize == 1);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_COSINE);
    sqlite3_free(col.name);
  }

  // normalize=unit with options in reverse order
  {
    const char *input = "emb float[128] normalize=unit distance_metric=cosine";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.normalize == 1);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_COSINE);
    sqlite3_free(col.name);
  }

  // Default: normalize=0 when not specified
  {
    const char *input = "emb float[128]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.normalize == 0);
    sqlite3_free(col.name);
  }

  // Error: normalize=unit on int8 column
  {
    const char *input = "emb int8[128] normalize=unit";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: normalize=unit on bit column
  {
    const char *input = "emb bit[64] normalize=unit";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: unknown normalize value
  {
    const char *input = "emb float[128] normalize=l2";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  printf("  All vec0_parse_vector_column tests passed.\n");
}

// Tests vec0_parse_partition_key_definition(), which parses a vec0 partition
// key column definition like "user_id integer partition key". Verifies correct
// parsing of integer and text partition keys, column name extraction, and
// rejection of invalid inputs: empty strings, non-partition-key definitions
// ("primary key"), and misspelled keywords.
void test_vec0_parse_partition_key_definition() {
  printf("Starting %s...\n", __func__);
  typedef struct {
    char * test;
    int expected_rc;
    const char *expected_column_name;
    int expected_column_type;
  } TestCase;

  TestCase suite[] = {
    {"user_id integer partition key", SQLITE_OK, "user_id", SQLITE_INTEGER},
    {"USER_id int partition key", SQLITE_OK, "USER_id", SQLITE_INTEGER},
    {"category text partition key", SQLITE_OK, "category", SQLITE_TEXT},

    {"", SQLITE_EMPTY, "", 0},
    {"document_id text primary key", SQLITE_EMPTY, "", 0},
    {"document_id text partition keyy", SQLITE_EMPTY, "", 0},
  };
  for(int i = 0; i < countof(suite); i++) {
    char * out_column_name;
    int out_column_name_length;
    int out_column_type;
    int rc;
    rc = vec0_parse_partition_key_definition(
      suite[i].test,
      strlen(suite[i].test),
      &out_column_name,
      &out_column_name_length,
      &out_column_type
    );
    assert(rc == suite[i].expected_rc);

    if(rc == SQLITE_OK) {
      assert(out_column_name_length == strlen(suite[i].expected_column_name));
      assert(strncmp(out_column_name, suite[i].expected_column_name, out_column_name_length) == 0);
      assert(out_column_type == suite[i].expected_column_type);
    }

    printf("  Passed: \"%s\"\n", suite[i].test);
  }
}

void test_distance_l2_sqr_float() {
  printf("Starting %s...\n", __func__);
  float d;

  // Identical vectors: distance = 0
  {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    d = _test_distance_l2_sqr_float(a, b, 3);
    assert(d == 0.0f);
  }

  // Orthogonal unit vectors: sqrt(1+1) = sqrt(2)
  {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    d = _test_distance_l2_sqr_float(a, b, 3);
    assert(fabsf(d - sqrtf(2.0f)) < 1e-6f);
  }

  // Known computation: [1,2,3] vs [4,5,6] = sqrt(9+9+9) = sqrt(27)
  {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    d = _test_distance_l2_sqr_float(a, b, 3);
    assert(fabsf(d - sqrtf(27.0f)) < 1e-5f);
  }

  // Single dimension: sqrt(16) = 4.0
  {
    float a[] = {3.0f};
    float b[] = {7.0f};
    d = _test_distance_l2_sqr_float(a, b, 1);
    assert(d == 4.0f);
  }

  printf("  All distance_l2_sqr_float tests passed.\n");
}

void test_distance_cosine_float() {
  printf("Starting %s...\n", __func__);
  float d;

  // Identical direction: distance = 0.0
  {
    float a[] = {1.0f, 0.0f};
    float b[] = {2.0f, 0.0f};
    d = _test_distance_cosine_float(a, b, 2);
    assert(fabsf(d - 0.0f) < 1e-6f);
  }

  // Orthogonal: distance = 1.0
  {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    d = _test_distance_cosine_float(a, b, 2);
    assert(fabsf(d - 1.0f) < 1e-6f);
  }

  // Opposite direction: distance = 2.0
  {
    float a[] = {1.0f, 0.0f};
    float b[] = {-1.0f, 0.0f};
    d = _test_distance_cosine_float(a, b, 2);
    assert(fabsf(d - 2.0f) < 1e-6f);
  }

  printf("  All distance_cosine_float tests passed.\n");
}

void test_distance_hamming() {
  printf("Starting %s...\n", __func__);
  float d;

  // Identical bitmaps: distance = 0
  {
    unsigned char a[] = {0xFF};
    unsigned char b[] = {0xFF};
    d = _test_distance_hamming(a, b, 8);
    assert(d == 0.0f);
  }

  // All different: distance = 8
  {
    unsigned char a[] = {0xFF};
    unsigned char b[] = {0x00};
    d = _test_distance_hamming(a, b, 8);
    assert(d == 8.0f);
  }

  // Half different: 0xFF vs 0x0F = 4 bits differ
  {
    unsigned char a[] = {0xFF};
    unsigned char b[] = {0x0F};
    d = _test_distance_hamming(a, b, 8);
    assert(d == 4.0f);
  }

  // Multi-byte: [0xFF, 0x00] vs [0x00, 0xFF] = 16 bits differ
  {
    unsigned char a[] = {0xFF, 0x00};
    unsigned char b[] = {0x00, 0xFF};
    d = _test_distance_hamming(a, b, 16);
    assert(d == 16.0f);
  }

  printf("  All distance_hamming tests passed.\n");
}

// TODO: Re-enable once batch/chunk distance functions are implemented in sqlite-vec.c
#if 0
void test_batch_distance_l2_sqr_float() {
  printf("Starting %s...\n", __func__);

  // Batch L2 now returns squared distances (no sqrt).
  // Compare against squared reference values.

  // 4 vectors of 3 dimensions
  {
    float base[] = {
        1.0f, 2.0f, 3.0f, // v0
        4.0f, 5.0f, 6.0f, // v1
        0.0f, 0.0f, 0.0f, // v2
        1.0f, 0.0f, 0.0f, // v3
    };
    float soa_base[4 * 3];
    transpose_aos_to_soa(base, soa_base, 4, 3);
    float query[] = {1.0f, 2.0f, 3.0f};
    float batch_dist[4];
    _test_batch_distance_l2_sqr_float(soa_base, query, batch_dist, 4, 3);

    // v0 == query => 0
    assert(batch_dist[0] == 0.0f);
    // v1: (4-1)^2+(5-2)^2+(6-3)^2 = 27
    assert(fabsf(batch_dist[1] - 27.0f) < 1e-5f);
    // v2: 1+4+9 = 14
    assert(fabsf(batch_dist[2] - 14.0f) < 1e-5f);
    // v3: 0+4+9 = 13
    assert(fabsf(batch_dist[3] - 13.0f) < 1e-5f);
  }

  // Single vector
  {
    float base[] = {3.0f, 0.0f};
    float soa_base[1 * 2];
    transpose_aos_to_soa(base, soa_base, 1, 2);
    float query[] = {0.0f, 4.0f};
    float dist[1];
    _test_batch_distance_l2_sqr_float(soa_base, query, dist, 1, 2);
    // 9+16 = 25 (squared distance)
    assert(fabsf(dist[0] - 25.0f) < 1e-5f);
  }

  // 9 vectors (tests tile boundary: 8 + 1 tail for AVX, 4 + 4 + 1 for NEON)
  {
    float base[9 * 2];
    float soa_base[9 * 2];
    float query[] = {0.0f, 0.0f};
    float batch_dist[9];
    for (int i = 0; i < 9; i++) {
      base[i * 2 + 0] = (float)(i + 1);
      base[i * 2 + 1] = 0.0f;
    }
    transpose_aos_to_soa(base, soa_base, 9, 2);
    _test_batch_distance_l2_sqr_float(soa_base, query, batch_dist, 9, 2);
    for (int i = 0; i < 9; i++) {
      float expected = (float)((i + 1) * (i + 1)); // squared
      assert(fabsf(batch_dist[i] - expected) < 1e-5f);
    }
  }

  // 16 vectors of 16 dimensions
  {
    float base[16 * 16];
    float soa_base[16 * 16];
    float query[16];
    float batch_dist[16];
    for (int d = 0; d < 16; d++) {
      query[d] = (float)d;
    }
    for (int i = 0; i < 16; i++) {
      for (int d = 0; d < 16; d++) {
        base[i * 16 + d] = (float)(i + d);
      }
    }
    transpose_aos_to_soa(base, soa_base, 16, 16);
    _test_batch_distance_l2_sqr_float(soa_base, query, batch_dist, 16, 16);
    for (int i = 0; i < 16; i++) {
      float ref = _test_distance_l2_sqr_float(base + i * 16, query, 16);
      // ref is sqrt-ed, batch is squared; compare batch vs ref^2
      assert(fabsf(batch_dist[i] - ref * ref) < 1e-2f);
    }
  }

  // 17 vectors of 32 dimensions (tests both tile and tail with larger dims)
  {
    float base[17 * 32];
    float soa_base[17 * 32];
    float query[32];
    float batch_dist[17];
    for (int d = 0; d < 32; d++) {
      query[d] = (float)d * 0.1f;
    }
    for (int i = 0; i < 17; i++) {
      for (int d = 0; d < 32; d++) {
        base[i * 32 + d] = (float)(i * 3 + d) * 0.5f;
      }
    }
    transpose_aos_to_soa(base, soa_base, 17, 32);
    _test_batch_distance_l2_sqr_float(soa_base, query, batch_dist, 17, 32);
    for (int i = 0; i < 17; i++) {
      float ref = _test_distance_l2_sqr_float(base + i * 32, query, 32);
      assert(fabsf(batch_dist[i] - ref * ref) < 1e-1f);
    }
  }

  // 33 vectors of 32 dims: four full 8-wide SoA tiles + 1 scalar remainder
  {
    float base[33 * 32];
    float soa_base[33 * 32];
    float query[32];
    float batch_dist[33];
    for (int d = 0; d < 32; d++) {
      query[d] = (float)d * 0.3f;
    }
    for (int i = 0; i < 33; i++) {
      for (int d = 0; d < 32; d++) {
        base[i * 32 + d] = (float)(i * 7 + d) * 0.2f;
      }
    }
    transpose_aos_to_soa(base, soa_base, 33, 32);
    _test_batch_distance_l2_sqr_float(soa_base, query, batch_dist, 33, 32);
    for (int i = 0; i < 33; i++) {
      float ref = _test_distance_l2_sqr_float(base + i * 32, query, 32);
      assert(fabsf(batch_dist[i] - ref * ref) < 1e-1f);
    }
  }

  // 64 vectors of 16 dims: eight full 8-wide tiles, no remainder
  {
    float base[64 * 16];
    float soa_base[64 * 16];
    float query[16];
    float batch_dist[64];
    for (int d = 0; d < 16; d++) {
      query[d] = (float)(d + 1);
    }
    for (int i = 0; i < 64; i++) {
      for (int d = 0; d < 16; d++) {
        base[i * 16 + d] = (float)(i * 2 + d) * 0.1f;
      }
    }
    transpose_aos_to_soa(base, soa_base, 64, 16);
    _test_batch_distance_l2_sqr_float(soa_base, query, batch_dist, 64, 16);
    for (int i = 0; i < 64; i++) {
      float ref = _test_distance_l2_sqr_float(base + i * 16, query, 16);
      assert(fabsf(batch_dist[i] - ref * ref) < 1e-1f);
    }
  }

  // 100 vectors of 32 dims: twelve 8-wide tiles (96) + 4 in scalar tail
  {
    float base[100 * 32];
    float soa_base[100 * 32];
    float query[32];
    float batch_dist[100];
    for (int d = 0; d < 32; d++) {
      query[d] = (float)d * -0.5f;
    }
    for (int i = 0; i < 100; i++) {
      for (int d = 0; d < 32; d++) {
        base[i * 32 + d] = (float)(i + d * 3) * 0.4f;
      }
    }
    transpose_aos_to_soa(base, soa_base, 100, 32);
    _test_batch_distance_l2_sqr_float(soa_base, query, batch_dist, 100, 32);
    for (int i = 0; i < 100; i++) {
      float ref = _test_distance_l2_sqr_float(base + i * 32, query, 32);
      assert(fabsf(batch_dist[i] - ref * ref) < 1e-1f);
    }
  }

  printf("  All batch_distance_l2_sqr_float tests passed.\n");
}

void test_batch_distance_cosine_float() {
  printf("Starting %s...\n", __func__);

  // 4 vectors of 16 dimensions — compare batch against per-vector reference
  {
    float base[4 * 16];
    float soa_base[4 * 16];
    float query[16];
    float batch_dist[4];
    for (int d = 0; d < 16; d++) {
      query[d] = (float)(d + 1);
    }
    for (int i = 0; i < 4; i++) {
      for (int d = 0; d < 16; d++) {
        base[i * 16 + d] = (float)(i * 3 + d + 1) * 0.5f;
      }
    }
    transpose_aos_to_soa(base, soa_base, 4, 16);
    _test_batch_distance_cosine_float(soa_base, query, batch_dist, 4, 16);
    for (int i = 0; i < 4; i++) {
      float ref = _test_distance_cosine_float(base + i * 16, query, 16);
      assert(fabsf(batch_dist[i] - ref) < 1e-5f);
    }
  }

  // 17 vectors of 32 dimensions — tests tile + tail
  {
    float base[17 * 32];
    float soa_base[17 * 32];
    float query[32];
    float batch_dist[17];
    for (int d = 0; d < 32; d++) {
      query[d] = (float)d * 0.2f;
    }
    for (int i = 0; i < 17; i++) {
      for (int d = 0; d < 32; d++) {
        base[i * 32 + d] = (float)(i * 5 + d) * 0.3f;
      }
    }
    transpose_aos_to_soa(base, soa_base, 17, 32);
    _test_batch_distance_cosine_float(soa_base, query, batch_dist, 17, 32);
    for (int i = 0; i < 17; i++) {
      float ref = _test_distance_cosine_float(base + i * 32, query, 32);
      assert(fabsf(batch_dist[i] - ref) < 1e-4f);
    }
  }

  printf("  All batch_distance_cosine_float tests passed.\n");
}

void test_batch_distance_l1_float() {
  printf("Starting %s...\n", __func__);

  // 4 vectors of 16 dimensions
  {
    float base[4 * 16];
    float soa_base[4 * 16];
    float query[16];
    float batch_dist[4];
    for (int d = 0; d < 16; d++) {
      query[d] = (float)(d + 1);
    }
    for (int i = 0; i < 4; i++) {
      for (int d = 0; d < 16; d++) {
        base[i * 16 + d] = (float)(i * 2 + d) * 0.4f;
      }
    }
    transpose_aos_to_soa(base, soa_base, 4, 16);
    _test_batch_distance_l1_float(soa_base, query, batch_dist, 4, 16);
    for (int i = 0; i < 4; i++) {
      // Compute scalar L1 reference
      float ref = 0.0f;
      for (int d = 0; d < 16; d++) {
        ref += fabsf(query[d] - base[i * 16 + d]);
      }
      assert(fabsf(batch_dist[i] - ref) < 1e-4f);
    }
  }

  // 17 vectors of 32 dimensions — tests tile + tail
  {
    float base[17 * 32];
    float soa_base[17 * 32];
    float query[32];
    float batch_dist[17];
    for (int d = 0; d < 32; d++) {
      query[d] = (float)d * -0.3f;
    }
    for (int i = 0; i < 17; i++) {
      for (int d = 0; d < 32; d++) {
        base[i * 32 + d] = (float)(i + d * 2) * 0.5f;
      }
    }
    transpose_aos_to_soa(base, soa_base, 17, 32);
    _test_batch_distance_l1_float(soa_base, query, batch_dist, 17, 32);
    for (int i = 0; i < 17; i++) {
      float ref = 0.0f;
      for (int d = 0; d < 32; d++) {
        ref += fabsf(query[d] - base[i * 32 + d]);
      }
      assert(fabsf(batch_dist[i] - ref) < 1e-2f);
    }
  }

  printf("  All batch_distance_l1_float tests passed.\n");
}

// Test vec0_compute_chunk_distances against per-element distance functions.
// Verifies the extracted helper produces identical results to the inline code.
void test_compute_chunk_distances() {
  printf("Starting %s...\n", __func__);

  // 8 vectors of 4 dimensions (chunk_size must be multiple of 8 for bitmaps)
  const int n = 8;
  const int dims = 4;
  float base[8 * 4] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 0.0f, 0.0f,
    0.5f, 0.5f, 0.5f, 0.5f,
    2.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
  };
  float query[4] = {1.0f, 0.0f, 0.0f, 0.0f};

  // All candidates valid
  unsigned char bitmap[1] = {0xFF}; // 8 bits, all set
  float distances[8];
  memset(distances, 0, sizeof(distances));

  // Test L2 metric
  _test_compute_chunk_distances_float(base, query, n, dims,
                                      VEC0_DISTANCE_METRIC_L2,
                                      bitmap, distances);

  // Verify against reference
  for (int i = 0; i < n; i++) {
    float ref = _test_distance_l2_sqr_float(base + i * dims, query, dims);
    assert(fabsf(distances[i] - ref) < 1e-6f);
  }

  // Test with partial bitmap (only vectors 0, 2, 5 valid)
  unsigned char partial_bitmap[1] = {0x25}; // bits 0,2,5 set = 0b00100101
  memset(distances, 0, sizeof(distances));
  _test_compute_chunk_distances_float(base, query, n, dims,
                                      VEC0_DISTANCE_METRIC_L2,
                                      partial_bitmap, distances);

  // Valid entries should have correct distances, others should remain 0
  float ref0 = _test_distance_l2_sqr_float(base + 0 * dims, query, dims);
  float ref2 = _test_distance_l2_sqr_float(base + 2 * dims, query, dims);
  float ref5 = _test_distance_l2_sqr_float(base + 5 * dims, query, dims);
  assert(fabsf(distances[0] - ref0) < 1e-6f);
  assert(fabsf(distances[1] - 0.0f) < 1e-6f); // not in bitmap
  assert(fabsf(distances[2] - ref2) < 1e-6f);
  assert(fabsf(distances[3] - 0.0f) < 1e-6f);
  assert(fabsf(distances[4] - 0.0f) < 1e-6f);
  assert(fabsf(distances[5] - ref5) < 1e-6f);

  // Test cosine metric (skip zero vector at index 7 — cosine undefined)
  memset(distances, 0, sizeof(distances));
  bitmap[0] = 0x7F; // bits 0-6 set, bit 7 clear (skip zero vector)
  _test_compute_chunk_distances_float(base, query, n, dims,
                                      VEC0_DISTANCE_METRIC_COSINE,
                                      bitmap, distances);
  for (int i = 0; i < 7; i++) {
    float ref = _test_distance_cosine_float(base + i * dims, query, dims);
    assert(fabsf(distances[i] - ref) < 1e-5f);
  }

  printf("  All compute_chunk_distances tests passed.\n");
}
#endif // #if 0 for batch/chunk tests

#ifdef SQLITE_VEC_ENABLE_THREADS
// Integration test: run a KNN query through the threaded path and verify
// correctness by checking results are sorted and self-lookup works.
void test_threaded_knn_integration() {
  printf("Starting %s...\n", __func__);

  sqlite3 *db;
  sqlite3_stmt *stmt;
  int rc;

  rc = sqlite3_auto_extension((void (*)(void))sqlite3_vec_init);
  assert(rc == SQLITE_OK);

  rc = sqlite3_open(":memory:", &db);
  assert(rc == SQLITE_OK);

  // Create vec0 table
  rc = sqlite3_exec(db,
      "CREATE VIRTUAL TABLE test_t USING vec0(embedding float[4])",
      NULL, NULL, NULL);
  assert(rc == SQLITE_OK);

  // Insert 100 vectors (enough to create multiple chunks if chunk_size is small)
  rc = sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);
  assert(rc == SQLITE_OK);

  rc = sqlite3_prepare_v2(db, "INSERT INTO test_t(embedding) VALUES (?)", -1, &stmt, NULL);
  assert(rc == SQLITE_OK);

  for (int i = 0; i < 100; i++) {
    float vec[4] = {(float)i, (float)(i * 2), (float)(i * 3), (float)(i * 4)};
    sqlite3_bind_blob(stmt, 1, vec, sizeof(vec), SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    assert(rc == SQLITE_DONE);
    sqlite3_reset(stmt);
  }
  sqlite3_finalize(stmt);

  rc = sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
  assert(rc == SQLITE_OK);

  // KNN query: find 5 nearest to vector[0]
  float query[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  rc = sqlite3_prepare_v2(db,
      "SELECT rowid, distance FROM test_t "
      "WHERE embedding MATCH ? AND k = 5 ORDER BY distance",
      -1, &stmt, NULL);
  assert(rc == SQLITE_OK);
  sqlite3_bind_blob(stmt, 1, query, sizeof(query), SQLITE_TRANSIENT);

  int row_count = 0;
  double prev_dist = -1.0;
  sqlite3_int64 first_rowid = -1;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    sqlite3_int64 rowid = sqlite3_column_int64(stmt, 0);
    double dist = sqlite3_column_double(stmt, 1);
    if (row_count == 0) {
      first_rowid = rowid;
      // First result should be rowid=1 (vector [0,0,0,0]) with distance ~0
      assert(rowid == 1);
      assert(dist < 1e-6);
    }
    // Results must be sorted by distance
    assert(dist >= prev_dist - 1e-9);
    prev_dist = dist;
    row_count++;
  }
  assert(rc == SQLITE_DONE);
  assert(row_count == 5);
  sqlite3_finalize(stmt);

  sqlite3_close(db);
  printf("  Threaded KNN integration test passed.\n");
}
#endif /* SQLITE_VEC_ENABLE_THREADS */

int main() {
  printf("Starting unit tests...\n");
#ifdef SQLITE_VEC_ENABLE_AVX
  printf("SQLITE_VEC_ENABLE_AVX=1\n");
#endif
#ifdef SQLITE_VEC_ENABLE_NEON
  printf("SQLITE_VEC_ENABLE_NEON=1\n");
#endif
#ifdef SQLITE_VEC_ENABLE_THREADS
  printf("SQLITE_VEC_ENABLE_THREADS=1\n");
#endif
#if !defined(SQLITE_VEC_ENABLE_AVX) && !defined(SQLITE_VEC_ENABLE_NEON)
  printf("SIMD: none\n");
#endif
  test_vec0_token_next();
  test_vec0_scanner();
  test_vec0_parse_vector_column();
  test_vec0_parse_partition_key_definition();
  test_distance_l2_sqr_float();
  test_distance_cosine_float();
  test_distance_hamming();
  // TODO: Re-enable once batch/chunk distance functions are implemented
  // test_batch_distance_l2_sqr_float();
  // test_batch_distance_cosine_float();
  // test_batch_distance_l1_float();
  // test_compute_chunk_distances();
#ifdef SQLITE_VEC_ENABLE_THREADS
  test_threaded_knn_integration();
#endif
  printf("All unit tests passed.\n");
}
