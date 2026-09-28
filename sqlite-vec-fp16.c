/* IEEE binary16 storage. Included by sqlite-vec.c; no external dependencies. */
static f32 vec_half_to_float(uint16_t h) {
  u32 sign = ((u32)h & 0x8000) << 16;
  u32 exponent = (h >> 10) & 31;
  u32 mantissa = h & 1023;
  u32 bits;
  if (exponent == 0) {
    if (!mantissa) bits = sign;
    else {
      int e = -14;
      while (!(mantissa & 1024)) { mantissa <<= 1; e--; }
      bits = sign | ((u32)(e + 127) << 23) | ((mantissa & 1023) << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  f32 result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static uint16_t vec_float_to_half(f32 value) {
  u32 bits;
  memcpy(&bits, &value, sizeof(bits));
  u32 sign = (bits >> 16) & 0x8000;
  u32 exponent = (bits >> 23) & 255;
  u32 mantissa = bits & 0x7fffff;
  if (exponent == 255)
    return (uint16_t)(sign | 0x7c00 | (mantissa ? ((mantissa >> 13) | 0x200) : 0));
  int e = (int)exponent - 127;
  if (e > 15) return (uint16_t)(sign | 0x7c00);
  if (e < -25) return (uint16_t)sign;
  if (e < -14) {
    mantissa |= 0x800000;
    unsigned shift = (unsigned)(-e - 1);
    u32 half = mantissa >> shift;
    u32 remainder = mantissa & (((u32)1 << shift) - 1);
    u32 midpoint = (u32)1 << (shift - 1);
    half += remainder > midpoint || (remainder == midpoint && (half & 1));
    return (uint16_t)(sign | half);
  }
  u32 half = ((u32)(e + 15) << 10) | (mantissa >> 13);
  u32 remainder = mantissa & 8191;
  half += remainder > 4096 || (remainder == 4096 && (half & 1));
  return (uint16_t)(sign | half);
}

/* memcpy also permits SQLite blobs that are not uint16_t-aligned. */
static f32 vec_half_at(const void *data, size_t i) {
  uint16_t h;
  memcpy(&h, (const u8 *)data + i * 2, 2);
  return vec_half_to_float(h);
}

/* Search inputs have passed vec_half_validate. The finite fast path avoids
 * exceptional-value handling and the general decoder's subnormal shift loop.
 * Subnormal half values become normal float32 values, so FTZ/DAZ are harmless.
 * Keep vec_half_to_float general for public conversions, including NaN/Inf. */
static inline f32 vec_half_to_float_finite(uint16_t h) {
  u32 bits;
  if (h & 0x7c00) {
    bits = (((u32)h & 0x7fff) << 13) + 0x38000000u;
  } else {
    bits = 0x3f000000u | (h & 1023);
    f32 small;
    memcpy(&small, &bits, sizeof(small));
    small -= 0.5f;
    memcpy(&bits, &small, sizeof(bits));
    /* Exact zero subtraction can be -0 under round-toward-negative. */
    bits &= 0x7fffffffu;
  }
  bits |= ((u32)h & 0x8000) << 16;
  f32 result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static inline f32 vec_half_at_finite(const void *data, size_t i) {
  uint16_t h;
  memcpy(&h, (const u8 *)data + i * 2, 2);
  return vec_half_to_float_finite(h);
}

static void vec_half_expand(const void *data, f32 *out, size_t n) {
  for (size_t i = 0; i < n; i++) out[i] = vec_half_at(data, i);
}

/* F16C is optional even in AVX builds. Keep the baseline translation unit
 * unchanged; only guarded functions below may execute these instructions. */
#if (defined(SQLITE_VEC_ENABLE_AVX) || defined(SQLITE_VEC_ENABLE_F16C)) && \
    (defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)) && \
    (defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__))
#define VEC_HAVE_F16C 1
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#if defined(__clang__) || defined(__GNUC__)
#define VEC_F16C_TARGET __attribute__((target("avx,f16c"), noinline))
#else
#define VEC_F16C_TARGET __declspec(noinline)
#endif

static int vec_half_f16c_usable(u32 features, u64 xcr0) {
  const u32 required = (1u << 27) | (1u << 28) | (1u << 29);
  return (features & required) == required && (xcr0 & 6) == 6;
}

static int vec_half_has_f16c(void) {
#if defined(SQLITE_VEC_TEST) && defined(SQLITE_VEC_TEST_NO_F16C)
  return 0;
#else
  u32 features;
  u64 xcr0;
#ifdef _MSC_VER
  int cpu[4];
  __cpuid(cpu, 1);
  features = (u32)cpu[2];
#else
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
  features = c;
#endif
  if (!vec_half_f16c_usable(features, 6)) return 0;
#ifdef _MSC_VER
  xcr0 = _xgetbv(0);
#else
  u32 lo, hi;
  __asm__ volatile ("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  xcr0 = ((u64)hi << 32) | lo;
#endif
  return vec_half_f16c_usable(features, xcr0);
#endif
}

VEC_F16C_TARGET
static f32 l2_sqr_float16_f16c(const void *a, const f32 *q, size_t n, f32 query_norm) {
  UNUSED_PARAMETER(query_norm);
  __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps()};
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    for (int lane = 0; lane < 4; lane++) {
      __m128i h = _mm_loadu_si128((const __m128i *)((const u8 *)a + (i + lane * 8) * 2));
      __m256 x = _mm256_cvtph_ps(h);
      __m256 y = _mm256_loadu_ps(q + i + lane * 8);
      __m256 diff = _mm256_sub_ps(x, y);
      sums[lane] = _mm256_add_ps(sums[lane], _mm256_mul_ps(diff, diff));
    }
  }
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)((const u8 *)a + i * 2)));
    __m256 y = _mm256_loadu_ps(q + i);
    __m256 diff = _mm256_sub_ps(x, y);
    sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(diff, diff));
  }
  f32 s[8];
  _mm256_storeu_ps(s, _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]),
                                  _mm256_add_ps(sums[2], sums[3])));
  f32 sum = 0;
  for (int j = 0; j < 8; j++) sum += s[j];
  for (; i < n; i++) {
    f32 diff = vec_half_at_finite(a, i) - q[i];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

VEC_F16C_TARGET
static f32 l1_float16_f16c(const void *a, const f32 *q, size_t n, f32 query_norm) {
  UNUSED_PARAMETER(query_norm);
  __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps()};
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    for (int lane = 0; lane < 4; lane++) {
      __m128i h = _mm_loadu_si128((const __m128i *)((const u8 *)a + (i + lane * 8) * 2));
      __m256 x = _mm256_cvtph_ps(h);
      __m256 y = _mm256_loadu_ps(q + i + lane * 8);
      __m256 diff = _mm256_sub_ps(x, y);
      sums[lane] = _mm256_add_ps(sums[lane], _mm256_andnot_ps(sign_mask, diff));
    }
  }
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)((const u8 *)a + i * 2)));
    __m256 y = _mm256_loadu_ps(q + i);
    __m256 diff = _mm256_sub_ps(x, y);
    sums[0] = _mm256_add_ps(sums[0], _mm256_andnot_ps(sign_mask, diff));
  }
  f32 s[8];
  _mm256_storeu_ps(s, _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]),
                                  _mm256_add_ps(sums[2], sums[3])));
  f32 sum = 0;
  for (int j = 0; j < 8; j++) sum += s[j];
  for (; i < n; i++) {
    f32 diff = vec_half_at_finite(a, i) - q[i];
    sum += fabsf(diff);
  }
  return sum;
}

VEC_F16C_TARGET
static f32 cosine_float16_f16c(const void *a, const f32 *q, size_t n, f32 query_norm) {
  __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps()};
  __m256 norms[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps(), _mm256_setzero_ps()};
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    for (int lane = 0; lane < 4; lane++) {
      __m128i h = _mm_loadu_si128((const __m128i *)((const u8 *)a + (i + lane * 8) * 2));
      __m256 x = _mm256_cvtph_ps(h);
      __m256 y = _mm256_loadu_ps(q + i + lane * 8);
      sums[lane] = _mm256_add_ps(sums[lane], _mm256_mul_ps(x, y));
      norms[lane] = _mm256_add_ps(norms[lane], _mm256_mul_ps(x, x));
    }
  }
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)((const u8 *)a + i * 2)));
    __m256 y = _mm256_loadu_ps(q + i);
    sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(x, y));
    norms[0] = _mm256_add_ps(norms[0], _mm256_mul_ps(x, x));
  }
  f32 s[8];
  _mm256_storeu_ps(s, _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]),
                                  _mm256_add_ps(sums[2], sums[3])));
  f32 sum = 0;
  for (int j = 0; j < 8; j++) sum += s[j];
  f32 m[8];
  _mm256_storeu_ps(m, _mm256_add_ps(_mm256_add_ps(norms[0], norms[1]),
                                  _mm256_add_ps(norms[2], norms[3])));
  f32 norm = 0;
  for (int j = 0; j < 8; j++) norm += m[j];
  for (; i < n; i++) {
    f32 x = vec_half_at_finite(a, i);
    sum += x * q[i];
    norm += x * x;
  }
  return 1 - sum / (sqrtf(norm) * query_norm);
}

#endif

/* The query has already been widened once by the caller. Like the existing
 * distance_l2_sqr_float/int8 functions, L2 returns the Euclidean distance. */
static f32 l2_sqr_float16(const void *a, const f32 *q, size_t n, f32 query_norm) {
  UNUSED_PARAMETER(query_norm);
  f32 sum = 0;
  for (size_t i = 0; i < n; i++) {
    f32 diff = vec_half_at_finite(a, i) - q[i];
    sum += diff * diff;
  }
  return sqrtf(sum);
}


static f32 l1_float16(const void *a, const f32 *q, size_t n, f32 query_norm) {
  UNUSED_PARAMETER(query_norm);
  f32 sum = 0;
  for (size_t i = 0; i < n; i++) {
    f32 diff = vec_half_at_finite(a, i) - q[i];
    sum += fabsf(diff);
  }
  return sum;
}


static f32 cosine_float16(const void *a, const f32 *q, size_t n, f32 query_norm) {
  f32 sum = 0;
  f32 norm = 0;
  for (size_t i = 0; i < n; i++) {
    f32 x = vec_half_at_finite(a, i);
    sum += x * q[i];
    norm += x * x;
  }
  return 1 - sum / (sqrtf(norm) * query_norm);
}


typedef f32 (*vec_half_distance_fn)(const void *, const f32 *, size_t, f32);

static vec_half_distance_fn vec_half_distance_select(int metric) {
#ifdef VEC_HAVE_F16C
  if (vec_half_has_f16c()) {
    switch (metric) {
    case VEC0_DISTANCE_METRIC_L2: return l2_sqr_float16_f16c;
    case VEC0_DISTANCE_METRIC_L1: return l1_float16_f16c;
    case VEC0_DISTANCE_METRIC_COSINE: return cosine_float16_f16c;
    }
  }
#endif
  switch (metric) {
  case VEC0_DISTANCE_METRIC_L2: return l2_sqr_float16;
  case VEC0_DISTANCE_METRIC_L1: return l1_float16;
  default: return cosine_float16;
  }
}

static f32 vec_query_norm(const f32 *q, size_t n) {
  f32 sum = 0;
  for (size_t i = 0; i < n; i++) sum += q[i] * q[i];
  return sqrtf(sum);
}

static const char *vec_half_validate(const void *data, size_t n, int metric) {
  int nonzero = 0;
  for (size_t i = 0; i < n; i++) {
    uint16_t h;
    memcpy(&h, (const u8 *)data + i * 2, 2);
    if ((h & 0x7c00) == 0x7c00) return "float16 vectors must contain only finite values";
    nonzero |= h & 0x7fff;
  }
  if (!nonzero && metric == VEC0_DISTANCE_METRIC_COSINE)
    return "float16 cosine vectors must have a nonzero norm";
  return NULL;
}
