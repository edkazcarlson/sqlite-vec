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

static void vec_half_expand(const void *data, f32 *out, size_t n) {
  for (size_t i = 0; i < n; i++) out[i] = vec_half_at(data, i);
}

/* The query has already been widened once by the caller. */
typedef f32 (*vec_half_distance_fn)(const void *, const f32 *, size_t, int, f32);

static f32 vec_half_distance_scalar(const void *a, const f32 *q, size_t n,
                                    int metric, f32 query_norm) {
  f32 sum = 0, norm = 0;
  for (size_t i = 0; i < n; i++) {
    f32 x = vec_half_at(a, i);
    if (metric == VEC0_DISTANCE_METRIC_COSINE) {
      sum += x * q[i];
      norm += x * x;
    } else {
      f32 diff = x - q[i];
      sum += metric == VEC0_DISTANCE_METRIC_L1 ? fabsf(diff) : diff * diff;
    }
  }
  if (metric == VEC0_DISTANCE_METRIC_COSINE)
    return 1 - (sum / (sqrtf(norm) * query_norm));
  return metric == VEC0_DISTANCE_METRIC_L2 ? sqrtf(sum) : sum;
}

#if defined(SQLITE_VEC_ENABLE_AVX) && (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define VEC_HAVE_F16C 1
__attribute__((target("avx,f16c")))
static f32 vec_half_distance_f16c(const void *a, const f32 *q, size_t n,
                                 int metric, f32 query_norm) {
  __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps()};
  __m256 norms[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps(), _mm256_setzero_ps()};
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    for (int lane = 0; lane < 4; lane++) {
      __m128i h = _mm_loadu_si128((const __m128i *)((const u8 *)a + (i + lane * 8) * 2));
      __m256 x = _mm256_cvtph_ps(h);
      __m256 y = _mm256_loadu_ps(q + i + lane * 8);
      if (metric == VEC0_DISTANCE_METRIC_COSINE) {
        sums[lane] = _mm256_add_ps(sums[lane], _mm256_mul_ps(x, y));
        norms[lane] = _mm256_add_ps(norms[lane], _mm256_mul_ps(x, x));
      } else {
        __m256 diff = _mm256_sub_ps(x, y);
        __m256 term = metric == VEC0_DISTANCE_METRIC_L1 ? _mm256_andnot_ps(sign_mask, diff) : _mm256_mul_ps(diff, diff);
        sums[lane] = _mm256_add_ps(sums[lane], term);
      }
    }
  }
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)((const u8 *)a + i * 2)));
    __m256 y = _mm256_loadu_ps(q + i);
    if (metric == VEC0_DISTANCE_METRIC_COSINE) {
      sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(x, y));
      norms[0] = _mm256_add_ps(norms[0], _mm256_mul_ps(x, x));
    } else {
      __m256 diff = _mm256_sub_ps(x, y);
      sums[0] = _mm256_add_ps(sums[0], metric == VEC0_DISTANCE_METRIC_L1 ? _mm256_andnot_ps(sign_mask, diff) : _mm256_mul_ps(diff, diff));
    }
  }
  f32 s[8], m[8];
  _mm256_storeu_ps(s, _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3])));
  _mm256_storeu_ps(m, _mm256_add_ps(_mm256_add_ps(norms[0], norms[1]), _mm256_add_ps(norms[2], norms[3])));
  f32 sum = 0, norm = 0;
  for (int j = 0; j < 8; j++) { sum += s[j]; norm += m[j]; }
  for (; i < n; i++) {
    f32 x = vec_half_at(a, i);
    if (metric == VEC0_DISTANCE_METRIC_COSINE) { sum += x * q[i]; norm += x * x; }
    else { f32 diff = x - q[i]; sum += metric == VEC0_DISTANCE_METRIC_L1 ? fabsf(diff) : diff * diff; }
  }
  if (metric == VEC0_DISTANCE_METRIC_COSINE) return 1 - sum / (sqrtf(norm) * query_norm);
  return metric == VEC0_DISTANCE_METRIC_L2 ? sqrtf(sum) : sum;
}
#endif

static vec_half_distance_fn vec_half_kernel(void) {
#ifdef VEC_HAVE_F16C
  /* Check AVX as well: older compilers' F16C check does not include OS support. */
  if (__builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c"))
    return vec_half_distance_f16c;
#endif
  return vec_half_distance_scalar;
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
