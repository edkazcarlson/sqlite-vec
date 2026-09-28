/* Internal FP16 dispatch checks; compile with sqlite3.c and -Ivendor. */
#define SQLITE_CORE
#define SQLITE_VEC_STATIC
#include "../sqlite-vec.c"
#include <stdio.h>
#if defined(_M_X64) || defined(__SSE__)
#include <xmmintrin.h>
#endif
int main(void) {
#ifdef VEC_HAVE_F16C
  const u32 flags = (1u << 27) | (1u << 28) | (1u << 29);
  assert(vec_half_f16c_usable(flags, 6));
  for (int bit = 27; bit <= 29; bit++)
    assert(!vec_half_f16c_usable(flags & ~(1u << bit), 6));
  for (u64 state = 0; state < 6; state++)
    assert(!vec_half_f16c_usable(flags, state));
#endif
#if defined(SQLITE_VEC_TEST) && defined(SQLITE_VEC_TEST_NO_F16C)
  assert(!vec_half_has_f16c());
  assert(vec_half_distance_select(VEC0_DISTANCE_METRIC_L2) == l2_sqr_float16);
#endif
  /* Compare every finite encoding bit-for-bit, including signed zero, with
   * public general conversion under all x86 FTZ/DAZ and rounding-mode combinations. */
#if defined(_M_X64) || defined(__SSE__)
  unsigned saved = _mm_getcsr();
  const unsigned modes[] = {
    0, 0x8000, 0x0040, 0x8040,
    0x2000, 0xa000, 0x2040, 0xa040,
    0x4000, 0xc000, 0x4040, 0xc040,
    0x6000, 0xe000, 0x6040, 0xe040
  };
#else
  const unsigned modes[] = {0};
#endif
  for (size_t m = 0; m < countof(modes); m++) {
#if defined(_M_X64) || defined(__SSE__)
    _mm_setcsr((saved & ~0xe040u) | modes[m]);
#endif
    for (u32 h = 0; h <= 65535; h++) {
      if ((h & 0x7c00) == 0x7c00) continue;
      uint16_t half = (uint16_t)h;
      u8 unaligned[3];
      memcpy(unaligned + 1, &half, 2);
      f32 expected = vec_half_to_float(half);
      f32 actual = vec_half_at_finite(unaligned + 1, 0);
      assert(memcmp(&actual, &expected, sizeof(actual)) == 0);
    }
  }
#if defined(_M_X64) || defined(__SSE__)
  _mm_setcsr(saved);
#endif
  const size_t dims[] = {1, 7, 8, 15, 16, 17, 31, 32, 33, 769, 8192};
  u8 bytes[8192 * 2 + 1];
  f32 q[8192];
  for (size_t i = 0; i < 8192; i++) {
    uint16_t h = vec_float_to_half((f32)((int)(i % 23) - 11) / 16);
    memcpy(bytes + 1 + i * 2, &h, 2);
    q[i] = (f32)((int)(i % 19) - 9) / 16;
  }
  for (size_t d = 0; d < countof(dims); d++) {
    size_t n = dims[d];
    f32 qn = vec_query_norm(q, n);
    for (int metric = 1; metric <= 3; metric++) {
      vec_half_distance_fn selected = vec_half_distance_select(metric);
      f32 actual = selected(bytes + 1, q, n, qn);
      f32 expected = metric == 1 ? l2_sqr_float16(bytes + 1, q, n, qn) :
                     metric == 2 ? cosine_float16(bytes + 1, q, n, qn) :
                                   l1_float16(bytes + 1, q, n, qn);
      assert(fabsf(actual - expected) <= 2e-5f * fmaxf(1, fabsf(expected)));
    }
  }
  puts("FP16 finite conversion (all modes), dispatch and unaligned/tail checks passed");
  return 0;
}
