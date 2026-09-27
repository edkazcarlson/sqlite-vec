/* Internal FP16 dispatch checks; compile with sqlite3.c and -Ivendor. */
#define SQLITE_CORE
#define SQLITE_VEC_STATIC
#include "../sqlite-vec.c"
#include <stdio.h>
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
  puts("FP16 CPU/OS dispatch and unaligned/tail checks passed");
  return 0;
}
