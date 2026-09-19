/* Standalone native correctness + sanitizer driver; no Python/SQLite
 * connection. */
#include "structural.c"
#include <assert.h>
#include <stdio.h>
int main(void) {
  const int dimensions[] = {1, 7, 8, 17, 32, 33, 65, 769};
  unsigned state = 19;
  for (int h = 0; h < 2; h++)
    for (int di = 0; di < 8; di++) {
      int n = 129, d = dimensions[di];
      float *x = malloc(n * d * sizeof(float)), *q = malloc(d * sizeof(float));
      uint16_t *half = malloc(n * d * sizeof(uint16_t));
      float *centers = calloc(d, sizeof(float));
      int *labels = calloc(n, sizeof(int));
      for (int i = 0; i < n * d; i++) {
        state = state * 1664525u + 1013904223u;
        x[i] = ((int)(state % 10001) - 5000) / 1000.0f;
        half[i] = vec_float_to_half(x[i]);
      }
      for (int j = 0; j < d; j++)
        q[j] = h ? vec_half_to_float(half[j]) : x[j];
      for (int metric = 0; metric < 2; metric++) {
        Experiment *ref = experiment_build(h ? (void *)half : (void *)x, n, d,
                                           h, metric, 0, NULL, NULL, 0);
        assert(ref);
        for (int k = 1; k <= n; k = k == 1 ? 10 : k == 10 ? n : n + 1) {
          int *ri = malloc(k * sizeof(int)), *ci = malloc(k * sizeof(int));
          float *rd = malloc(k * sizeof(float)),
                *cd = malloc(k * sizeof(float));
          Stats stats;
          assert(experiment_query(ref, q, k, ri, rd, &stats, 0) == k);
          for (int mode = 1; mode <= 7; mode++) {
            if (mode == 4 ||
                (metric &&
                 (mode == 1 || mode == 2 || mode == 3 || mode == 7)) ||
                (!metric && mode == 5))
              continue;
            Experiment *e =
                experiment_build(h ? (void *)half : (void *)x, n, d, h, metric,
                                 mode, centers, labels, 1);
            assert(e);
            assert(experiment_query(e, q, k, ci, cd, &stats, 1) == k);
            assert(memcmp(ri, ci, k * sizeof(int)) == 0);
            assert(memcmp(rd, cd, k * sizeof(float)) == 0);
            experiment_free(e);
          }
          free(ri);
          free(ci);
          free(rd);
          free(cd);
        }
        experiment_free(ref);
      }
      free(x);
      free(q);
      free(half);
      free(centers);
      free(labels);
    }
  puts("native structural checks passed");
  return 0;
}
