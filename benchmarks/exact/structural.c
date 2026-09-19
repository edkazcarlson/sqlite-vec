/* Native research harness, deliberately separate from persistent vec0 formats.
 * Build by structural.py. No fast-math: pruning depends on conservative bounds.
 * Uses the production distance kernels to make scan comparisons meaningful. */
#include "../../sqlite-vec.c"
#include <float.h>
#include <time.h>

typedef struct {
  unsigned long long full, coordinates, bounds, cells, bytes;
  double bound_ms, score_ms, select_ms;
} Stats;
typedef struct {
  const void *raw;
  int n, d, half, metric, mode;
  unsigned char *codes;
  double *lo, *step;
  float *tiles, *norms;
  int8_t *signed_codes;
  double *scales, *errors;
  int64_t *code_norms;
  const float *centers;
  int nc, *offsets, *members;
  double *radii;
} Experiment;
static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static float value(Experiment *e, int i, int j) {
  if (!e->half)
    return ((const float *)e->raw)[(size_t)i * e->d + j];
  return vec_half_to_float(((const uint16_t *)e->raw)[(size_t)i * e->d + j]);
}
void experiment_free(Experiment *e) {
  if (!e)
    return;
  free(e->codes);
  free(e->lo);
  free(e->step);
  free(e->tiles);
  free(e->signed_codes);
  free(e->scales);
  free(e->errors);
  free(e->code_norms);
  free(e->norms);
  free(e->offsets);
  free(e->members);
  free(e->radii);
  free(e);
}
/* mode: 0 two-pass scan, 1 fused/early L2, 2 VA, 3 cells, 4 tiles,
 * 5 cached cosine norms, 6 fused full-distance scan. */
Experiment *experiment_build(const void *raw, int n, int d, int half,
                             int metric, int mode, const float *centers,
                             const int *labels, int nc) {
  if (!raw || n <= 0 || d <= 0 || d > 8192 || (metric != 0 && metric != 1) ||
      mode < 0 || mode > 7)
    return NULL;
  if ((mode == 2 || mode == 3 || mode == 7) && metric)
    return NULL;
  if (mode == 5 && !metric)
    return NULL;
  Experiment *e = calloc(1, sizeof(*e));
  if (!e)
    return NULL;
  e->raw = raw;
  e->n = n;
  e->d = d;
  e->half = half;
  e->metric = metric;
  e->mode = mode;
  if (mode == 7) {
    e->signed_codes = malloc((size_t)n * d);
    e->scales = malloc(n * sizeof(double));
    e->errors = malloc(n * sizeof(double));
    e->code_norms = malloc(n * sizeof(int64_t));
    if (!e->signed_codes || !e->scales || !e->errors || !e->code_norms)
      goto fail;
    for (int i = 0; i < n; i++) {
      double maxabs = 0;
      for (int j = 0; j < d; j++)
        maxabs = fmax(maxabs, fabs(value(e, i, j)));
      double scale = maxabs ? maxabs / 127 : 1, error = 0;
      int64_t norm = 0;
      e->scales[i] = scale;
      for (int j = 0; j < d; j++) {
        double x = value(e, i, j);
        int c = (int)round(x / scale);
        c = c < -127 ? -127 : c > 127 ? 127 : c;
        e->signed_codes[(size_t)i * d + j] = (int8_t)c;
        norm += c * c;
        double diff = x - c * scale;
        error += diff * diff;
      }
      e->code_norms[i] = norm;
      e->errors[i] = sqrt(error) * (1 + 16 * d * DBL_EPSILON) +
                     maxabs * sqrt((double)d) * 32 * DBL_EPSILON;
    }
  }
  if (mode == 2) {
    e->codes = malloc((size_t)n * d);
    e->lo = malloc(d * sizeof(double));
    e->step = malloc(d * sizeof(double));
    if (!e->codes || !e->lo || !e->step)
      goto fail;
    for (int j = 0; j < d; j++) {
      double lo = INFINITY, hi = -INFINITY;
      for (int i = 0; i < n; i++) {
        double x = value(e, i, j);
        lo = fmin(lo, x);
        hi = fmax(hi, x);
      }
      e->lo[j] = lo;
      e->step[j] = hi == lo ? 1 : nextafter((hi - lo) / 255.0, INFINITY);
    }
    for (int i = 0; i < n; i++)
      for (int j = 0; j < d; j++) {
        int code = (int)floor((value(e, i, j) - e->lo[j]) / e->step[j]);
        e->codes[(size_t)i * d + j] = (unsigned char)(code < 0     ? 0
                                                      : code > 255 ? 255
                                                                   : code);
      }
  }
  if (mode == 3) {
    if (!centers || !labels || nc <= 0)
      goto fail;
    e->centers = centers;
    e->nc = nc;
    e->offsets = calloc(nc + 1, sizeof(int));
    e->members = malloc(n * sizeof(int));
    e->radii = calloc(nc, sizeof(double));
    if (!e->offsets || !e->members || !e->radii)
      goto fail;
    for (int i = 0; i < n; i++) {
      if (labels[i] < 0 || labels[i] >= nc)
        goto fail;
      e->offsets[labels[i] + 1]++;
    }
    for (int c = 1; c <= nc; c++)
      e->offsets[c] += e->offsets[c - 1];
    int *pos = malloc(nc * sizeof(int));
    if (!pos)
      goto fail;
    memcpy(pos, e->offsets, nc * sizeof(int));
    for (int i = 0; i < n; i++) {
      int c = labels[i];
      e->members[pos[c]++] = i;
      double r = 0;
      for (int j = 0; j < d; j++) {
        double x = (double)value(e, i, j) - centers[c * d + j];
        r += x * x;
      }
      e->radii[c] = fmax(e->radii[c], sqrt(r) * (1 + 8 * d * DBL_EPSILON));
    }
    free(pos);
  }
  if (mode == 4) {
    e->tiles = calloc((size_t)((n + 7) / 8) * 8 * d, sizeof(float));
    if (!e->tiles)
      goto fail;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < d; j++)
        e->tiles[((size_t)(i / 8) * d + j) * 8 + i % 8] = value(e, i, j);
  }
  if (mode == 5) {
    e->norms = malloc(n * sizeof(float));
    if (!e->norms)
      goto fail;
    for (int i = 0; i < n; i++) {
      float sum = 0;
      int j = 0;
#if defined(SQLITE_VEC_ENABLE_AVX)
      __m256 a[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps(), _mm256_setzero_ps()};
      for (; j + 32 <= d; j += 32)
        for (int l = 0; l < 4; l++) {
          __m256 x = half ? _mm256_cvtph_ps(_mm_loadu_si128(
                                (const __m128i *)((const uint16_t *)raw +
                                                  (size_t)i * d + j + l * 8)))
                          : _mm256_loadu_ps((const float *)raw + (size_t)i * d +
                                            j + l * 8);
          a[l] = _mm256_add_ps(a[l], _mm256_mul_ps(x, x));
        }
      for (; j + 8 <= d; j += 8) {
        __m256 x =
            half ? _mm256_cvtph_ps(
                       _mm_loadu_si128((const __m128i *)((const uint16_t *)raw +
                                                         (size_t)i * d + j)))
                 : _mm256_loadu_ps((const float *)raw + (size_t)i * d + j);
        a[0] = _mm256_add_ps(a[0], _mm256_mul_ps(x, x));
      }
      __m256 total =
          _mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3]));
      if (half) {
        float lanes[8];
        _mm256_storeu_ps(lanes, total);
        for (int l = 0; l < 8; l++)
          sum += lanes[l];
      } else
        sum = vec_avx_sum(total);
#endif
      for (; j < d; j++) {
        float v = value(e, i, j);
        sum += v * v;
      }
      e->norms[i] = sqrtf(sum);
    }
  }
  return e;
fail:
  experiment_free(e);
  return NULL;
}
static int worse(float a, int ai, float b, int bi) {
  return a > b || (a == b && ai > bi);
}
static void down(float *ds, int *ids, int n, int p) {
  for (;;) {
    int c = 2 * p + 1;
    if (c >= n)
      break;
    if (c + 1 < n && worse(ds[c + 1], ids[c + 1], ds[c], ids[c]))
      c++;
    if (!worse(ds[c], ids[c], ds[p], ids[p]))
      break;
    float v = ds[p];
    ds[p] = ds[c];
    ds[c] = v;
    int id = ids[p];
    ids[p] = ids[c];
    ids[c] = id;
    p = c;
  }
}
static void offer(float *ds, int *ids, int *used, int k, float x, int id) {
  if (*used < k) {
    int p = (*used)++;
    ds[p] = x;
    ids[p] = id;
    while (p) {
      int parent = (p - 1) / 2;
      if (!worse(ds[p], ids[p], ds[parent], ids[parent]))
        break;
      float v = ds[p];
      ds[p] = ds[parent];
      ds[parent] = v;
      int t = ids[p];
      ids[p] = ids[parent];
      ids[parent] = t;
      p = parent;
    }
  } else if (worse(ds[0], ids[0], x, id)) {
    ds[0] = x;
    ids[0] = id;
    down(ds, ids, k, 0);
  }
}
static float score(Experiment *e, int i, const float *q, float qnorm) {
  size_t d = e->d;
  if (e->half)
    return vec_half_kernel()((const uint8_t *)e->raw + (size_t)i * d * 2, q, d,
                             e->metric ? VEC0_DISTANCE_METRIC_COSINE
                                       : VEC0_DISTANCE_METRIC_L2,
                             qnorm);
  const float *x = (const float *)e->raw + (size_t)i * d;
  if (!e->metric)
    return distance_l2_sqr_float(x, q, &d);
#if defined(SQLITE_VEC_ENABLE_AVX)
  if (d >= 8)
    return vec_avx_distance(x, q, d, VEC0_DISTANCE_METRIC_COSINE, qnorm);
#endif
  return distance_cosine_float(x, q, &d);
}
/* A deliberately loose envelope on positive fp32 L2 accumulation error.
 * Refuse pruning if the envelope is not useful. No fast-math or probabilistic
 * projection bounds. Borderline candidates always use the production scorer. */
static double safe_bound(double squared, int d) {
  double error = 16.0 * (d + 1) * FLT_EPSILON;
  if (error >= 0.5 || !isfinite(squared))
    return 0;
  return fmax(0,
              sqrt(fmax(0, squared)) * (1 - error) - sqrt((double)d * FLT_MIN));
}
void experiment_kernel(Experiment *e, const float *q, float *out);
int experiment_query(Experiment *e, const float *q, int k, int *ids, float *ds,
                     Stats *stats, int instrument) {
  if (k <= 0 || k > e->n)
    return -1;
  memset(stats, 0, sizeof(*stats));
  int used = 0;
  float qnorm = e->metric ? vec_query_norm(q, e->d) : 0;
  float *all =
      (e->mode == 0 || e->mode == 5) ? malloc(e->n * sizeof(float)) : NULL;
  if ((e->mode == 0 || e->mode == 5) && !all)
    return -2;
  float *lookup = NULL;
  if (e->mode == 2) {
    lookup = malloc((size_t)e->d * 256 * sizeof(float));
    if (!lookup)
      return -2;
    for (int j = 0; j < e->d; j++)
      for (int c = 0; c < 256; c++) {
        double lo = nextafter(e->lo[j] + c * e->step[j], -INFINITY);
        double hi = nextafter(e->lo[j] + (c + 1) * e->step[j], INFINITY);
        double delta = fmax(fmax(lo - q[j], q[j] - hi), 0);
        lookup[j * 256 + c] = nextafterf((float)(delta * delta), 0);
      }
  }
  int8_t *qcodes = NULL;
  double qscale = 0, qerror = 0;
  int64_t qcode_norm = 0;
  if (e->mode == 7) {
    qcodes = malloc(e->d);
    if (!qcodes)
      return -2;
    double maxabs = 0;
    for (int j = 0; j < e->d; j++)
      maxabs = fmax(maxabs, fabs(q[j]));
    qscale = maxabs ? maxabs / 127 : 1;
    for (int j = 0; j < e->d; j++) {
      int c = (int)round(q[j] / qscale);
      c = c < -127 ? -127 : c > 127 ? 127 : c;
      qcodes[j] = (int8_t)c;
      qcode_norm += c * c;
      double diff = q[j] - c * qscale;
      qerror += diff * diff;
    }
    qerror = sqrt(qerror) * (1 + 16 * e->d * DBL_EPSILON) +
             maxabs * sqrt((double)e->d) * 32 * DBL_EPSILON;
  }
  int *order = NULL;
  double *cb = NULL;
  if (e->mode == 3) {
    order = malloc(e->nc * sizeof(int));
    cb = malloc(e->nc * sizeof(double));
    if (!order || !cb) {
      free(order);
      free(cb);
      return -2;
    }
    for (int c = 0; c < e->nc; c++) {
      double sum = 0;
      for (int j = 0; j < e->d; j++) {
        double x = (double)q[j] - e->centers[c * e->d + j];
        sum += x * x;
      }
      double b =
          fmax(0, sqrt(sum) * (1 - 8 * e->d * DBL_EPSILON) - e->radii[c]);
      cb[c] = safe_bound(b * b, e->d);
      order[c] = c;
    }
    for (int c = 1; c < e->nc; c++) {
      int v = order[c], j = c;
      while (j && cb[order[j - 1]] > cb[v]) {
        order[j] = order[j - 1];
        j--;
      }
      order[j] = v;
    }
  }
  if (e->mode == 5) {
    double start = instrument ? now_ms() : 0;
    experiment_kernel(e, q, all);
    stats->full = e->n;
    stats->bytes = (unsigned long long)e->n * (e->d * (e->half ? 2 : 4) + 4);
    if (instrument)
      stats->score_ms += now_ms() - start;
  }
  for (int t = 0; t < (e->mode == 5 ? 0 : e->mode == 3 ? e->nc : e->n); t++) {
    int begin = t, end = t + 1;
    if (e->mode == 3) {
      int c = order[t];
      stats->cells++;
      if (used == k && cb[c] > ds[0])
        continue;
      begin = e->offsets[c];
      end = e->offsets[c + 1];
    }
    for (int p = begin; p < end; p++) {
      int i = e->mode == 3 ? e->members[p] : p;
      double start = instrument ? now_ms() : 0;
      int reject = 0;
      if (e->mode == 7 && used == k) {
        const int8_t *x = e->signed_codes + (size_t)i * e->d;
        int j = 0;
        int64_t dot = 0;
#if defined(SQLITE_VEC_ENABLE_AVX)
        __m256i sums = _mm256_setzero_si256();
        for (; j + 16 <= e->d; j += 16) {
          __m256i a =
              _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + j)));
          __m256i b = _mm256_cvtepi8_epi16(
              _mm_loadu_si128((const __m128i *)(qcodes + j)));
          sums = _mm256_add_epi32(sums, _mm256_madd_epi16(a, b));
        }
        int lanes[8];
        _mm256_storeu_si256((__m256i *)lanes, sums);
        for (int l = 0; l < 8; l++)
          dot += lanes[l];
#endif
        for (; j < e->d; j++)
          dot += (int)x[j] * qcodes[j];
        double xx = e->code_norms[i] * e->scales[i] * e->scales[i];
        double qq = qcode_norm * qscale * qscale,
               cross = 2 * dot * e->scales[i] * qscale;
        double squared = fmax(
            0, xx + qq - cross - (xx + qq + fabs(cross)) * 32 * DBL_EPSILON);
        double bound = fmax(0, sqrt(squared) - e->errors[i] - qerror);
        reject = safe_bound(bound * bound, e->d) > ds[0];
        stats->bounds++;
        stats->coordinates += e->d;
      }
      if ((e->mode == 1 || e->mode == 2) && !e->metric && used == k) {
        double sum = 0;
        stats->bounds++;
        int j = 0;
#if defined(SQLITE_VEC_ENABLE_AVX)
        if (e->mode == 2) {
          __m256 acc = _mm256_setzero_ps();
          const __m256i offsets =
              _mm256_setr_epi32(0, 256, 512, 768, 1024, 1280, 1536, 1792);
          for (; j + 8 <= e->d; j += 8) {
            __m128i packed = _mm_loadl_epi64(
                (const __m128i *)(e->codes + (size_t)i * e->d + j));
            __m256i indices =
                _mm256_add_epi32(_mm256_cvtepu8_epi32(packed), offsets);
            acc = _mm256_add_ps(
                acc, _mm256_i32gather_ps(lookup + j * 256, indices, 4));
            stats->coordinates += 8;
            if (j % 32 == 24) {
              float lanes[8];
              _mm256_storeu_ps(lanes, acc);
              sum = 0;
              for (int l = 0; l < 8; l++)
                sum += lanes[l];
              if (safe_bound(sum, e->d) > ds[0]) {
                reject = 1;
                break;
              }
            }
          }
          float lanes[8];
          _mm256_storeu_ps(lanes, acc);
          sum = 0;
          for (int l = 0; l < 8; l++)
            sum += lanes[l];
        }
#endif
        for (; !reject && j < e->d; j++) {
          double delta;
          if (e->mode == 2) {
            int code = e->codes[(size_t)i * e->d + j];
            sum += lookup[j * 256 + code];
          } else {
            delta = (double)value(e, i, j) - q[j];
            sum += delta * delta;
          }
          stats->coordinates++;
          if ((j % 32) == 31 && safe_bound(sum, e->d) > ds[0]) {
            reject = 1;
            break;
          }
        }
        if (safe_bound(sum, e->d) > ds[0])
          reject = 1;
      }
      if (instrument)
        stats->bound_ms += now_ms() - start;
      if (reject)
        continue;
      start = instrument ? now_ms() : 0;
      float dist;
      if (e->mode == 4 || e->mode == 5) {
        /* Research arithmetic; rerank candidates with the original kernel.
         * This experiment currently computes every final score to guarantee
         * exactness; tile/norm kernels are measured separately below. */
        dist = score(e, i, q, qnorm);
      } else
        dist = score(e, i, q, qnorm);
      stats->full++;
      stats->bytes += (unsigned long long)e->d * (e->half ? 2 : 4);
      if (instrument)
        stats->score_ms += now_ms() - start;
      start = instrument ? now_ms() : 0;
      if (all)
        all[i] = dist;
      else
        offer(ds, ids, &used, k, dist, i);
      if (instrument)
        stats->select_ms += now_ms() - start;
    }
  }
  if (all) {
    double start = instrument ? now_ms() : 0;
    for (int i = 0; i < e->n; i++)
      offer(ds, ids, &used, k, all[i], i);
    if (instrument)
      stats->select_ms += now_ms() - start;
    free(all);
  }
  for (int n = used; n > 1; n--) {
    float v = ds[0];
    ds[0] = ds[n - 1];
    ds[n - 1] = v;
    int id = ids[0];
    ids[0] = ids[n - 1];
    ids[n - 1] = id;
    down(ds, ids, n - 1, 0);
  }
  free(order);
  free(cb);
  free(lookup);
  free(qcodes);
  return used;
}
/* Alternative arithmetic only: caller must verify against the production
 * scorer and may not use the timings as end-to-end exact SQL latency. */
void experiment_kernel(Experiment *e, const float *q, float *out) {
  float qnorm = e->metric ? vec_query_norm(q, e->d) : 0;
  if (e->mode == 4) {
    for (int i = 0; i < e->n; i += 8) {
#if defined(SQLITE_VEC_ENABLE_AVX)
      __m256 a = _mm256_setzero_ps(), b = a;
      for (int j = 0; j < e->d; j++) {
        __m256 x = _mm256_loadu_ps(e->tiles + ((size_t)(i / 8) * e->d + j) * 8),
               y = _mm256_set1_ps(q[j]);
        if (e->metric) {
          a = _mm256_add_ps(a, _mm256_mul_ps(x, y));
          b = _mm256_add_ps(b, _mm256_mul_ps(x, x));
        } else {
          x = _mm256_sub_ps(x, y);
          a = _mm256_add_ps(a, _mm256_mul_ps(x, x));
        }
      }
      if (e->metric)
        a = _mm256_sub_ps(
            _mm256_set1_ps(1),
            _mm256_div_ps(
                a, _mm256_mul_ps(_mm256_sqrt_ps(b), _mm256_set1_ps(qnorm))));
      else
        a = _mm256_sqrt_ps(a);
      float tmp[8];
      _mm256_storeu_ps(tmp, a);
      for (int l = 0; l < 8 && i + l < e->n; l++)
        out[i + l] = tmp[l];
#else
      for (int l = 0; l < 8 && i + l < e->n; l++)
        out[i + l] = score(e, i + l, q, qnorm);
#endif
    }
  } else if (e->mode == 5) {
#if defined(SQLITE_VEC_ENABLE_AVX)
    if (!e->half && e->d < 8) {
#else
    if (!e->half) {
#endif
      for (int i = 0; i < e->n; i++)
        out[i] = score(e, i, q, qnorm);
      return;
    }
    for (int i = 0; i < e->n; i++) {
      float dot = 0;
      int j = 0;
#if defined(SQLITE_VEC_ENABLE_AVX)
      __m256 a[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps(), _mm256_setzero_ps()};
      for (; j + 32 <= e->d; j += 32)
        for (int l = 0; l < 4; l++) {
          __m256 x = e->half
                         ? _mm256_cvtph_ps(_mm_loadu_si128(
                               (const __m128i *)((const uint16_t *)e->raw +
                                                 (size_t)i * e->d + j + l * 8)))
                         : _mm256_loadu_ps((const float *)e->raw +
                                           (size_t)i * e->d + j + l * 8);
          a[l] = _mm256_add_ps(
              a[l], _mm256_mul_ps(x, _mm256_loadu_ps(q + j + l * 8)));
        }
      for (; j + 8 <= e->d; j += 8) {
        __m256 x =
            e->half
                ? _mm256_cvtph_ps(_mm_loadu_si128(
                      (const __m128i *)((const uint16_t *)e->raw +
                                        (size_t)i * e->d + j)))
                : _mm256_loadu_ps((const float *)e->raw + (size_t)i * e->d + j);
        a[0] = _mm256_add_ps(a[0], _mm256_mul_ps(x, _mm256_loadu_ps(q + j)));
      }
      __m256 total =
          _mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3]));
      if (e->half) {
        float lanes[8];
        _mm256_storeu_ps(lanes, total);
        for (int l = 0; l < 8; l++)
          dot += lanes[l];
      } else
        dot = vec_avx_sum(total);
#endif
      for (; j < e->d; j++)
        dot += value(e, i, j) * q[j];
      out[i] = 1 - dot / (e->norms[i] * qnorm);
    }
  } else
    for (int i = 0; i < e->n; i++)
      out[i] = score(e, i, q, qnorm);
}
