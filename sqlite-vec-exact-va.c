/* Experimental exact L2 index: int8 reconstructions with deterministic residual
 * bounds. Original vectors remain intact. A row-addressable copy avoids walking
 * SQLite overflow chains for every survivor (explicit storage/write tradeoff).
 * Format v1: per slot [double scale, double radius, int64 norm, int8 codes[d]].
 * Like the existing float BLOBs, records use the host byte order. */
static int exact_va_exec(vec0_vtab *p, char *sql) {
  if (!sql)
    return SQLITE_NOMEM;
  char *error = NULL;
  int rc = sqlite3_exec(p->db, sql, NULL, NULL, &error);
  if (error) {
    vtab_set_error(&p->base, "%s", error);
    sqlite3_free(error);
  }
  sqlite3_free(sql);
  return rc;
}
static int exact_va_tables(vec0_vtab *p, int create) {
  for (int i = 0; i < p->numVectorColumns; i++) {
    if (!p->vector_columns[i].exact_va)
      continue;
    int rc;
    if (create)
      rc = exact_va_exec(
          p,
          sqlite3_mprintf(
              "CREATE TABLE \"%w\".\"%w_exactvachunks%02d\"(rowid INTEGER "
              "PRIMARY KEY,vectors BLOB NOT NULL);"
              "CREATE TABLE \"%w\".\"%w_exactvavectors%02d\"(rowid INTEGER "
              "PRIMARY KEY,vector BLOB NOT NULL);",
              p->schemaName, p->tableName, i, p->schemaName, p->tableName, i));
    else
      rc = exact_va_exec(
          p,
          sqlite3_mprintf("DROP TABLE \"%w\".\"%w_exactvachunks%02d\";DROP "
                          "TABLE \"%w\".\"%w_exactvavectors%02d\";",
                          p->schemaName, p->tableName, i, p->schemaName,
                          p->tableName, i));
    if (rc != SQLITE_OK)
      return rc;
  }
  return SQLITE_OK;
}
static double exact_va_value(const void *raw, int half, int j) {
  if (half) {
    uint16_t h;
    memcpy(&h, (const u8 *)raw + j * 2, 2);
    return vec_half_to_float(h);
  }
  f32 f;
  memcpy(&f, (const u8 *)raw + j * 4, 4);
  return f;
}
static void exact_va_quantize(const void *raw, int half, int d, u8 *record) {
  double maxabs = 0, error = 0;
  i64 norm = 0;
  for (int j = 0; j < d; j++)
    maxabs = fmax(maxabs, fabs(exact_va_value(raw, half, j)));
  double scale = maxabs ? maxabs / 127 : 1;
  for (int j = 0; j < d; j++) {
    double x = exact_va_value(raw, half, j);
    int c = (int)round(x / scale);
    c = c < -127 ? -127 : c > 127 ? 127 : c;
    record[24 + j] = (u8)(i8)c;
    norm += (i64)c * c;
    double delta = x - c * scale;
    error += delta * delta;
  }
  error = sqrt(error) * (1 + 16 * d * DBL_EPSILON) +
          maxabs * sqrt((double)d) * 32 * DBL_EPSILON;
  memcpy(record, &scale, 8);
  memcpy(record + 8, &error, 8);
  memcpy(record + 16, &norm, 8);
}
static int exact_va_record_valid(const u8 *record, int d) {
  double scale, error;
  i64 norm;
  memcpy(&scale, record, 8);
  memcpy(&error, record + 8, 8);
  memcpy(&norm, record + 16, 8);
  return isfinite(scale) && scale > 0 && isfinite(error) && error >= 0 &&
         norm >= 0 && norm <= (i64)d * 127 * 127;
}
static double exact_va_bound(const u8 *a, const u8 *q, int d) {
  double sa, ea, sq, eq;
  i64 na, nq, dot = 0;
  memcpy(&sa, a, 8);
  memcpy(&ea, a + 8, 8);
  memcpy(&na, a + 16, 8);
  memcpy(&sq, q, 8);
  memcpy(&eq, q + 8, 8);
  memcpy(&nq, q + 16, 8);
  const i8 *x = (const i8 *)a + 24, *y = (const i8 *)q + 24;
  int j = 0;
#ifdef SQLITE_VEC_ENABLE_AVX
  __m256i sums = _mm256_setzero_si256();
  for (; j + 16 <= d; j += 16) {
    __m256i ax =
        _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + j)));
    __m256i by =
        _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(y + j)));
    sums = _mm256_add_epi32(sums, _mm256_madd_epi16(ax, by));
  }
  int lanes[8];
  _mm256_storeu_si256((__m256i *)lanes, sums);
  for (int l = 0; l < 8; l++)
    dot += lanes[l];
#endif
  for (; j < d; j++)
    dot += (int)x[j] * y[j];
  double xx = na * sa * sa, qq = nq * sq * sq, cross = 2 * dot * sa * sq;
  double squared =
      fmax(0, xx + qq - cross - (xx + qq + fabs(cross)) * 32 * DBL_EPSILON);
  double lower = fmax(0, sqrt(squared) - ea - eq);
  /* Cover subtraction/product/summation/sqrt rounding in the original fp32
   * scorer, including subnormal underflow. The factor is deliberately loose;
   * reject only on a strict comparison. 8192 dimensions keeps gamma < .016. */
  return fmax(0, lower * (1 - 16.0 * (d + 1) * FLT_EPSILON) -
                     sqrt((double)d * FLT_MIN));
}
static int exact_va_write(vec0_vtab *p, int col, i64 chunk, i64 offset,
                          i64 rowid, const void *raw) {
  int d = p->vector_columns[col].dimensions, size = d + 24;
  u8 *record = sqlite3_malloc(size);
  if (!record)
    return SQLITE_NOMEM;
  exact_va_quantize(raw,
                    p->vector_columns[col].element_type ==
                        SQLITE_VEC_ELEMENT_TYPE_FLOAT16,
                    d, record);
  sqlite3_stmt *stmt = NULL;
  sqlite3_blob *blob = NULL;
  char *sql =
      sqlite3_mprintf("INSERT OR IGNORE INTO \"%w\".\"%w_exactvachunks%02d\" "
                      "VALUES (?,zeroblob(?))",
                      p->schemaName, p->tableName, col);
  int rc = sql ? sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL) : SQLITE_NOMEM;
  sqlite3_free(sql);
  if (rc != SQLITE_OK)
    goto done;
  sqlite3_bind_int64(stmt, 1, chunk);
  sqlite3_bind_int64(stmt, 2, p->chunk_size * size);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  stmt = NULL;
  if (rc != SQLITE_DONE)
    goto done;
  char *name = sqlite3_mprintf("%s_exactvachunks%02d", p->tableName, col);
  rc = name ? sqlite3_blob_open(p->db, p->schemaName, name, "vectors", chunk, 1,
                                &blob)
            : SQLITE_NOMEM;
  sqlite3_free(name);
  if (rc != SQLITE_OK)
    goto done;
  if (sqlite3_blob_bytes(blob) != p->chunk_size * size) {
    rc = SQLITE_CORRUPT_VTAB;
    goto done;
  }
  rc = sqlite3_blob_write(blob, record, size, offset * size);
  if (rc != SQLITE_OK)
    goto done;
  rc = sqlite3_blob_close(blob);
  blob = NULL;
  if (rc != SQLITE_OK)
    goto done;
  sql = sqlite3_mprintf(
      "INSERT OR REPLACE INTO \"%w\".\"%w_exactvavectors%02d\" VALUES (?,?)",
      p->schemaName, p->tableName, col);
  rc = sql ? sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL) : SQLITE_NOMEM;
  sqlite3_free(sql);
  if (rc != SQLITE_OK)
    goto done;
  sqlite3_bind_int64(stmt, 1, rowid);
  sqlite3_bind_blob(stmt, 2, raw,
                    vector_column_byte_size(p->vector_columns[col]),
                    SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_DONE)
    rc = SQLITE_OK;
done:
  sqlite3_finalize(stmt);
  int brc = sqlite3_blob_close(blob);
  sqlite3_free(record);
  return rc == SQLITE_OK ? brc : rc;
}
static int exact_va_delete(vec0_vtab *p, i64 id, int chunk) {
  for (int i = 0; i < p->numVectorColumns; i++)
    if (p->vector_columns[i].exact_va) {
      int rc = exact_va_exec(
          p, sqlite3_mprintf(
                 "DELETE FROM \"%w\".\"%w_exactva%s%02d\" WHERE rowid=%lld",
                 p->schemaName, p->tableName, chunk ? "chunks" : "vectors", i,
                 id));
      if (rc != SQLITE_OK)
        return rc;
    }
  return SQLITE_OK;
}
