#include "my_utils.hpp"

char err_msg[err_msg_size] = { '\0' };

unsigned long long atou(const char *const s) throw()
{
  if (!s) {
    DIE("atou(NULL)");
  }
  const char *s_ = s;
  while (*s_) {
    if (isspace(*s_)) {
      ++s_;
    }
    else if ('-' == *s_) {
      DIE("atou(-)");
    }
    else if ('+' == *s_) {
      break;
    }
    else if (isdigit(*s_)) {
      break;
    }
    else {
      DIE("atou(\?)");
    }
  }
  if (!*s_) {
    DIE("atou(\"\")");
  }
  char *e = static_cast<char*>(NULL);
  const unsigned long long u = strtoull(s_, &e, 0);
  if (e && *e) {
    DIE("atou(!)");
  }
  return u;
}

long long timestamp() throw()
{
  struct timeval tv;
  SYSI_CALL(gettimeofday(&tv, static_cast<struct timezone*>(NULL)));
  return (tv.tv_sec * TS_S + tv.tv_usec);
}

void stopwatch_reset(long long &sw) throw()
{
  sw = timestamp();
}

long long stopwatch_lap(long long &sw) throw()
{
  const long long
    ts = timestamp(),
    lap = ts - sw;
  sw = ts;
  return lap;
}

int fresize(FILE *const f, const size_t s) throw()
{
  return (f ? ftruncate(fileno(f), static_cast<off_t>(s)) : -1);
}

int fread_bycol(FILE *const f, const size_t m, const size_t n, double *const A, const size_t ldA, const long off, const long stride) throw()
{
  if (!f)
    return -1;
  if (!m)
    return -2;
  if (!n)
    return -3;
  if (!A)
    return -4;
  if (!ldA)
    return -5;
  if (ldA < m)
    return -5;
  if (off < 0l)
    return -6;
  if (stride <= 0l)
    return -7;

  const long o = ftell(f);
  SYSI_CALL(o < 0l);
  const long od = off * static_cast<long>(sizeof(double));
  if (o != od)
    SYSI_CALL(fseek(f, od, SEEK_SET));

  if (stride == 1l) {
    for (size_t j = 0u; j < n; ++j) {
      double *const c = A + j * ldA;
      SYSI_CALL(fread(c, sizeof(double), m, f) != m);
    }
  }
  else {
    const long sd = (stride - 1l) * static_cast<long>(sizeof(double));
    for (size_t j = 0u; j < n; ++j) {
      double *const c = A + j * ldA;
      for (size_t i = 0u; i < m; ++i) {
        SYSI_CALL(fread((c + i), sizeof(double), 1u, f) != 1u);
        if ((j != (n - 1u)) || (i != (m - 1u))) {
          SYSI_CALL(fseek(f, sd, SEEK_CUR));
        }
      }
    }
  }

  return 0;
}

int fwrite_bycol(FILE *const f, const size_t m, const size_t n, const double *const A, const size_t ldA, const long off, const long stride) throw()
{
  if (!f)
    return -1;
  if (!m)
    return -2;
  if (!n)
    return -3;
  if (!A)
    return -4;
  if (!ldA)
    return -5;
  if (ldA < m)
    return -5;
  if (off < 0l)
    return -6;
  if (stride <= 0l)
    return -7;

  const long o = ftell(f);
  SYSI_CALL(o < 0l);
  const long od = off * static_cast<long>(sizeof(double));
  if (o != od)
    SYSI_CALL(fseek(f, od, SEEK_SET));

  if (stride == 1l) {
    for (size_t j = 0u; j < n; ++j) {
      const double *const c = A + j * ldA;
      SYSI_CALL(fwrite(c, sizeof(double), m, f) != m);
    }
  }
  else {
    const long sd = (stride - 1l) * static_cast<long>(sizeof(double));
    for (size_t j = 0u; j < n; ++j) {
      const double *const c = A + j * ldA;
      for (size_t i = 0u; i < m; ++i) {
        SYSI_CALL(fwrite((c + i), sizeof(double), 1u, f) != 1u);
        if ((j != (n - 1u)) || (i != (m - 1u))) {
          SYSI_CALL(fseek(f, sd, SEEK_CUR));
        }
      }
    }
  }

  return 0;
}
