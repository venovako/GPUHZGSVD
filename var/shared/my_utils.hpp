#ifndef MY_UTILS_HPP
#define MY_UTILS_HPP

#include "defines.hpp"

#ifndef err_msg_size
#define err_msg_size static_cast<size_t>(1024u)
#else /* err_msg_size */
#error err_msg_size not definable externally
#endif /* ?err_msg_size */

extern char err_msg[err_msg_size];

#ifndef EXIT
#define EXIT exit(EXIT_FAILURE)
#else /* EXIT */
#error EXIT not definable externally
#endif /* ?EXIT */

#ifndef WARN
#define WARN(msg) {                                                             \
    (void)fprintf(stderr, "[WARNING] %s(%d): %s\n", __FILE__, __LINE__, (msg)); \
    (void)fflush(stderr);                                                       \
  }
#else /* WARN */
#error WARN not definable externally
#endif /* ?WARN */

#ifndef DIE
#define DIE(msg) {                                                            \
    (void)fprintf(stderr, "[ERROR] %s(%d): %s\n", __FILE__, __LINE__, (msg)); \
    (void)fflush(stderr);                                                     \
    EXIT;                                                                     \
  }
#else /* DIE */
#error DIE not definable externally
#endif /* ?DIE */

#ifndef SYSI_CALL
#define SYSI_CALL(call) {                                                                 \
    if (0 != static_cast<int>(call)) {                                                    \
      (void)fprintf(stderr, "[ERROR] %s(%d): %s\n", __FILE__, __LINE__, strerror(errno)); \
      (void)fflush(stderr);                                                               \
      EXIT;                                                                               \
    }                                                                                     \
  }
#else /* SYSI_CALL */
#error SYSI_CALL not definable externally
#endif /* ?SYSI_CALL */

#ifndef SYSP_CALL
#define SYSP_CALL(call) {                                                                 \
    if (NULL == static_cast<const void*>(call)) {                                         \
      (void)fprintf(stderr, "[ERROR] %s(%d): %s\n", __FILE__, __LINE__, strerror(errno)); \
      EXIT;                                                                               \
    }                                                                                     \
  }
#else /* SYSP_CALL */
#error SYSP_CALL not definable externally
#endif /* ?SYSP_CALL */

template <typename T>
inline T udiv_ceil(const T a, const T b) throw()
{
  return (b ? ((a + (b - static_cast<T>(1u))) / b) : static_cast<T>(0u));
}

template <typename T>
T dimToMod(const T dim, const T mod) throw()
{
  T ret = static_cast<T>(0u);

  if (mod) {
    const T o = dim % mod;
    ret = (o ? dim + (mod - o) : dim);
  }

  return ret;
}

EXTERN_C unsigned long long atou(const char *const s) throw();

#ifndef TS2S
#define TS2S 1e-6
#else /* TS2S */
#error TS2S not definable externally
#endif /* ?TS2S */

#ifndef TS_S
#define TS_S 1000000ll
#else /* TS_S */
#error TS_S not definable externally
#endif /* ?TS_S */

EXTERN_C long long timestamp() throw();
EXTERN_C void stopwatch_reset(long long &sw) throw();
EXTERN_C long long stopwatch_lap(long long &sw) throw();

EXTERN_C int fresize(FILE *const f, const size_t s) throw();

template <typename T>
int fread_bycol(FILE *const f, const size_t m, const size_t n, T *const A, const size_t ldA) throw()
{
  if (!f)
    return -1;
  if (!m)
    return 0;
  if (!n)
    return 0;
  if (!A)
    return -4;
  if (ldA < m)
    return -5;

  const long co = ftell(f);
  SYSI_CALL(co < 0l);
  if (co)
    SYSI_CALL(fseek(f, 0l, SEEK_SET));

  for (size_t j = 0u; j < n; ++j) {
    T *const c = (A + ldA * j);
    SYSI_CALL(fread(c, sizeof(T), m, f) != m);
  }

  return 0;
}

template <typename T>
int fwrite_bycol(FILE *const f, const size_t m, const size_t n, const T *const A, const size_t ldA) throw()
{
  if (!f)
    return -1;
  if (!m)
    return 0;
  if (!n)
    return 0;
  if (!A)
    return -4;
  if (ldA < m)
    return -5;

  const long co = ftell(f);
  SYSI_CALL(co < 0l);
  if (co)
    SYSI_CALL(fseek(f, 0l, SEEK_SET));

  for (size_t j = 0u; j < n; ++j) {
    const T *const c = (A + ldA * j);
    SYSI_CALL(fwrite(c, sizeof(T), m, f) != m);
  }

  return 0;
}
#endif /* !MY_UTILS_HPP */
