#ifndef MY_UTILS_HPP
#define MY_UTILS_HPP

#include "defines.hpp"

#ifndef err_msg_size
#define err_msg_size static_cast<size_t>(1024u)
#else // err_msg_size
#error err_msg_size not definable externally
#endif // !err_msg_size

EXTERN_C char err_msg[err_msg_size];

#ifndef WARN
#define WARN(msg) {                                                             \
    (void)fprintf(stderr, "[WARNING] %s(%d): %s\n", __FILE__, __LINE__, (msg)); \
  }
#else // WARN
#error WARN not definable externally
#endif // !WARN

#ifndef DIE
#define DIE(msg) {                                                            \
    (void)fprintf(stderr, "[ERROR] %s(%d): %s\n", __FILE__, __LINE__, (msg)); \
    exit(EXIT_FAILURE);                                                       \
  }
#else // DIE
#error DIE not definable externally
#endif // !DIE

#ifndef SYSI_CALL
#define SYSI_CALL(call) {						\
    if (0 != static_cast<int>(call)) {					\
      (void)fprintf(stderr, "[ERROR] %s(%d): %s",                       \
		    __FILE__, __LINE__, strerror(errno));		\
      exit(EXIT_FAILURE);                                               \
    }									\
  }
#else // SYSI_CALL
#error SYSI_CALL not definable externally
#endif // !SYSI_CALL

#ifndef SYSP_CALL
#define SYSP_CALL(call) {						\
    if (NULL == static_cast<const void*>(call)) {			\
      (void)fprintf(stderr, "[ERROR] %s(%d): %s",                       \
		    __FILE__, __LINE__, strerror(errno));		\
      exit(EXIT_FAILURE);                                               \
    }									\
  }
#else // SYSP_CALL
#error SYSP_CALL not definable externally
#endif // !SYSP_CALL

EXTERN_C unsigned long long atou(const char *const s) throw();
EXTERN_C int fexist(const char *const fn) throw();

EXTERN_C void *strat_open(const char *const sdy) throw();
EXTERN_C int strat_close(void *const h) throw();
EXTERN_C const void *strat_ptr(void *const h, const char *const snp, const unsigned n) throw();

template <typename T>
T udiv_ceil(const T a, const T b) throw()
{
  return (a + b - static_cast<T>(1u)) / b;
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

#ifndef TS2S
#define TS2S 1e-6
#else // TS2S
#error TS2S not definable externally
#endif // !TS2S

#ifndef TS_S
#define TS_S 1000000ll
#else // TS_S
#error TS_S not definable externally
#endif // !TS_S

EXTERN_C long long timestamp() throw();
EXTERN_C void stopwatch_reset(long long &sw) throw();
EXTERN_C long long stopwatch_lap(long long &sw) throw();

EXTERN_C int fread_bycol(FILE *const f, const size_t m, const size_t n, double *const A, const size_t ldA, const long off = 0l, const long stride = 1l) throw();
EXTERN_C int fwrite_bycol(FILE *const f, const size_t m, const size_t n, const double *const A, const size_t ldA, const long off = 0l, const long stride = 1l) throw();

#endif // !MY_UTILS_HPP
