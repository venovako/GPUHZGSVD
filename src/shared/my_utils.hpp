#ifndef MY_UTILS_HPP
#define MY_UTILS_HPP

#include "defines.hpp"

#ifdef USE_MPI
#include "mpi_helper.hpp"
#endif // USE_MPI

#ifndef err_msg_size
#define err_msg_size static_cast<size_t>(1024u)
#else // err_msg_size
#error err_msg_size not definable externally
#endif // ?err_msg_size

extern char err_msg[err_msg_size];

#ifndef EXIT
#ifdef USE_MPI
#define EXIT exit(fini_MPI())
#else // !USE_MPI
#define EXIT exit(EXIT_FAILURE)
#endif // ?USE_MPI
#else // EXIT
#error EXIT not definable externally
#endif // ?EXIT

#ifndef WARN
#define WARN(msg) {                                                             \
    (void)fprintf(stderr, "[WARNING] %s(%d): %s\n", __FILE__, __LINE__, (msg)); \
    (void)fflush(stderr);                                                       \
  }
#else // WARN
#error WARN not definable externally
#endif // ?WARN

#ifndef DIE
#define DIE(msg) {                                                            \
    (void)fprintf(stderr, "[ERROR] %s(%d): %s\n", __FILE__, __LINE__, (msg)); \
    (void)fflush(stderr);                                                     \
    EXIT;                                                                     \
  }
#else // DIE
#error DIE not definable externally
#endif // ?DIE

#ifndef SYSI_CALL
#define SYSI_CALL(call) {                                                                 \
    if (0 != static_cast<int>(call)) {                                                    \
      (void)fprintf(stderr, "[ERROR] %s(%d): %s\n", __FILE__, __LINE__, strerror(errno)); \
      (void)fflush(stderr);                                                               \
      EXIT;                                                                               \
    }                                                                                     \
  }
#else // SYSI_CALL
#error SYSI_CALL not definable externally
#endif // ?SYSI_CALL

#ifndef SYSP_CALL
#define SYSP_CALL(call) {                                                                 \
    if (NULL == static_cast<const void*>(call)) {                                         \
      (void)fprintf(stderr, "[ERROR] %s(%d): %s\n", __FILE__, __LINE__, strerror(errno)); \
      EXIT;                                                                               \
    }                                                                                     \
  }
#else // SYSP_CALL
#error SYSP_CALL not definable externally
#endif // ?SYSP_CALL

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
#else // TS2S
#error TS2S not definable externally
#endif // ?TS2S

#ifndef TS_S
#define TS_S 1000000ll
#else // TS_S
#error TS_S not definable externally
#endif // ?TS_S

EXTERN_C long long timestamp() throw();
EXTERN_C void stopwatch_reset(long long &sw) throw();
EXTERN_C long long stopwatch_lap(long long &sw) throw();

#endif // !MY_UTILS_HPP
