#ifndef CUDA_HELPER_HPP
#define CUDA_HELPER_HPP

#include "defines.hpp"
#include "my_utils.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>
#if (defined(PROFILE) && (PROFILE != 0))
#include <cuda_profiler_api.h>
#endif /* ?PROFILE */
#include <cublas_v2.h>
#ifdef USE_CUSOLVER
#include <cusolverDn.h>
#endif /* USE_CUSOLVER */

#ifndef CUDA_CALL
#define CUDA_CALL(call) {                                             \
    const cudaError_t err = (call);                                   \
    if (cudaSuccess != err) {                                         \
      (void)fprintf(stderr, "CUDA runtime error %d [%s] @ %s(%d)!\n", \
                    static_cast<int>(err), cudaGetErrorString(err),   \
                    __FILE__, __LINE__);                              \
      EXIT;                                                           \
    }                                                                 \
}
#else /* CUDA_CALL */
#error CUDA_CALL not definable externally
#endif /* ?CUDA_CALL */

#ifndef CUBLAS_CALL
#define CUBLAS_CALL(call) {                                           \
    const cublasStatus_t err = (call);                                \
    if (CUBLAS_STATUS_SUCCESS != err) {                               \
      (void)fprintf(stderr, "CUBLAS error %d @ %s(%d)!\n",            \
                    static_cast<int>(err), __FILE__, __LINE__);       \
      EXIT;                                                           \
    }                                                                 \
}
#else /* CUBLAS_CALL */
#error CUBLAS_CALL not definable externally
#endif /* ?CUBLAS_CALL */

#ifdef USE_CUSOLVER
#ifndef CUSOLVER_CALL
#define CUSOLVER_CALL(call) {                                   \
    const cusolverStatus_t err = (call);                        \
    if (CUSOLVER_STATUS_SUCCESS != err) {                       \
      (void)fprintf(stderr, "CUSOLVER error %d @ %s(%d)!\n",    \
                    static_cast<int>(err), __FILE__, __LINE__); \
      EXIT;                                                     \
    }                                                           \
}
#else /* CUSOLVER_CALL */
#error CUSOLVER_CALL not definable externally
#endif /* ?CUSOLVER_CALL */
#endif /* USE_CUSOLVER */

#ifndef WARP_SZ
#define WARP_SZ 32u
#else /* WARP_SZ */
#error WARP_SZ not definable externally
#endif /* ?WARP_SZ */

EXTERN_C int configureGPU(const int dev, cublasHandle_t &handle,
#ifdef USE_CUSOLVER
                          cusolverDnHandle_t &csh,
#endif /* USE_CUSOLVER */
                          const cudaStream_t s = 0) throw();
EXTERN_C void freeGPU(cublasHandle_t &handle
#ifdef USE_CUSOLVER
                      , cusolverDnHandle_t &csh
#endif /* USE_CUSOLVER */
                      ) throw();

EXTERN_C void cuda_prof_start() throw();
EXTERN_C void cuda_prof_stop() throw();

#endif /* !CUDA_HELPER_HPP */
