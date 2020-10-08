#include "cuda_helper.hpp"

#include "my_utils.hpp"

int configureGPU(const int dev, cublasHandle_t &handle,
#ifdef USE_CUSOLVER
                 cusolverDnHandle_t &csh,
#endif /* USE_CUSOLVER */
                 const cudaStream_t s) throw()
{
  assert(dev >= 0);
  cudaDeviceProp cdp;
  CUDA_CALL(cudaGetDeviceProperties(&cdp, dev));
#ifndef USE_CUSOLVER
  if (WARP_SZ != static_cast<unsigned>(cdp.warpSize)) {
    (void)snprintf(err_msg, err_msg_size, "CUDA Device %d has %d threads in a warp, must be %u", dev, cdp.warpSize, WARP_SZ);
    DIE(err_msg);
  }
#endif /* !USE_CUSOLVER */
  CUDA_CALL(cudaSetDevice(dev));
#ifndef USE_CUSOLVER
  CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
#endif /* !USE_CUSOLVER */
  CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSetStream(handle, s));
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
#ifdef USE_CUSOLVER
  CUSOLVER_CALL(cusolverDnCreate(&csh));
  CUSOLVER_CALL(cusolverDnSetStream(csh, s));
#else /* !USE_CUSOLVER */
  CUBLAS_CALL(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
#endif /* ?USE_CUSOLVER */
  return (cdp.major * 10 + cdp.minor);
}

void freeGPU(cublasHandle_t &handle
#ifdef USE_CUSOLVER
             , cusolverDnHandle_t &csh
#endif /* USE_CUSOLVER */
             ) throw()
{
#ifdef USE_CUSOLVER
  CUSOLVER_CALL(cusolverDnDestroy(csh));
#endif /* USE_CUSOLVER */
  CUBLAS_CALL(cublasDestroy(handle));
#if (defined(PROFILE) && (PROFILE != 0))
  CUDA_CALL(cudaDeviceReset());
#endif /* ?PROFILE */
}

void cuda_prof_start() throw()
{
#if (defined(PROFILE) && (PROFILE != 0))
  CUDA_CALL(cudaProfilerStart());
#endif /* ?PROFILE */
}

void cuda_prof_stop() throw()
{
#if (defined(PROFILE) && (PROFILE != 0))
  CUDA_CALL(cudaProfilerStop());
#endif /* ?PROFILE */
}
