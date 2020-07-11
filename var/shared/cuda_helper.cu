#include "cuda_helper.hpp"

#include "my_utils.hpp"

int configureGPU(const int dev, cublasHandle_t &handle) throw()
{
  assert(dev >= 0);
  cudaDeviceProp cdp;
  CUDA_CALL(cudaGetDeviceProperties(&cdp, dev));
  if (WARP_SZ != static_cast<unsigned>(cdp.warpSize)) {
    (void)snprintf(err_msg, err_msg_size, "CUDA Device %d has %d threads in a warp, must be %u", dev, cdp.warpSize, WARP_SZ);
    DIE(err_msg);
  }

  CUDA_CALL(cudaSetDevice(dev));
  CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSetStream(handle, static_cast<cudaStream_t>(NULL)));
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CALL(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));

  return (cdp.major * 10 + cdp.minor);
}

void freeGPU(cublasHandle_t &handle) throw()
{
  CUBLAS_CALL(cublasDestroy(handle));
#if (defined(PROFILE) && (PROFILE != 0))
  CUDA_CALL(cudaDeviceReset());
#else /* !PROFILE || PROFILE == 0 */
  CUDA_CALL(cudaDeviceSynchronize());
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
