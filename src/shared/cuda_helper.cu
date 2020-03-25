#include "cuda_helper.hpp"

#include "my_utils.hpp"
/* use nvprof
#ifdef PROFILE
#ifdef USE_MPI
#include "mpi_helper.hpp"
#endif // USE_MPI
#endif // PROFILE
*/

int configureGPUex(const int dev, const unsigned maxShMemB) throw()
{
  assert(dev >= 0);
  CUDA_CALL(cudaSetDevice(dev));

  cudaDeviceProp cdp;
  CUDA_CALL(cudaGetDeviceProperties(&cdp, dev));
  const int dcc = cdp.major * 10 + cdp.minor;

  if (dcc < 30) {
    (void)snprintf(err_msg, err_msg_size, "CUDA Device %d Compute Capability %d < 30", dev, dcc);
    DIE(err_msg);
  }

  if (WARP_SZ != static_cast<unsigned>(cdp.warpSize)) {
    (void)snprintf(err_msg, err_msg_size, "CUDA Device %d has %d threads in a warp, must be %u", dev, cdp.warpSize, WARP_SZ);
    DIE(err_msg);
  }

  cudaFuncCache cacheConfig = cudaFuncCachePreferNone;
  if (maxShMemB) {
    if (maxShMemB <= 16384u) // 16 kB
      cacheConfig = cudaFuncCachePreferL1;
    else if (maxShMemB <= 32768u) // 32 kB
      cacheConfig = cudaFuncCachePreferEqual;
    else if (maxShMemB <= 49152u) // 48 kB
      cacheConfig = cudaFuncCachePreferShared;
    else { // > 48 kB
      (void)snprintf(err_msg, err_msg_size, "Maximum shared memory requested (%u B) > 48 kB", maxShMemB);
      WARN(err_msg);
    }
  }
  CUDA_CALL(cudaDeviceSetCacheConfig(cacheConfig));
  CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  /* use nvprof
#ifdef PROFILE
#ifdef USE_MPI
#ifdef USE_COMPLEX
  (void)snprintf(err_msg, err_msg_size, "Z%d_%d_%d_%d.csv", CVG, PROFILE, mpi_rank, dev);
#else // !USE_COMPLEX
  (void)snprintf(err_msg, err_msg_size, "D%d_%d_%d_%d.csv", CVG, PROFILE, mpi_rank, dev);
#endif // ?USE_COMPLEX
#else // !USE_MPI
#ifdef USE_COMPLEX
  (void)snprintf(err_msg, err_msg_size, "Z%d_%d_%d.csv", CVG, PROFILE, dev);
#else // !USE_COMPLEX
  (void)snprintf(err_msg, err_msg_size, "D%d_%d_%d.csv", CVG, PROFILE, dev);
#endif // ?USE_COMPLEX
#endif // ?USE_MPI
  const size_t err_msg_size_2 = err_msg_size >> 1u;
  (void)snprintf(err_msg + err_msg_size_2, err_msg_size_2, "%d.cfg", PROFILE);
  CUDA_CALL(cudaProfilerInitialize(err_msg + err_msg_size_2, err_msg, cudaCSV));
#endif // PROFILE
  */

  return dcc;
}

int configureGPU(const int dev) throw()
{
#ifdef USE_COMPLEX
  static const unsigned maxShMemB = 49152u; // 48 kB
#else // !USE_COMPLEX
  static const unsigned maxShMemB = 24576u; // 24 kB
#endif // ?USE_COMPLEX
  return configureGPUex(dev, maxShMemB);
}

void cuda_prof_start() throw()
{
#ifdef PROFILE
  CUDA_CALL(cudaProfilerStart());
#endif // PROFILE
}

void cuda_prof_stop() throw()
{
#ifdef PROFILE
  CUDA_CALL(cudaProfilerStop());
#endif // PROFILE
}
