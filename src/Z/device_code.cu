#include "HZ.hpp"

#include "device_code.hpp"
#include "device_code_common.hpp"
#include "device_code_accumV.hpp"
#if (CVG == 0)
#include "device_code_nosort_0.hpp"
#elif (CVG == 1)
#include "device_code_nosort_1.hpp"
#elif (CVG == 2)
#include "device_code_nosort_2.hpp"
#elif (CVG == 3)
#include "device_code_nosort_3.hpp"
#elif (CVG == 4)
#include "device_code_nosort_4.hpp"
#elif (CVG == 5)
#include "device_code_nosort_5.hpp"
#elif (CVG == 6)
#include "device_code_nosort_6.hpp"
#elif (CVG == 7)
#include "device_code_nosort_7.hpp"
#else // unknown CVG
#error CVG unknown
#endif // ?CVG
#include "device_code_nosort_accumV.hpp"

#include "my_utils.hpp"

static const dim3 hzL1bD(HZ_L1_THREADS_PER_BLOCK_X, HZ_L1_THREADS_PER_BLOCK_Y, 1u);

void HZ_L1_v(const unsigned step) throw()
{
  const dim3 hzL1gD(STRAT1_PAIRS, 1u, 1u);
  CUDA_CALL(cudaConfigureCall(hzL1gD, hzL1bD));
  CUDA_CALL(cudaSetupArgument(step, static_cast<size_t>(0u)));
  CUDA_CALL(cudaLaunch(zHZ_L1_v));
}

void initS(const int full, const unsigned nRank, const cudaStream_t s) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);

  CUDA_CALL(cudaConfigureCall(gD, bD, shmD, s));
  CUDA_CALL(cudaSetupArgument(full, static_cast<size_t>(0u)));
  CUDA_CALL(cudaLaunch(zInitS));
}

void initV(const int sclV, const unsigned nRank, const cudaStream_t s) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);

  CUDA_CALL(cudaConfigureCall(gD, bD, shmD, s));
  CUDA_CALL(cudaSetupArgument(sclV, static_cast<size_t>(0u)));
  CUDA_CALL(cudaLaunch(zInitV));
}

void initSymbols
(cuD *const FD, cuJ *const FJ,
 cuD *const GD, cuJ *const GJ,
 cuD *const VD, cuJ *const VJ,
 double *const S,
 double *const H,
 double *const K,
 const unsigned nRow,
 const unsigned nRank,
 const unsigned ldF,
 const unsigned ldG,
 const unsigned ldV,
 const unsigned nSwp
) throw()
{
  CUDA_CALL(cudaMemcpyToSymbolAsync(_FD, &FD, sizeof(cuD*)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_FJ, &FJ, sizeof(cuJ*)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_GD, &GD, sizeof(cuD*)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_GJ, &GJ, sizeof(cuJ*)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_VD, &VD, sizeof(cuD*)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_VJ, &VJ, sizeof(cuJ*)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_S, &S, sizeof(double*)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_H, &H, sizeof(double*)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_K, &K, sizeof(double*)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRow, &nRow, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRank, &nRank, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_ldF, &ldF, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_ldG, &ldG, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_ldV, &ldV, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nSwp, &nSwp, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_STRAT0_STEPS, &STRAT0_STEPS, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_STRAT0_PAIRS, &STRAT0_PAIRS, sizeof(unsigned)));
  // copy strategy tables
  CUDA_CALL(cudaMemcpyToSymbolAsync(_strat0, strat0, sizeof(strat0)));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_strat1, strat1, sizeof(strat1)));
}
