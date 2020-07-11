#include "device_code.hpp"

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "cuda_helper.hpp"

#include "device_code_common.hpp"
#include "device_code_cdsort_0.hpp"
#include "device_code_cdsort_accumV.hpp"

static const dim3 hzL1bD(HZ_L1_THREADS_PER_BLOCK_X, HZ_L1_THREADS_PER_BLOCK_Y, 1u);

void HZ_L1_sv(double *const F, double *const G) throw()
{
  const dim3 hzL1gD(STRAT1_PAIRS, 1u, 1u);
  dHZ_L1_sv<<< hzL1gD, hzL1bD >>>(F, G);
}

void initS(double *const F, double *const G, double *const V, double *const S, double *const H, double *const K, const unsigned nRank) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  dInitS<<< gD, bD, shmD >>>(F, G, V, S, H, K);
}

void initS(double *const F, double *const G, double *const V, const unsigned nRank) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  dInitS<<< gD, bD, shmD >>>(F, G, V);
}

void initV(double *const F, double *const G, double *const V, const unsigned nRank) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  dInitV<<< gD, bD, shmD >>>(F, G, V);
}

void initSymbols
(double *const W,
 unsigned long long *const C,
 const unsigned nRowF,
 const unsigned nRowG,
 const unsigned nRowV,
 const unsigned nRowW,
 const unsigned ldF,
 const unsigned ldG,
 const unsigned ldV,
 const unsigned ldW,
 const unsigned nRank,
 const unsigned nSwp
) throw()
{
  CUDA_CALL(cudaMemcpyToSymbol(_W, &W, sizeof(double*)));
  CUDA_CALL(cudaMemcpyToSymbol(_C, &C, sizeof(unsigned long long*)));

  CUDA_CALL(cudaMemcpyToSymbol(_nRowF, &nRowF, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_nRowG, &nRowG, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_nRowV, &nRowV, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_nRowW, &nRowW, sizeof(unsigned)));

  CUDA_CALL(cudaMemcpyToSymbol(_ldF, &ldF, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_ldG, &ldG, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_ldV, &ldV, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_ldW, &ldW, sizeof(unsigned)));

  CUDA_CALL(cudaMemcpyToSymbol(_nRank, &nRank, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_nSwp, &nSwp, sizeof(unsigned)));

  CUDA_CALL(cudaMemcpyToSymbol(_STRAT0_STEPS, &STRAT0_STEPS, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_STRAT0_PAIRS, &STRAT0_PAIRS, sizeof(unsigned)));

  CUDA_CALL(cudaMemcpyToSymbol(_strat0, strat0, sizeof(strat0)));
}
