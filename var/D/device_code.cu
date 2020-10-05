#include "device_code.hpp"

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "cuda_helper.hpp"

#include "device_code_common.hpp"
#include "device_code_cdsort_0.hpp"
#include "device_code_cdsort_accumV.hpp"

static const dim3 hzL1bD(HZ_L1_THREADS_PER_BLOCK_X, HZ_L1_THREADS_PER_BLOCK_Y, 1u);

void HZ_L1_sv(double *const F, double *const G, const cudaStream_t s) throw()
{
  const dim3 hzL1gD(STRAT1_PAIRS, 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  dHZ_L1_sv<<< hzL1gD, hzL1bD, shmD, s >>>(F, G);
}

void initS(double *const F, double *const G, double *const V, double *const S, double *const H, double *const K, const unsigned nRank, const cudaStream_t s) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  dInitS<<< gD, bD, shmD, s >>>(F, G, V, S, H, K);
}

void initS(double *const F, double *const G, double *const V, const unsigned nRank, const cudaStream_t s) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  dInitS<<< gD, bD, shmD, s >>>(F, G, V);
}

void initV(double *const F, double *const G, double *const V, const unsigned nRank, const cudaStream_t s) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  dInitV<<< gD, bD, shmD, s >>>(F, G, V);
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
 const unsigned nSwp,
 const cudaStream_t s
) throw()
{
  const size_t off = static_cast<size_t>(0u);
  CUDA_CALL(cudaMemcpyToSymbolAsync(_W, &W, sizeof(double*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_C, &C, sizeof(unsigned long long*), off, cudaMemcpyHostToDevice, s));

  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRowF, &nRowF, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRowG, &nRowG, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRowV, &nRowV, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRowW, &nRowW, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));

  CUDA_CALL(cudaMemcpyToSymbolAsync(_ldF, &ldF, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_ldG, &ldG, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_ldV, &ldV, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_ldW, &ldW, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));

  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRank, &nRank, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nSwp, &nSwp, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));

  CUDA_CALL(cudaMemcpyToSymbolAsync(_STRAT0_STEPS, &STRAT0_STEPS, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_STRAT0_PAIRS, &STRAT0_PAIRS, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));

  CUDA_CALL(cudaMemcpyToSymbolAsync(_strat0, strat0, sizeof(strat0), off, cudaMemcpyHostToDevice, s));
}
