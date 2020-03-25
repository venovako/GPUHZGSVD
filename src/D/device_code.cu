#include "device_code.hpp"

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "cuda_helper.hpp"

#include "device_code_common.hpp"
#include "device_code_accumV.hpp"
#if (CVG == 0)
#include "device_code_cdsort_0.hpp"
#elif (CVG == 1)
#include "device_code_cdsort_1.hpp"
#elif (CVG == 2)
#include "device_code_cdsort_2.hpp"
#elif (CVG == 3)
#include "device_code_cdsort_3.hpp"
#elif (CVG == 4)
#include "device_code_cdsort_4.hpp"
#elif (CVG == 5)
#include "device_code_cdsort_5.hpp"
#elif (CVG == 6)
#include "device_code_cdsort_6.hpp"
#elif (CVG == 7)
#include "device_code_cdsort_7.hpp"
#else // unknown CVG
#error CVG unknown
#endif // ?CVG
#include "device_code_cdsort_accumV.hpp"

static const dim3 hzL1bD(HZ_L1_THREADS_PER_BLOCK_X, HZ_L1_THREADS_PER_BLOCK_Y, 1u);

void HZ_L1_sv(const unsigned step) throw()
{
  const dim3 hzL1gD(STRAT1_PAIRS, 1u, 1u);
  cuda_prof_start();
  dHZ_L1_sv<<< hzL1gD, hzL1bD >>>(step);
  cuda_prof_stop();
}

void initS(const int full, const unsigned nRank) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  cuda_prof_start();
  dInitS<<< gD, bD, shmD >>>(full);
  cuda_prof_stop();
}

void initV(const int sclV, const unsigned nRank
#ifdef USE_MPI
  , const unsigned ifc0, const unsigned ifc1
#endif // USE_MPI
) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  cuda_prof_start();
  dInitV<<< gD, bD, shmD >>>(sclV
#ifdef USE_MPI
    , ifc0, ifc1
#endif // USE_MPI
  );
  cuda_prof_stop();
}

void initSymbols
(double *const F,
 double *const G,
 double *const V,
 double *const S,
 double *const H,
 double *const K,
 unsigned long long *const C, 
 const unsigned nRowF,
 const unsigned nRowG,
 const unsigned nRowV,
 const unsigned nRank,
 const unsigned ldF,
 const unsigned ldG,
 const unsigned ldV,
 const unsigned nSwp
) throw()
{
  CUDA_CALL(cudaMemcpyToSymbol(_F, &F, sizeof(double*)));
  CUDA_CALL(cudaMemcpyToSymbol(_G, &G, sizeof(double*)));
  CUDA_CALL(cudaMemcpyToSymbol(_V, &V, sizeof(double*)));
  CUDA_CALL(cudaMemcpyToSymbol(_S, &S, sizeof(double*)));
  CUDA_CALL(cudaMemcpyToSymbol(_H, &H, sizeof(double*)));
  CUDA_CALL(cudaMemcpyToSymbol(_K, &K, sizeof(double*)));
  CUDA_CALL(cudaMemcpyToSymbol(_C, &C, sizeof(unsigned long long*)));
  CUDA_CALL(cudaMemcpyToSymbol(_nRowF, &nRowF, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_nRowG, &nRowG, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_nRowV, &nRowV, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_nRank, &nRank, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_ldF, &ldF, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_ldG, &ldG, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_ldV, &ldV, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_nSwp, &nSwp, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_STRAT0_STEPS, &STRAT0_STEPS, sizeof(unsigned)));
  CUDA_CALL(cudaMemcpyToSymbol(_STRAT0_PAIRS, &STRAT0_PAIRS, sizeof(unsigned)));
  // copy strategy tables
  CUDA_CALL(cudaMemcpyToSymbol(_strat0, strat0, sizeof(strat0)));
  CUDA_CALL(cudaMemcpyToSymbol(_strat1, strat1, sizeof(strat1)));
}
