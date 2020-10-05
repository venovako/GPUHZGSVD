#include "device_code.hpp"

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "cuda_helper.hpp"

#include "device_code_common.hpp"
#include "device_code_accumV.hpp"
#if ((CVG == 0) || (CVG == 1))
#include "device_code_cdsort_0.hpp"
#include "device_code_nosort_0.hpp"
#elif ((CVG == 2) || (CVG == 3))
#include "device_code_cdsort_1.hpp"
#include "device_code_nosort_1.hpp"
#elif ((CVG == 4) || (CVG == 5))
#include "device_code_cdsort_2.hpp"
#include "device_code_nosort_2.hpp"
#elif ((CVG == 6) || (CVG == 7))
#include "device_code_cdsort_3.hpp"
#include "device_code_nosort_3.hpp"
#else /* unknown CVG */
#error CVG unknown
#endif /* ?CVG */
#include "device_code_cdsort_accumV.hpp"
#include "device_code_nosort_accumV.hpp"

static const dim3 hzL1bD(HZ_L1_THREADS_PER_BLOCK_X, HZ_L1_THREADS_PER_BLOCK_Y, 1u);

void HZ_L1_sv(const unsigned step, const cudaStream_t s) throw()
{
  const dim3 hzL1gD(STRAT1_PAIRS, 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  zHZ_L1_sv<<< hzL1gD, hzL1bD, shmD, s >>>(step);
}

void HZ_L1_v(const unsigned step, const cudaStream_t s) throw()
{
  const dim3 hzL1gD(STRAT1_PAIRS, 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  zHZ_L1_v<<< hzL1gD, hzL1bD, shmD, s >>>(step);
}

void initS(const int full, const unsigned nRank, const cudaStream_t s) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  zInitS<<< gD, bD, shmD, s >>>(full);
}

void initV(const int sclV, const unsigned nRank
#ifdef USE_MPI
  , const unsigned ifc0, const unsigned ifc1
#endif /* USE_MPI */
  , const cudaStream_t s
) throw()
{
  const dim3 bD(2u * WARP_SZ, 1u, 1u);
  const dim3 gD(udiv_ceil(nRank * WARP_SZ, bD.x), 1u, 1u);
  const size_t shmD = static_cast<size_t>(0u);
  zInitV<<< gD, bD, shmD, s >>>(sclV
#ifdef USE_MPI
    , ifc0, ifc1
#endif /* USE_MPI */
  );
}

void initSymbols
(cuD *const FD, cuJ *const FJ,
 cuD *const GD, cuJ *const GJ,
 cuD *const VD, cuJ *const VJ,
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
 const unsigned nSwp,
 const cudaStream_t s
) throw()
{
  const size_t off = static_cast<size_t>(0u);
  CUDA_CALL(cudaMemcpyToSymbolAsync(_FD, &FD, sizeof(cuD*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_FJ, &FJ, sizeof(cuJ*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_GD, &GD, sizeof(cuD*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_GJ, &GJ, sizeof(cuJ*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_VD, &VD, sizeof(cuD*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_VJ, &VJ, sizeof(cuJ*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_S, &S, sizeof(double*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_H, &H, sizeof(double*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_K, &K, sizeof(double*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_C, &C, sizeof(unsigned long long*), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRowF, &nRowF, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRowG, &nRowG, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRowV, &nRowV, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nRank, &nRank, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_ldF, &ldF, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_ldG, &ldG, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_ldV, &ldV, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_nSwp, &nSwp, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_STRAT0_STEPS, &STRAT0_STEPS, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_STRAT0_PAIRS, &STRAT0_PAIRS, sizeof(unsigned), off, cudaMemcpyHostToDevice, s));
  // copy strategy tables
  CUDA_CALL(cudaMemcpyToSymbolAsync(_strat0, strat0, sizeof(strat0), off, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpyToSymbolAsync(_strat1, strat1, sizeof(strat1), off, cudaMemcpyHostToDevice, s));
}
