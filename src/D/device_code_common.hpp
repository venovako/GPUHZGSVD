#ifndef DEVICE_CODE_COMMON_HPP
#define DEVICE_CODE_COMMON_HPP

#include "device_code_prof.hpp"
#include "device_code_common_defs.hpp"
#include "device_code_globals.hpp"

#include "device_code_common_rotate.hpp"
#include "device_code_common_Kepler.hpp"
#include "device_code_common_Cholesky.hpp"

MYDEVFN void dMultAV
(double *const __restrict__ A0,
 double *const __restrict__ A1,
 volatile double *const __restrict__ A,
 volatile const double *const __restrict__ B,
 const unsigned x,
 const unsigned y0,
 const unsigned y1,
 const unsigned m)
{
  // Cannon-like A*B
  for (unsigned i = x; i < m; i += 32u) {
    F32(A, x, y0) = A0[i];
    F32(A, x, y1) = A1[i];
    __syncthreads();

    double
      Cxy0 = +0.0,
      Cxy1 = +0.0;

    // skew (mod 32)
    unsigned
      p0 = ((y0 + x) & 0x1Fu),
      p1 = ((y1 + x) & 0x1Fu);

    // mult-and-cshift (mod 32)
    #pragma unroll
    for (unsigned k = 0u; k < 32u; ++k) {
      Cxy0 = _fma_rn(F32(A, x, p0), F32(B, p0, y0), Cxy0);
      Cxy1 = _fma_rn(F32(A, x, p1), F32(B, p1, y1), Cxy1);
      p0 = (p0 + 1u) & 0x1Fu;
      p1 = (p1 + 1u) & 0x1Fu;
    }
    __syncthreads();

    A0[i] = Cxy0;
    A1[i] = Cxy1;
    __syncthreads();
  }
}

#if ((CVG == 0) || (CVG == 2) || (CVG == 4) || (CVG == 6))
MYDEVFN double dSsqC
(const double *const __restrict__ bA,
 const double *const __restrict__ eA)
{
  double x = +0.0, y;
  for (const double *pA = bA; pA < eA; pA += WARP_SZ) {
    y = *pA;
    x = _fma_rn(y, y, x);
  }
  return dSum32(x);
}
#else /* ((CVG == 1) || (CVG == 3) || (CVG == 5) || (CVG == 7)) */
MYDEVFN double dSsqC
(const double *const __restrict__ bA,
 const double *const __restrict__ eA)
{
  double x = +0.0, xx = +0.0, y, yy;
  for (const double *pA = bA; pA < eA; pA += WARP_SZ) {
    y = *pA;
    yy = _dmul_rd(y, y);
    y = _fma_rn(y, y, -yy);
    xx = _dadd_rn(xx, yy);
    x = _dadd_rn(x, y);
  }
  yy = dSum32(xx);
  y = dSum32(x);
  return _dadd_rn(yy, y);
}
#endif /* ?CVG */

MYDEVFN void dInvNrm2C
(const double *const __restrict__ bA,
 const double *const __restrict__ eA,
 double &ssq,
 double &inv_nrm)
{
  ssq = dSsqC(bA, eA);
  inv_nrm = _drsqrt_rn(ssq);
}

MYDEVFN void dNrm2InvC
(const double *const __restrict__ bA,
 const double *const __restrict__ eA,
 double &ssq,
 double &nrm,
 double &inv_nrm)
{
  dInvNrm2C(bA, eA, ssq, inv_nrm);
  nrm = _dsqrt_rn(ssq);
}

MYDEVFN void dScalC
(double *const __restrict__ bA,
 const double *const __restrict__ eA,
 const double scl)
{
  for (double *pA = bA; pA < eA; pA += WARP_SZ)
    *pA = _dmul_rn(*pA, scl);
}

MYDEVFN void dGlobalPostScaleFast
(double *const __restrict__ F,
 double *const __restrict__ G,
 double *const __restrict__ V,
 const unsigned nRowF,
 const unsigned nRowG,
 const unsigned nRowV,
 const unsigned nRank,
 const unsigned ldF,
 const unsigned ldG,
 const unsigned ldV)
{
  const unsigned wpb = (blockDim.x + WARP_SZ_SUB1) >> WARP_SZ_LG;
  const unsigned wid = threadIdx.x >> WARP_SZ_LG;

  const unsigned cix = blockIdx.x * wpb + wid;
  if (cix < nRank) {
    const unsigned lid = static_cast<unsigned>(threadIdx.x) & WARP_SZ_SUB1;
    double *const bFi = F + (cix * ldF + lid);
    const double *const eFi = F + (cix * ldF + nRowF);
    double *const bGi = G + (cix * ldG + lid);
    const double *const eGi = G + (cix * ldG + nRowG);
    const double Fi_ssq = dSsqC(bFi, eFi);
    const double Gi_ssq = dSsqC(bGi, eGi);
    const double Rhyp = _drsqrt_rn(_dadd_rn(Fi_ssq, Gi_ssq));
    if (Rhyp != 1.0) {
      double *const bVi = V + (cix * ldV + lid);
      const double *const eVi = V + (cix * ldV + nRowV);
      dScalC(bVi, eVi, Rhyp);
    }
  }
}

MYDEVFN void dGlobalPostScaleFull
(double *const __restrict__ F,
 double *const __restrict__ G,
 double *const __restrict__ V,
 double *const __restrict__ S,
 double *const __restrict__ H,
 double *const __restrict__ K,
 const unsigned nRowF,
 const unsigned nRowG,
 const unsigned nRowV,
 const unsigned nRank,
 const unsigned ldF,
 const unsigned ldG,
 const unsigned ldV)
{
  const unsigned wpb = (blockDim.x + WARP_SZ_SUB1) >> WARP_SZ_LG;
  const unsigned wid = threadIdx.x >> WARP_SZ_LG;

  const unsigned cix = blockIdx.x * wpb + wid;
  if (cix < nRank) {
    const unsigned lid = static_cast<unsigned>(threadIdx.x) & WARP_SZ_SUB1;
    double *const bFi = F + (cix * ldF + lid);
    const double *const eFi = F + (cix * ldF + nRowF);
    double *const bGi = G + (cix * ldG + lid);
    const double *const eGi = G + (cix * ldG + nRowG);
    double Fi_ssq, Fi_nrm, Fi_inv_nrm;
    dNrm2InvC(bFi, eFi, Fi_ssq, Fi_nrm, Fi_inv_nrm);
    double Gi_ssq, Gi_nrm, Gi_inv_nrm;
    dNrm2InvC(bGi, eGi, Gi_ssq, Gi_nrm, Gi_inv_nrm); 
    double Sigmai = Fi_nrm;
    if (Fi_inv_nrm != 1.0)
      dScalC(bFi, eFi, Fi_inv_nrm);
    if (Gi_inv_nrm != 1.0) {
      dScalC(bGi, eGi, Gi_inv_nrm);
      Sigmai = _dmul_rn(Sigmai, Gi_inv_nrm);
    }
    double Hi = Fi_nrm;
    double Ki = Gi_nrm;
    const double Rhyp = _drsqrt_rn(_dadd_rn(Fi_ssq, Gi_ssq));
    if (Rhyp != 1.0) {
      Hi = _dmul_rn(Hi, Rhyp);
      Ki = _dmul_rn(Ki, Rhyp);
      double *const bVi = V + (cix * ldV + lid);
      const double *const eVi = V + (cix * ldV + nRowV);
      dScalC(bVi, eVi, Rhyp);
    }
    if (!lid) {
      S[cix] = Sigmai;
      H[cix] = Hi;
      K[cix] = Ki;
    }
  }
}

MYKERN dInitS(const int full)
{
  if (full)
    dGlobalPostScaleFull(_F, _G, _V, _S, _H, _K, _nRowF, _nRowG, _nRowV, _nRank, _ldF, _ldG, _ldV);
  else
    dGlobalPostScaleFast(_F, _G, _V, _nRowF, _nRowG, _nRank, _nRowV, _ldF, _ldG, _ldV);
}

MYDEVFN void dGlobalInitV
(double *const __restrict__ V,
 const unsigned nRank,
 const unsigned ldV
#ifdef USE_MPI
 , const unsigned ifc0
 , const unsigned ifc1
#endif /* USE_MPI */
) {
  const unsigned wpb = (blockDim.x + WARP_SZ_SUB1) >> WARP_SZ_LG;
  const unsigned wid = threadIdx.x >> WARP_SZ_LG;

  const unsigned cix = blockIdx.x * wpb + wid;
  if (cix < nRank) {
    const unsigned lid = static_cast<unsigned>(threadIdx.x) & WARP_SZ_SUB1;
    if (!lid) {
#ifdef USE_MPI
      const unsigned nRank_2 = nRank >> 1u;
      if (cix < nRank_2)
        V[cix * ldV + (cix + ifc0)] = 1.0;
      else
        V[cix * ldV + ((cix - nRank_2) + ifc1)] = 1.0;
#else /* !USE_MPI */
      V[cix * ldV + cix] = 1.0;
#endif /* ?USE_MPI */
    }
  }
}

MYDEVFN void dGlobalInitVscl
(double *const __restrict__ F,
 double *const __restrict__ G,
 double *const __restrict__ V,
 const unsigned nRowF,
 const unsigned nRowG,
 const unsigned nRank,
 const unsigned ldF,
 const unsigned ldG,
 const unsigned ldV
#ifdef USE_MPI
 , const unsigned ifc0
 , const unsigned ifc1
#endif /* USE_MPI */
) {
  const unsigned wpb = (blockDim.x + WARP_SZ_SUB1) >> WARP_SZ_LG;
  const unsigned wid = threadIdx.x >> WARP_SZ_LG;

  const unsigned cix = blockIdx.x * wpb + wid;
  if (cix < nRank) {
    const unsigned lid = static_cast<unsigned>(threadIdx.x) & WARP_SZ_SUB1;
    double *const bGi = G + (cix * ldG + lid);
    const double *const eGi = G + (cix * ldG + nRowG);
    double Gi_ssq, Gi_inv_nrm;
    dInvNrm2C(bGi, eGi, Gi_ssq, Gi_inv_nrm);
    if (Gi_inv_nrm != 1.0) {
      double *const bFi = F + (cix * ldF + lid);
      const double *const eFi = F + (cix * ldF + nRowF);
      dScalC(bFi, eFi, Gi_inv_nrm);
      dScalC(bGi, eGi, Gi_inv_nrm);
    }
    if (!lid) {
#ifdef USE_MPI
      const unsigned nRank_2 = nRank >> 1u;
      if (cix < nRank_2)
        V[cix * ldV + (cix + ifc0)] = Gi_inv_nrm;
      else
        V[cix * ldV + ((cix - nRank_2) + ifc1)] = Gi_inv_nrm;
#else /* !USE_MPI */
      V[cix * ldV + cix] = Gi_inv_nrm;
#endif /* ?USE_MPI */
    }
  }
}

MYKERN dInitV(const int sclV
#ifdef USE_MPI
  , const unsigned ifc0, const unsigned ifc1
#endif /* USE_MPI */
) {
  if (sclV)
    dGlobalInitVscl(_F, _G, _V, _nRowF, _nRowG, _nRank, _ldF, _ldG, _ldV
#ifdef USE_MPI
      , ifc0, ifc1
#endif /* USE_MPI */
    );
  else
    dGlobalInitV(_V, _nRank, _ldV
#ifdef USE_MPI
      , ifc0, ifc1
#endif /* USE_MPI */
    );
}

#endif /* !DEVICE_CODE_COMMON_HPP */
