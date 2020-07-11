#ifndef DEVICE_CODE_COMMON_HPP
#define DEVICE_CODE_COMMON_HPP

#include "device_code_prof.hpp"
#include "device_code_common_defs.hpp"
#include "device_code_common_globals.hpp"

#include "device_code_common_rotate.hpp"
#include "device_code_common_Kepler.hpp"
#include "device_code_common_Cholesky.hpp"

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

MYKERN dInitS
(double *const __restrict__ _F,
 double *const __restrict__ _G,
 double *const __restrict__ _V,
 double *const __restrict__ _S,
 double *const __restrict__ _H,
 double *const __restrict__ _K)
{
  dGlobalPostScaleFull(_F, _G, _V, _S, _H, _K, _nRowF, _nRowG, _nRowV, _nRank, _ldF, _ldG, _ldV);
}

MYKERN dInitS
(double *const __restrict__ _F,
 double *const __restrict__ _G,
 double *const __restrict__ _V)
{
  dGlobalPostScaleFast(_F, _G, _V, _nRowF, _nRowG, _nRank, _nRowV, _ldF, _ldG, _ldV);
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
 const unsigned ldV)
{
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
    if (!lid)
      V[cix * ldV + cix] = Gi_inv_nrm;
  }
}

MYKERN dInitV
(double *const __restrict__ _F,
 double *const __restrict__ _G,
 double *const __restrict__ _V)
{
  dGlobalInitVscl(_F, _G, _V, _nRowF, _nRowG, _nRank, _ldF, _ldG, _ldV);
}

#endif /* !DEVICE_CODE_COMMON_HPP */
