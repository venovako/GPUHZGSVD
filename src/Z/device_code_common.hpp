#ifndef DEVICE_CODE_COMMON_HPP
#define DEVICE_CODE_COMMON_HPP

#include "device_code_prof.hpp"
#include "device_code_common_defs.hpp"
#include "device_code_globals.hpp"
#include "cuZ.hpp"
#include "device_code_common_rotate.hpp"
#include "device_code_common_Kepler.hpp"
#include "device_code_common_Cholesky.hpp"

MYDEVFN void zMultAV
(cuD *const __restrict__ A0D, cuJ *const __restrict__ A0J,
 cuD *const __restrict__ A1D, cuJ *const __restrict__ A1J,
 volatile cuD *const __restrict__ AD, volatile cuJ *const __restrict__ AJ,
 volatile const cuD *const __restrict__ BD, volatile const cuJ *const __restrict__ BJ,
 const unsigned x,
 const unsigned y0,
 const unsigned y1,
 const unsigned m)
{
  // Cannon-like A*B
  for (unsigned i = x; i < m; i += 32u) {
    F32(AD, x, y0) = A0D[i];
    F32(AJ, x, y0) = A0J[i];
    F32(AD, x, y1) = A1D[i];
    F32(AJ, x, y1) = A1J[i];
    __syncthreads();

    cuD
      Cxy0D = +0.0,
      Cxy1D = +0.0;
    cuJ
      Cxy0J = +0.0,
      Cxy1J = +0.0;

    // skew (mod 32)
    unsigned
      p0 = ((y0 + x) & 0x1Fu),
      p1 = ((y1 + x) & 0x1Fu);

    // mult-and-cshift (mod 32)
    #pragma unroll
    for (unsigned k = 0u; k < 32u; ++k) {
      Zfma(Cxy0D, Cxy0J, F32(AD, x, p0), F32(AJ, x, p0), F32(BD, p0, y0), F32(BJ, p0, y0), Cxy0D, Cxy0J);
      Zfma(Cxy1D, Cxy1J, F32(AD, x, p1), F32(AJ, x, p1), F32(BD, p1, y1), F32(BJ, p1, y1), Cxy1D, Cxy1J);
      p0 = (p0 + 1u) & 0x1Fu;
      p1 = (p1 + 1u) & 0x1Fu;
    }
    __syncthreads();

    A0D[i] = Cxy0D;
    A0J[i] = Cxy0J;
    A1D[i] = Cxy1D;
    A1J[i] = Cxy1J;
    __syncthreads();
  }
}

#if ((CVG == 0) || (CVG == 2) || (CVG == 4) || (CVG == 6))
MYDEVFN double zSsqC
(const cuD *const __restrict__ bAD, const cuD *const __restrict__ eAD,
 const cuJ *const __restrict__ bAJ)
{
  cuD re = +0.0;
  cuJ im = +0.0;
  const cuD *pAD = bAD;
  const cuJ *pAJ = bAJ;
  while (pAD < eAD) {
    const cuD x = *pAD;
    const cuJ y = *pAJ;
    re = _fma_rn(x, x, re);
    pAD += WARP_SZ;
    im = _fma_rn(y, y, im);
    pAJ += WARP_SZ;
  }
  const double z = _dadd_rn(re, im);
  return dSum32(z);
}
#else /* ((CVG == 1) || (CVG == 3) || (CVG == 5) || (CVG == 7)) */
MYDEVFN double zSsqC
(const cuD *const __restrict__ bAD, const cuD *const __restrict__ eAD,
 const cuJ *const __restrict__ bAJ)
{
  cuD rev = +0.0, ree = +0.0;
  cuJ imv = +0.0, ime = +0.0;
  cuD re0, re1, re2;
  cuJ im0, im1, im2;
  const cuD *pAD = bAD;
  const cuJ *pAJ = bAJ;
  while (pAD < eAD) {
    const cuD x = *pAD;
    pAD += WARP_SZ;
    const cuJ y = *pAJ;
    pAJ += WARP_SZ;
    re1 = x;
    im1 = y;
    re2 = _dmul_rd(re1, re1);
    re0 = _fma_rn(re1, re1, -re2);
    im2 = _dmul_rd(im1, im1);
    im0 = _fma_rn(im1, im1, -im2);
    rev = _dadd_rn(rev, re2);
    imv = _dadd_rn(imv, im2);
    ree = _dadd_rn(ree, re0);
    ime = _dadd_rn(ime, im0);
  }
  re0 = _dadd_rn(ree, ime);
  if (rev <= imv) {
    re1 = _dadd_rn(re0, rev);
    re2 = _dadd_rn(re1, imv);
  }
  else {
    re1 = _dadd_rn(re0, imv);
    re2 = _dadd_rn(re1, rev);
  }
  return dSum32(re2);
}
#endif /* ?CVG */

MYDEVFN void zInvNrm2C
(const cuD *const __restrict__ bAD, const cuD *const __restrict__ eAD,
 const cuJ *const __restrict__ bAJ,
 double &ssq,
 double &inv_nrm)
{
  ssq = zSsqC(bAD, eAD, bAJ);
  inv_nrm = _drsqrt_rn(ssq);
}

MYDEVFN void zNrm2InvC
(const cuD *const __restrict__ bAD, const cuD *const __restrict__ eAD,
 const cuJ *const __restrict__ bAJ,
 double &ssq,
 double &nrm,
 double &inv_nrm)
{
  zInvNrm2C(bAD, eAD, bAJ, ssq, inv_nrm);
  nrm = _dsqrt_rn(ssq);
}

MYDEVFN void zScalC
(cuD *const __restrict__ bAD, const cuD *const __restrict__ eAD,
 cuJ *const __restrict__ bAJ,
 const double scl)
{
  cuD *pAD = bAD;
  cuJ *pAJ = bAJ;
  while (pAD < eAD) {
    *pAD = _dmul_rn(*pAD, scl);
    pAD += WARP_SZ;
    *pAJ = _dmul_rn(*pAJ, scl);
    pAJ += WARP_SZ;
  }
}

MYDEVFN void zGlobalPostScaleFast
(cuD *const __restrict__ FD, cuJ *const __restrict__ FJ,
 cuD *const __restrict__ GD, cuJ *const __restrict__ GJ,
 cuD *const __restrict__ VD, cuJ *const __restrict__ VJ,
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
    unsigned off = cix * ldF + lid;
    cuD *const bFiD = FD + off;
    cuJ *const bFiJ = FJ + off;
    off = cix * ldF + nRowF;
    const cuD *const eFiD = FD + off;
    off = cix * ldG + lid;
    cuD *const bGiD = GD + off;
    cuJ *const bGiJ = GJ + off;
    off = cix * ldG + nRowG;
    const cuD *const eGiD = GD + off;
    const double Fi_ssq = zSsqC(bFiD, eFiD, bFiJ);
    const double Gi_ssq = zSsqC(bGiD, eGiD, bGiJ);
    const double Rhyp = _drsqrt_rn(_dadd_rn(Fi_ssq, Gi_ssq));
    if (Rhyp != 1.0) {
      off = cix * ldV + lid;
      cuD *const bViD = VD + off;
      cuJ *const bViJ = VJ + off;
      off = cix * ldV + nRowV;
      const cuD *const eViD = VD + off;
      zScalC(bViD, eViD, bViJ, Rhyp);
    }
  }
}

MYDEVFN void zGlobalPostScaleFull
(cuD *const __restrict__ FD, cuJ *const __restrict__ FJ,
 cuD *const __restrict__ GD, cuJ *const __restrict__ GJ,
 cuD *const __restrict__ VD, cuJ *const __restrict__ VJ,
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
    unsigned off = cix * ldF + lid;
    cuD *const bFiD = FD + off;
    cuJ *const bFiJ = FJ + off;
    off = cix * ldF + nRowF;
    const cuD *const eFiD = FD + off;
    off = cix * ldG + lid;
    cuD *const bGiD = GD + off;
    cuJ *const bGiJ = GJ + off;
    off = cix * ldG + nRowG;
    const cuD *const eGiD = GD + off;
    double Fi_ssq, Fi_nrm, Fi_inv_nrm;
    zNrm2InvC(bFiD, eFiD, bFiJ, Fi_ssq, Fi_nrm, Fi_inv_nrm);
    double Gi_ssq, Gi_nrm, Gi_inv_nrm;
    zNrm2InvC(bGiD, eGiD, bGiJ, Gi_ssq, Gi_nrm, Gi_inv_nrm); 
    double Sigmai = Fi_nrm;
    if (Fi_inv_nrm != 1.0)
      zScalC(bFiD, eFiD, bFiJ, Fi_inv_nrm);
    if (Gi_inv_nrm != 1.0) {
      zScalC(bGiD, eGiD, bGiJ, Gi_inv_nrm);
      Sigmai = _dmul_rn(Sigmai, Gi_inv_nrm);
    }
    double Hi = Fi_nrm;
    double Ki = Gi_nrm;
    const double Rhyp = _drsqrt_rn(_dadd_rn(Fi_ssq, Gi_ssq));
    if (Rhyp != 1.0) {
      Hi = _dmul_rn(Hi, Rhyp);
      Ki = _dmul_rn(Ki, Rhyp);
      off = cix * ldV + lid;
      cuD *const bViD = VD + off;
      cuJ *const bViJ = VJ + off;
      off = cix * ldV + nRowV;
      const cuD *const eViD = VD + off;
      zScalC(bViD, eViD, bViJ, Rhyp);
    }
    if (!lid) {
      S[cix] = Sigmai;
      H[cix] = Hi;
      K[cix] = Ki;
    }
  }
}

MYKERN zInitS(const int full)
{
  if (full)
    zGlobalPostScaleFull(_FD, _FJ, _GD, _GJ, _VD, _VJ, _S, _H, _K, _nRowF, _nRowG, _nRowV, _nRank, _ldF, _ldG, _ldV);
  else
    zGlobalPostScaleFast(_FD, _FJ, _GD, _GJ, _VD, _VJ, _nRowF, _nRowG, _nRowV, _nRank, _ldF, _ldG, _ldV);
}

MYDEVFN void zGlobalInitV
(cuD *const __restrict__ VD,
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
        VD[cix * ldV + (cix + ifc0)] = 1.0;
      else
        VD[cix * ldV + ((cix - nRank_2) + ifc1)] = 1.0;
#else /* !USE_MPI */
      VD[cix * ldV + cix] = 1.0;
#endif /* ?USE_MPI */
    }
  }
}

MYDEVFN void zGlobalInitVscl
(cuD *const __restrict__ FD, cuJ *const __restrict__ FJ,
 cuD *const __restrict__ GD, cuJ *const __restrict__ GJ,
 cuD *const __restrict__ VD,
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
    unsigned off = cix * ldG + lid;
    cuD *const bGiD = GD + off;
    cuJ *const bGiJ = GJ + off;
    off = cix * ldG + nRowG;
    const cuD *const eGiD = GD + off;
    double Gi_ssq, Gi_inv_nrm;
    zInvNrm2C(bGiD, eGiD, bGiJ, Gi_ssq, Gi_inv_nrm);
    if (Gi_inv_nrm != 1.0) {
      off = cix * ldF + lid;
      cuD *const bFiD = FD + off;
      cuJ *const bFiJ = FJ + off;
      off = cix * ldF + nRowF;
      const cuD *const eFiD = FD + off;
      zScalC(bFiD, eFiD, bFiJ, Gi_inv_nrm);
      zScalC(bGiD, eGiD, bGiJ, Gi_inv_nrm);
    }
    if (!lid) {
#ifdef USE_MPI
      const unsigned nRank_2 = nRank >> 1u;
      if (cix < nRank_2)
        VD[cix * ldV + (cix + ifc0)] = Gi_inv_nrm;
      else
        VD[cix * ldV + ((cix - nRank_2) + ifc1)] = Gi_inv_nrm;
#else /* !USE_MPI */
      VD[cix * ldV + cix] = Gi_inv_nrm;
#endif /* ?USE_MPI */
    }
  }
}

MYKERN zInitV(const int sclV
#ifdef USE_MPI
  , const unsigned ifc0, const unsigned ifc1
#endif /* USE_MPI */
) {
  if (sclV)
    zGlobalInitVscl(_FD, _FJ, _GD, _GJ, _VD, _nRowF, _nRowG, _nRank, _ldF, _ldG, _ldV
#ifdef USE_MPI
      , ifc0, ifc1
#endif /* USE_MPI */
    );
  else
    zGlobalInitV(_VD, _nRank, _ldV
#ifdef USE_MPI
      , ifc0, ifc1
#endif /* USE_MPI */
    );
}

#endif /* !DEVICE_CODE_COMMON_HPP */
