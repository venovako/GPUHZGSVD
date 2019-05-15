#ifndef DEVICE_CODE_COMMON_HPP
#define DEVICE_CODE_COMMON_HPP

#ifndef MYKERN
#define MYKERN __global__ void
#else // MYKERN
#error MYKERN not definable externally
#endif // !MYKERN

#ifndef MYDEVFN
#ifdef NDEBUG
#define MYDEVFN __device__ __forceinline__
#else // DEBUG
#define MYDEVFN __device__
#endif // NDEBUG
#else // MYDEVFN
#error MYDEVFN not definable externally
#endif // !MYDEVFN

#ifndef HZ_L1_MAX_THREADS_PER_BLOCK
#define HZ_L1_MAX_THREADS_PER_BLOCK 512
#else // HZ_L1_MAX_THREADS_PER_BLOCK
#error HZ_L1_MAX_THREADS_PER_BLOCK not definable externally
#endif // !HZ_L1_MAX_THREADS_PER_BLOCK

#ifndef HZ_L1_THREADS_PER_BLOCK_X
#define HZ_L1_THREADS_PER_BLOCK_X 32u
#else // HZ_L1_THREADS_PER_BLOCK_X
#error HZ_L1_THREADS_PER_BLOCK_X not definable externally
#endif // !HZ_L1_THREADS_PER_BLOCK_X

#ifndef HZ_L1_THREADS_PER_BLOCK_Y
#define HZ_L1_THREADS_PER_BLOCK_Y 16u
#else // HZ_L1_THREADS_PER_BLOCK_Y
#error HZ_L1_THREADS_PER_BLOCK_Y not definable externally
#endif // !HZ_L1_THREADS_PER_BLOCK_Y

#ifndef HZ_L1_MIN_BLOCKS_PER_SM
#define HZ_L1_MIN_BLOCKS_PER_SM 1
#else // HZ_L1_MIN_BLOCKS_PER_SM
#error HZ_L1_MIN_BLOCKS_PER_SM not definable externally
#endif // !HZ_L1_MIN_BLOCKS_PER_SM

#if (WARP_SZ != 32u)
#error WARP_SZ not 32
#endif // WARP_SZ

#ifndef WARP_SZ_LG
#define WARP_SZ_LG 5u
#else // WARP_SZ_LG
#error WARP_SZ_LG not definable externally
#endif // !WARP_SZ_LG

#ifndef WARP_SZ_LGi
#define WARP_SZ_LGi 5
#else // WARP_SZ_LGi
#error WARP_SZ_LGi not definable externally
#endif // !WARP_SZ_LGi

#ifndef WARP_SZ_SUB1
#define WARP_SZ_SUB1 31u
#else // WARP_SZ_SUB1
#error WARP_SZ_SUB1 not definable externally
#endif // !WARP_SZ_SUB1

#ifndef INFTY
#define INFTY CUDART_INF
#endif // !INFTY

#ifndef F32
#define F32(A, r, c) ((A)[(c) * 32u + (r)])
#else // F32
#error F32 not definable externally
#endif // !F32

#ifndef F64
#define F64(A, r, c) ((A)[(c) * 64u + (r)])
#else // F64
#error F64 not definable externally
#endif // !F64

#include "cuZ.hpp"
#include "device_code_globals.hpp"

// Thanks to Norbert Juffa of NVIDIA.
MYDEVFN double
my_drsqrt_rn(double a)
{
  double y, h, l, e;
  unsigned int ilo, ihi, g, f;
  int d;

  ihi = __double2hiint(a);
  ilo = __double2loint(a);
  if (((unsigned int)ihi) - 0x00100000U < 0x7fe00000U) {
    f = ihi | 0x3fe00000;
    g = f & 0x3fffffff;
    d = g - ihi;
    a = __hiloint2double(g, ilo); 
    y = rsqrt(a);
    h = __dmul_rn(y, y);
    l = __fma_rn(y, y, -h);
    e = __fma_rn(l, -a, __fma_rn(h, -a, 1.0));
    /* Round as shown in Peter Markstein, "IA-64 and Elementary Functions" */
    y = __fma_rn(__fma_rn(0.375, e, 0.5), e * y, y);
    d = d >> 1;
    a = __hiloint2double(__double2hiint(y) + d, __double2loint(y));
  } else if (a == 0.0) {
    a = __hiloint2double((ihi & 0x80000000) | 0x7ff00000, 0x00000000);
  } else if (a < 0.0) {
    a = __hiloint2double(0xfff80000, 0x00000000);
  } else if (isinf(a)) {
    a = __hiloint2double(ihi & 0x80000000, 0x00000000);
  } else if (isnan(a)) {
    a = a + a;
  } else {
    a = a * __hiloint2double(0x7fd00000, 0);
    y = rsqrt(a);
    h = __dmul_rn(y, y);
    l = __fma_rn(y, y, -h);
    e = __fma_rn(l, -a, __fma_rn(h, -a, 1.0));
    /* Round as shown in Peter Markstein, "IA-64 and Elementary Functions" */
    y = __fma_rn(__fma_rn(0.375, e, 0.5), e * y, y);
    a = __hiloint2double(__double2hiint(y) + 0x1ff00000,__double2loint(y));
  }
  return a;
}

#include "device_code_common_rotate.hpp"
#include "device_code_common_Kepler.hpp"
#include "device_code_common_Cholesky.hpp"

MYDEVFN void zMultAV
(cuD *const A0D, cuJ *const A0J,
 cuD *const A1D, cuJ *const A1J,
 volatile cuD *const AD, volatile cuJ *const AJ,
 volatile const cuD *const BD, volatile const cuJ *const BJ,
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
(const cuD *const bAD, const cuD *const eAD,
 const cuJ *const bAJ)
{
  cuD re = +0.0;
  cuJ im = +0.0;
  const cuD *pAD = bAD;
  const cuJ *pAJ = bAJ;
  while (pAD < eAD) {
    const cuD x = *pAD;
    const cuJ y = *pAJ;
    re = __fma_rn(x, x, re);
    pAD += WARP_SZ;
    im = __fma_rn(y, y, im);
    pAJ += WARP_SZ;
  }
  const double z = re + im;
  return dSum32(z);
}
#else // ((CVG == 1) || (CVG == 3) || (CVG == 5) || (CVG == 7))
MYDEVFN double zSsqC
(const cuD *const bAD, const cuD *const eAD,
 const cuJ *const bAJ)
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
    re2 = __dmul_rd(re1, re1);
    re0 = __fma_rn(re1, re1, -re2);
    im2 = __dmul_rd(im1, im1);
    im0 = __fma_rn(im1, im1, -im2);
    rev += re2;
    imv += im2;
    ree += re0;
    ime += im0;
  }
  re0 = ree + ime;
  if (rev <= imv) {
    re1 = re0 + rev;
    re2 = re1 + imv;
  }
  else {
    re1 = re0 + imv;
    re2 = re1 + rev;
  }
  return dSum32(re2);
}
#endif // ?CVG

MYDEVFN void zInvNrm2C
(const cuD *const bAD, const cuD *const eAD,
 const cuJ *const bAJ,
 double &ssq,
 double &inv_nrm)
{
  ssq = zSsqC(bAD, eAD, bAJ);
  inv_nrm = my_drsqrt_rn(ssq);
}

MYDEVFN void zNrm2InvC
(const cuD *const bAD, const cuD *const eAD,
 const cuJ *const bAJ,
 double &ssq,
 double &nrm,
 double &inv_nrm)
{
  zInvNrm2C(bAD, eAD, bAJ, ssq, inv_nrm);
  nrm = __dsqrt_rn(ssq);
}

MYDEVFN void zScalC
(cuD *const bAD, const cuD *const eAD,
 cuJ *const bAJ,
 const double scl)
{
  cuD *pAD = bAD;
  cuJ *pAJ = bAJ;
  while (pAD < eAD) {
    *pAD *= scl;
    pAD += WARP_SZ;
    *pAJ *= scl;
    pAJ += WARP_SZ;
  }
}

MYDEVFN void zGlobalPostScaleFast
(cuD *const FD, cuJ *const FJ,
 cuD *const GD, cuJ *const GJ,
 cuD *const VD, cuJ *const VJ,
 double *const S,
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
    const double Rhyp = my_drsqrt_rn(Fi_ssq + Gi_ssq);
    if (Rhyp != 1.0) {
      off = cix * ldV + lid;
      cuD *const bViD = VD + off;
      cuJ *const bViJ = VJ + off;
      off = cix * ldV + nRank;
      const cuD *const eViD = VD + off;
      zScalC(bViD, eViD, bViJ, Rhyp);
    }
    if (!lid)
      S[cix] = +0.0;
  }
}

MYDEVFN void zGlobalPostScaleFull
(cuD *const FD, cuJ *const FJ,
 cuD *const GD, cuJ *const GJ,
 cuD *const VD, cuJ *const VJ,
 double *const S,
 double *const H,
 double *const K,
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
      Sigmai *= Gi_inv_nrm;
    }
    double Hi = Fi_nrm;
    double Ki = Gi_nrm;
    const double Rhyp = my_drsqrt_rn(Fi_ssq + Gi_ssq);
    if (Rhyp != 1.0) {
      Hi *= Rhyp;
      Ki *= Rhyp;
      off = cix * ldV + lid;
      cuD *const bViD = VD + off;
      cuJ *const bViJ = VJ + off;
      off = cix * ldV + nRank;
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
    zGlobalPostScaleFull(_FD, _FJ, _GD, _GJ, _VD, _VJ, _S, _H, _K, _nRowF, _nRowG, _nRank, _ldF, _ldG, _ldV);
  else
    zGlobalPostScaleFast(_FD, _FJ, _GD, _GJ, _VD, _VJ, _S, _nRowF, _nRowG, _nRank, _ldF, _ldG, _ldV);
}

MYDEVFN void zGlobalInitV
(cuD *const VD,
 const unsigned nRank,
 const unsigned ldV)
{
  const unsigned wpb = (blockDim.x + WARP_SZ_SUB1) >> WARP_SZ_LG;
  const unsigned wid = threadIdx.x >> WARP_SZ_LG;

  const unsigned cix = blockIdx.x * wpb + wid;
  if (cix < nRank) {
    const unsigned lid = static_cast<unsigned>(threadIdx.x) & WARP_SZ_SUB1;
    if (!lid)
      VD[cix * ldV + cix] = 1.0;
  }
}

MYDEVFN void zGlobalInitVscl
(cuD *const FD, cuJ *const FJ,
 cuD *const GD, cuJ *const GJ,
 cuD *const VD,
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
    if (!lid)
      VD[cix * ldV + cix] = Gi_inv_nrm;
  }
}

MYKERN zInitV(const int sclV)
{
  if (sclV)
    zGlobalInitVscl(_FD, _FJ, _GD, _GJ, _VD, _nRowF, _nRowG, _nRank, _ldF, _ldG, _ldV);
  else
    zGlobalInitV(_VD, _nRank, _ldV);
}

#endif // !DEVICE_CODE_COMMON_HPP
