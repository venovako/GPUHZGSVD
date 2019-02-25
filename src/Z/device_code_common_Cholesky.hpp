#ifndef DEVICE_CODE_COMMON_CHOLESKY_HPP
#define DEVICE_CODE_COMMON_CHOLESKY_HPP

MYDEVFN void zAhA
(const cuD *const A0D, const cuJ *const A0J,
 const cuD *const A1D, const cuJ *const A1J,
 volatile cuD *const AD, volatile cuJ *const AJ,
 const unsigned x,
 const unsigned y0,
 const unsigned y1)
{
  cuD
    y0xD = +0.0,
    y1xD = +0.0;
  cuJ
    y0xJ = +0.0,
    y1xJ = +0.0;

  const unsigned
    x32 = x + 32u;

  for (unsigned i = x; i < _nRow; i += 32u) {
    F64(AD, x, y0) = A0D[i];
    F64(AJ, x, y0) = A0J[i];
    F64(AD, x, y1) = A1D[i];
    F64(AJ, x, y1) = A1J[i];

    i += 32u;

    F64(AD, x32, y0) = A0D[i];
    F64(AJ, x32, y0) = A0J[i];
    F64(AD, x32, y1) = A1D[i];
    F64(AJ, x32, y1) = A1J[i];
    __syncthreads();

    #pragma unroll
    for (unsigned j = 0u; j < 64u; ++j) {
      // x_64 = (x + j) % 64u
      const unsigned x_64 = (x + j) & 0x3Fu;
      const cuD _x_hD =  F64(AD, x_64, x);
      const cuJ _x_hJ = -F64(AJ, x_64, x);
      const cuD _y0_D =  F64(AD, x_64, y0);
      const cuJ _y0_J =  F64(AJ, x_64, y0);
      const cuD _y1_D =  F64(AD, x_64, y1);
      const cuJ _y1_J =  F64(AJ, x_64, y1);
      Zfma(y0xD, y0xJ, _x_hD, _x_hJ, _y0_D, _y0_J, y0xD, y0xJ);
      Zfma(y1xD, y1xJ, _x_hD, _x_hJ, _y1_D, _y1_J, y1xD, y1xJ);
    }
    __syncthreads();
  }

  if (x == y0) {
    assert(y0xD > 0.0);
    assert(y0xD < INFTY);
  }
  if (x == y1) {
    assert(y1xD > 0.0);
    assert(y1xD < INFTY);
  }

  // first 32 columns set (A^H A unsymmetrized)
  F32(AD, x, y0) = y0xD;
  F32(AJ, x, y0) = y0xJ;
  F32(AD, x, y1) = y1xD;
  F32(AJ, x, y1) = y1xJ;
  __syncthreads();
}

MYDEVFN void zCholesky32
(volatile cuD *const AD, volatile cuJ *const AJ,
 const unsigned x,
 const unsigned y0,
 const unsigned y1)
{
  //      [ L ? ? ]
  // A -> [ L L ? ]
  //      [ L L L ]

  #pragma unroll
  for (unsigned k = 0u; k < 16u; ++k) {
    // cdiv(k)
    const cuD Akk = (((y0 == k) && (x >= k)) ? F32(AD, k, k) : +0.0);
    __syncthreads();
    if ((y0 == k) && (x >= k)) {
      assert(Akk > 0.0);
      assert(Akk < INFTY);
      if (x > k) {
        const double d = my_drsqrt_rn(Akk);
        F32(AD, x, k) *= d;
        F32(AJ, x, k) *= d;
      }
      else {
        F32(AD, x, k) = __dsqrt_rn(Akk);
        F32(AJ, x, k) = -0.0;
      }
    }
    __syncthreads();

    unsigned j = (k + 1u) + y0;

    // cmod(j,k)
    if (x >= j) {
      cuD AijD = F32(AD, x, j);
      cuJ AijJ = F32(AJ, x, j);
      const cuD _AikD = -F32(AD, x, k);
      const cuJ _AikJ = -F32(AJ, x, k);
      const cuD Ajk_D =  F32(AD, j, k);
      const cuJ Ajk_J = -F32(AJ, j, k);
      Zfma(AijD, AijJ, _AikD, _AikJ, Ajk_D, Ajk_J, AijD, AijJ);
      F32(AD, x, j) = AijD;
      F32(AJ, x, j) = AijJ;
    }
    __syncthreads();

    j += 16u;

    // cmod(j+16,k)
    if (x >= j) {
      cuD AijD = F32(AD, x, j);
      cuJ AijJ = F32(AJ, x, j);
      const cuD _AikD = -F32(AD, x, k);
      const cuJ _AikJ = -F32(AJ, x, k);
      const cuD Ajk_D =  F32(AD, j, k);
      const cuJ Ajk_J = -F32(AJ, j, k);
      Zfma(AijD, AijJ, _AikD, _AikJ, Ajk_D, Ajk_J, AijD, AijJ);
      F32(AD, x, j) = AijD;
      F32(AJ, x, j) = AijJ;
    }
    __syncthreads();
  }

  #pragma unroll
  for (unsigned k = 16u; k < 32u; ++k) {
    // cdiv(k)
    const cuD Akk = (((y1 == k) && (x >= k)) ? F32(AD, k, k) : +0.0);
    __syncthreads();
    if ((y1 == k) && (x >= k)) {
      assert(Akk > 0.0);
      assert(Akk < INFTY);
      if (x > k) {
        const double d = my_drsqrt_rn(Akk);
        F32(AD, x, k) *= d;
        F32(AJ, x, k) *= d;
      }
      else {
        F32(AD, x, k) = __dsqrt_rn(Akk);
        F32(AJ, x, k) = -0.0;
      }
    }
    __syncthreads();

    const unsigned j = (k + 1u) + y0;

    // cmod(j,k)
    if (x >= j) {
      cuD AijD = F32(AD, x, j);
      cuJ AijJ = F32(AJ, x, j);
      const cuD _AikD = -F32(AD, x, k);
      const cuJ _AikJ = -F32(AJ, x, k);
      const cuD Ajk_D =  F32(AD, j, k);
      const cuJ Ajk_J = -F32(AJ, j, k);
      Zfma(AijD, AijJ, _AikD, _AikJ, Ajk_D, Ajk_J, AijD, AijJ);
      F32(AD, x, j) = AijD;
      F32(AJ, x, j) = AijJ;
    }
    __syncthreads();
  }

  //      [ U U U ]
  // A -> [ 0 U U ]
  //      [ 0 0 U ]

  cuD Axy0D, Axy1D;
  cuJ Axy0J, Axy1J;

  if (x >= y0) {
    Axy0D =  F32(AD, x, y0);
    Axy0J = -F32(AJ, x, y0);
  }
  else {
    Axy0D = +0.0;
    Axy0J = +0.0;
  }

  if (x >= y1) {
    Axy1D =  F32(AD, x, y1);
    Axy1J = -F32(AJ, x, y1);
  }
  else {
    Axy1D = +0.0;
    Axy1J = +0.0;
  }

  __syncthreads();

  F32(AD, y0, x) = Axy0D;
  F32(AJ, y0, x) = Axy0J;
  F32(AD, y1, x) = Axy1D;
  F32(AJ, y1, x) = Axy1J;

  __syncthreads();
}

MYDEVFN void zFactorize
(const cuD *const F0D, const cuJ *const F0J,
 const cuD *const F1D, const cuJ *const F1J,
 const cuD *const G0D, const cuJ *const G0J,
 const cuD *const G1D, const cuJ *const G1J,
 volatile cuD *const AD, volatile cuJ *const AJ,
 volatile cuD *const BD, volatile cuJ *const BJ,
 const unsigned x,
 const unsigned y0,
 const unsigned y1)
{
  zAhA(F0D, F0J, F1D, F1J, AD, AJ, x, y0, y1);
  zAhA(G0D, G0J, G1D, G1J, BD, BJ, x, y0, y1);
  zCholesky32(AD, AJ, x, y0, y1);
  zCholesky32(BD, BJ, x, y0, y1);
}

#endif // !DEVICE_CODE_COMMON_CHOLESKY_HPP
