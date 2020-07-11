#ifndef DEVICE_CODE_COMMON_CHOLESKY_HPP
#define DEVICE_CODE_COMMON_CHOLESKY_HPP

MYDEVFN void dCholesky32
(volatile double *const __restrict__ A,
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
    const double Akk = (((y0 == k) && (x >= k)) ? F32(A, k, k) : 0.0);
    __syncthreads();
    if ((y0 == k) && (x >= k))
      F32(A, x, k) = ((x > k) ? _dmul_rn(F32(A, x, k), _drsqrt_rn(Akk)) : _dsqrt_rn(Akk));
    __syncthreads();

    unsigned j = (k + 1u) + y0;

    // cmod(j,k)
    if (x >= j) {
      const double Aij = F32(A, x, j);
      const double _Aik = -F32(A, x, k);
      const double Ajk = F32(A, j, k);
      F32(A, x, j) = _fma_rn(_Aik, Ajk, Aij);
    }
    __syncthreads();

    j += 16u;

    // cmod(j+16,k)
    if (x >= j) {
      const double Aij = F32(A, x, j);
      const double _Aik = -F32(A, x, k);
      const double Ajk = F32(A, j, k);
      F32(A, x, j) = _fma_rn(_Aik, Ajk, Aij);
    }
    __syncthreads();
  }

  #pragma unroll
  for (unsigned k = 16u; k < 32u; ++k) {
    // cdiv(k)
    const double Akk = (((y1 == k) && (x >= k)) ? F32(A, k, k) : 0.0);
    __syncthreads();
    if ((y1 == k) && (x >= k))
      F32(A, x, k) = ((x > k) ? _dmul_rn(F32(A, x, k), _drsqrt_rn(Akk)) : _dsqrt_rn(Akk));
    __syncthreads();

    const unsigned j = (k + 1u) + y0;

    // cmod(j,k)
    if (x >= j) {
      const double Aij = F32(A, x, j);
      const double _Aik = -F32(A, x, k);
      const double Ajk = F32(A, j, k);
      F32(A, x, j) = _fma_rn(_Aik, Ajk, Aij);
    }
    __syncthreads();
  }

  //      [ U U U ]
  // A -> [ 0 U U ]
  //      [ 0 0 U ]
  
  double
    Axy0 = +0.0,
    Axy1 = +0.0;

  if (x >= y0)
    Axy0 = F32(A, x, y0);
  if (x >= y1)
    Axy1 = F32(A, x, y1);

  __syncthreads();

  F32(A, y0, x) = Axy0;
  F32(A, y1, x) = Axy1;

  __syncthreads();
}

#endif /* !DEVICE_CODE_COMMON_CHOLESKY_HPP */
