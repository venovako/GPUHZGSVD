#ifndef DEVICE_CODE_COMMON_DEFS_HPP
#define DEVICE_CODE_COMMON_DEFS_HPP

#ifndef MYKERN
#define MYKERN __global__ void
#else /* MYKERN */
#error MYKERN not definable externally
#endif /* ?MYKERN */

#ifndef MYDEVFN
#ifdef NDEBUG
#define MYDEVFN __device__ __forceinline__
#else /* DEBUG */
#define MYDEVFN __device__
#endif /* ?NDEBUG */
#else /* MYDEVFN */
#error MYDEVFN not definable externally
#endif /* ?MYDEVFN */

#ifndef HZ_L1_MAX_THREADS_PER_BLOCK
#define HZ_L1_MAX_THREADS_PER_BLOCK 512
#else /* HZ_L1_MAX_THREADS_PER_BLOCK */
#error HZ_L1_MAX_THREADS_PER_BLOCK not definable externally
#endif /* ?HZ_L1_MAX_THREADS_PER_BLOCK */

#ifndef HZ_L1_THREADS_PER_BLOCK_X
#define HZ_L1_THREADS_PER_BLOCK_X 32u
#else /* HZ_L1_THREADS_PER_BLOCK_X */
#error HZ_L1_THREADS_PER_BLOCK_X not definable externally
#endif /* ?HZ_L1_THREADS_PER_BLOCK_X */

#ifndef HZ_L1_THREADS_PER_BLOCK_Y
#define HZ_L1_THREADS_PER_BLOCK_Y 16u
#else /* HZ_L1_THREADS_PER_BLOCK_Y */
#error HZ_L1_THREADS_PER_BLOCK_Y not definable externally
#endif /* ?HZ_L1_THREADS_PER_BLOCK_Y */

#ifndef HZ_L1_MIN_BLOCKS_PER_SM
#define HZ_L1_MIN_BLOCKS_PER_SM 1
#else /* HZ_L1_MIN_BLOCKS_PER_SM */
#error HZ_L1_MIN_BLOCKS_PER_SM not definable externally
#endif /* ?HZ_L1_MIN_BLOCKS_PER_SM */

#if (WARP_SZ != 32u)
#error WARP_SZ not 32
#endif /* ?WARP_SZ */

#ifndef WARP_SZ_LGi
#define WARP_SZ_LGi 5
#else /* WARP_SZ_LGi */
#error WARP_SZ_LGi not definable externally
#endif /* ?WARP_SZ_LGi */

#ifndef WARP_SZ_LG
#define WARP_SZ_LG 5u
#else /* WARP_SZ_LG */
#error WARP_SZ_LG not definable externally
#endif /* ?WARP_SZ_LG */

#ifndef WARP_SZ_SUB1
#define WARP_SZ_SUB1 31u
#else /* WARP_SZ_SUB1 */
#error WARP_SZ_SUB1 not definable externally
#endif /* ?WARP_SZ_SUB1 */

#ifndef INFTY
#define INFTY CUDART_INF
#endif /* !INFTY */

#ifndef F32
#define F32(A, r, c) ((A)[(c) * 32u + (r)])
#else /* F32 */
#error F32 not definable externally
#endif /* ?F32 */

#ifndef F64
#define F64(A, r, c) ((A)[(c) * 64u + (r)])
#else /* F64 */
#error F64 not definable externally
#endif /* ?F64 */

#ifndef _shfl_xor
#define _shfl_xor(x,y) __shfl_xor_sync(~0u, (x), (y))
#else /* _shfl_xor */
#error _shfl_xor already defined
#endif /* !_shfl_xor */

#ifndef _shfl
#define _shfl(x,y) __shfl_sync(~0u, (x), (y))
#else /* _shfl */
#error _shfl already defined
#endif /* !_shfl */

// Thanks to Norbert Juffa of NVIDIA.
MYDEVFN double
nj_drsqrt_rn(double a)
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
    // Round as shown in Peter Markstein, "IA-64 and Elementary Functions"
    y = __fma_rn(__fma_rn(0.375, e, 0.5), __dmul_rn(e, y), y);
    d = d >> 1;
    a = __hiloint2double(__double2hiint(y) + d, __double2loint(y));
  } else if (a == 0.0) {
    a = __hiloint2double((ihi & 0x80000000) | 0x7ff00000, 0x00000000);
  } else if (a < 0.0) {
    a = __hiloint2double(0xfff80000, 0x00000000);
  } else if (isinf(a)) {
    a = __hiloint2double(ihi & 0x80000000, 0x00000000);
  } else if (isnan(a)) {
    a = __dadd_rn(a, a);
  } else {
    a = __dmul_rn(a, __hiloint2double(0x7fd00000, 0));
    y = rsqrt(a);
    h = __dmul_rn(y, y);
    l = __fma_rn(y, y, -h);
    e = __fma_rn(l, -a, __fma_rn(h, -a, 1.0));
    // Round as shown in Peter Markstein, "IA-64 and Elementary Functions"
    y = __fma_rn(__fma_rn(0.375, e, 0.5), __dmul_rn(e, y), y);
    a = __hiloint2double(__double2hiint(y) + 0x1ff00000, __double2loint(y));
  }
  return a;
}

// sum x
// Kepler warp shuffle
MYDEVFN double
dSum32(const double x)
{
  int lo_my, hi_my, lo_his, hi_his;
  double x_my = x, x_his;

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 16);
  hi_his = _shfl_xor(hi_my, 16);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = _dadd_rn(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 8);
  hi_his = _shfl_xor(hi_my, 8);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = _dadd_rn(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 4);
  hi_his = _shfl_xor(hi_my, 4);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = _dadd_rn(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 2);
  hi_his = _shfl_xor(hi_my, 2);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = _dadd_rn(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 1);
  hi_his = _shfl_xor(hi_my, 1);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = _dadd_rn(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl(lo_my, 0);
  hi_his = _shfl(hi_my, 0);
  x_his = __hiloint2double(hi_his, lo_his);

  return x_his;
}

// max|x|
// Kepler warp shuffle
MYDEVFN double
dMax32(const double x)
{
  int lo_my, hi_my, lo_his, hi_his;
  double x_my = fabs(x), x_his;

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 16);
  hi_his = _shfl_xor(hi_my, 16);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = fmax(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 8);
  hi_his = _shfl_xor(hi_my, 8);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = fmax(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 4);
  hi_his = _shfl_xor(hi_my, 4);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = fmax(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 2);
  hi_his = _shfl_xor(hi_my, 2);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = fmax(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 1);
  hi_his = _shfl_xor(hi_my, 1);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = fmax(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl(lo_my, 0);
  hi_his = _shfl(hi_my, 0);
  x_his = __hiloint2double(hi_his, lo_his);

  return x_his;
}

// min|x|, x =/= 0
// Kepler warp shuffle
MYDEVFN double
dMin32(const double x)
{
  int lo_my, hi_my, lo_his, hi_his;
  double x_my = ((x == 0.0) ? DBL_MAX : fabs(x)), x_his;

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 16);
  hi_his = _shfl_xor(hi_my, 16);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = fmin(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 8);
  hi_his = _shfl_xor(hi_my, 8);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = fmin(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 4);
  hi_his = _shfl_xor(hi_my, 4);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = fmin(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 2);
  hi_his = _shfl_xor(hi_my, 2);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = fmin(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 1);
  hi_his = _shfl_xor(hi_my, 1);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my = fmin(x_my, x_his);

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl(lo_my, 0);
  hi_his = _shfl(hi_my, 0);
  x_his = __hiloint2double(hi_his, lo_his);

  return x_his;
}

#endif /* !DEVICE_CODE_COMMON_DEFS_HPP */
