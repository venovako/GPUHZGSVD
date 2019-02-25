#ifndef DEVICE_CODE_COMMON_KEPLER_HPP
#define DEVICE_CODE_COMMON_KEPLER_HPP

#ifndef _shfl_xor
#if __CUDACC_VER_MAJOR__ >= 9
#define _shfl_xor(x,y) __shfl_xor_sync(~0u, (x), (y))
#else // __CUDACC_VER_MAJOR__ < 9
#define _shfl_xor(x,y) __shfl_xor((x), (y))
#endif // ?__CUDACC_VER_MAJOR__
#else // _shfl_xor
#error _shfl_xor already defined
#endif // !_shfl_xor

#ifndef _shfl
#if __CUDACC_VER_MAJOR__ >= 9
#define _shfl(x,y) __shfl_sync(~0u, (x), (y))
#else // __CUDACC_VER_MAJOR__ < 9
#define _shfl(x,y) __shfl((x), (y))
#endif // ?__CUDACC_VER_MAJOR__
#else // _shfl
#error _shfl already defined
#endif // !_shfl

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
  x_my += x_his;

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 8);
  hi_his = _shfl_xor(hi_my, 8);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my += x_his;

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 4);
  hi_his = _shfl_xor(hi_my, 4);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my += x_his;

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 2);
  hi_his = _shfl_xor(hi_my, 2);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my += x_his;

  lo_my = __double2loint(x_my);
  hi_my = __double2hiint(x_my);
  lo_his = _shfl_xor(lo_my, 1);
  hi_his = _shfl_xor(hi_my, 1);
  x_his = __hiloint2double(hi_his, lo_his);
  x_my += x_his;

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

#if ((CVG == 0) || (CVG == 2) || (CVG == 4) || (CVG == 6))
MYDEVFN double zSsq32(const cuD D, const cuJ J)
{
  const double z = __fma_rn(D, D, J * J);
  return dSum32(z);
}
#else // ((CVG == 1) || (CVG == 3) || (CVG == 5) || (CVG == 7))
MYDEVFN double zSsq32(const cuD D, const cuJ J)
{
  const cuD vD = __dmul_rd(D, D);
  const cuD eD = __fma_rn(D, D, -vD);
  const cuJ vJ = __dmul_rd(J, J);
  const cuJ eJ = __fma_rn(J, J, -vJ);
  double s = eD + eJ;
  s = ((vD <= vJ) ? ((s + vD) + vJ) : ((s + vJ) + vD));
  return dSum32(s);
}
#endif // ?CVG

MYDEVFN void zDot32(cuD &cD, cuJ &cJ, const cuD aD, const cuJ aJ, const cuD bD, const cuJ bJ)
{
  cuD x = +0.0;
  cuJ y = +0.0;
  Zfma(x, y, aD, -aJ, bD, bJ, x, y);
  cD = dSum32(x);
  cJ = dSum32(y);
}

#endif // !DEVICE_CODE_COMMON_KEPLER_HPP
