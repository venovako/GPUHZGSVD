#ifndef DEVICE_CODE_COMMON_KEPLER_HPP
#define DEVICE_CODE_COMMON_KEPLER_HPP

#ifndef _shfl_xor
#define _shfl_xor(x,y) __shfl_xor_sync(~0u, (x), (y))
#else // _shfl_xor
#error _shfl_xor already defined
#endif // !_shfl_xor

#ifndef _shfl
#define _shfl(x,y) __shfl_sync(~0u, (x), (y))
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

MYDEVFN double
dSsq32(const double x)
{
  return dSum32(x * x);
}

#if ((CVG == 1) || (CVG == 3) || (CVG == 5) || (CVG == 7))
MYDEVFN double
dSUM_PROD_32(const double x, const double y, double &s0, double &s1)
{
  const double xy = __dmul_rd(x, y); // (x *_RD y)
  const double rp = __fma_rn(x, y, -xy); // rounded-off part (always >= 0)
  s0 = dSum32(xy);
  s1 = dSum32(rp);
  return (s0 + s1);
}

MYDEVFN double
dSSQ32(const double x, double &s0, double &s1)
{
  return dSUM_PROD_32(x, x, s0, s1);
}
#endif // ?CVG

#endif // !DEVICE_CODE_COMMON_KEPLER_HPP
