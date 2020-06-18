#ifndef DEVICE_CODE_COMMON_KEPLER_HPP
#define DEVICE_CODE_COMMON_KEPLER_HPP

MYDEVFN double
dSsq32(const double x)
{
  return dSum32(_dmul_rn(x, x));
}

#if ((CVG == 1) || (CVG == 3) || (CVG == 5) || (CVG == 7))
MYDEVFN double
dSUM_PROD_32(const double x, const double y, double &s0, double &s1)
{
  const double xy = _dmul_rd(x, y); // (x *_RD y)
  const double rp = _fma_rn(x, y, -xy); // rounded-off part (always >= 0)
  s0 = dSum32(xy);
  s1 = dSum32(rp);
  return _dadd_rn(s0, s1);
}

MYDEVFN double
dSSQ32(const double x, double &s0, double &s1)
{
  return dSUM_PROD_32(x, x, s0, s1);
}
#endif /* ?CVG */

#endif /* !DEVICE_CODE_COMMON_KEPLER_HPP */
