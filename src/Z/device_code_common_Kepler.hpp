#ifndef DEVICE_CODE_COMMON_KEPLER_HPP
#define DEVICE_CODE_COMMON_KEPLER_HPP

#if ((CVG == 0) || (CVG == 2) || (CVG == 4) || (CVG == 6))
MYDEVFN double zSsq32(const cuD D, const cuJ J)
{
  const double z = _fma_rn(D, D, _dmul_rn(J, J));
  return dSum32(z);
}
#else /* ((CVG == 1) || (CVG == 3) || (CVG == 5) || (CVG == 7)) */
MYDEVFN double zSsq32(const cuD D, const cuJ J)
{
  const cuD vD = _dmul_rd(D, D);
  const cuD eD = _fma_rn(D, D, -vD);
  const cuJ vJ = _dmul_rd(J, J);
  const cuJ eJ = _fma_rn(J, J, -vJ);
  double s = _dadd_rn(eD, eJ);
  s = ((vD <= vJ) ? _dadd_rn(_dadd_rn(s, vD), vJ) : _dadd_rn(_dadd_rn(s, vJ), vD));
  return dSum32(s);
}
#endif /* ?CVG */

MYDEVFN void zDot32(cuD &cD, cuJ &cJ, const cuD aD, const cuJ aJ, const cuD bD, const cuJ bJ)
{
  cuD x;
  cuJ y;
  Zmul(x, y, aD, -aJ, bD, bJ);
  cD = dSum32(x);
  cJ = dSum32(y);
}

#endif /* !DEVICE_CODE_COMMON_KEPLER_HPP */
