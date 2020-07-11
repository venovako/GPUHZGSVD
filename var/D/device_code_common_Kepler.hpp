#ifndef DEVICE_CODE_COMMON_KEPLER_HPP
#define DEVICE_CODE_COMMON_KEPLER_HPP

MYDEVFN double
dSsq32(const double x)
{
  return dSum32(_dmul_rn(x, x));
}

#endif /* !DEVICE_CODE_COMMON_KEPLER_HPP */
