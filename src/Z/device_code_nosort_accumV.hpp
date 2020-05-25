#ifndef DEVICE_CODE_NOSORT_ACCUMV_HPP
#define DEVICE_CODE_NOSORT_ACCUMV_HPP

MYKERN __launch_bounds__(HZ_L1_MAX_THREADS_PER_BLOCK, HZ_L1_MIN_BLOCKS_PER_SM)
  zHZ_L1_v(const unsigned step)
{
  __shared__ double shMem[3u * 2u * 32u * 32u];

  const unsigned
    x = threadIdx.x,
    y0 = threadIdx.y,
    iBlk = _strat1[step][blockIdx.x][0u],
    jBlk = _strat1[step][blockIdx.x][1u],
    y1 = y0 + 16u;

  const unsigned
    ix = iBlk * 16u + y0,
    jx = jBlk * 16u + y0;

  cuD
    *const F0D = static_cast<cuD*>(_FD + ix * _ldF),
    *const F1D = static_cast<cuD*>(_FD + jx * _ldF),
    *const G0D = static_cast<cuD*>(_GD + ix * _ldG),
    *const G1D = static_cast<cuD*>(_GD + jx * _ldG),
    *const V0D = static_cast<cuD*>(_VD + ix * _ldV),
    *const V1D = static_cast<cuD*>(_VD + jx * _ldV);
  volatile cuD
    *const FD = static_cast<cuD*>(shMem + 0u * 32u * 32u);
  volatile cuD
    *const GD = static_cast<cuD*>(shMem + 1u * 32u * 32u);
  volatile cuD
    *const VD = static_cast<cuD*>(shMem + 2u * 32u * 32u);

  cuJ
    *const F0J = static_cast<cuJ*>(_FJ + ix * _ldF),
    *const F1J = static_cast<cuJ*>(_FJ + jx * _ldF),
    *const G0J = static_cast<cuJ*>(_GJ + ix * _ldG),
    *const G1J = static_cast<cuJ*>(_GJ + jx * _ldG),
    *const V0J = static_cast<cuJ*>(_VJ + ix * _ldV),
    *const V1J = static_cast<cuJ*>(_VJ + jx * _ldV);
  volatile cuJ
    *const FJ = static_cast<cuJ*>(shMem + 3u * 32u * 32u);
  volatile cuJ
    *const GJ = static_cast<cuJ*>(shMem + 4u * 32u * 32u);
  volatile cuJ
    *const VJ = static_cast<cuJ*>(shMem + 5u * 32u * 32u);

  zFactorize(F0D, F0J, F1D, F1J, G0D, G0J, G1D, G1J, FD, FJ, GD, GJ, x, y0, y1);
  (void)zHZ_L0_v(FD, FJ, GD, GJ, VD, VJ, x, y0);
  zMultV(F0D, F0J, F1D, F1J, G0D, G0J, G1D, G1J, V0D, V0J, V1D, V1J, FD, FJ, GD, GJ, VD, VJ, x, y0, y1);
}

#endif /* !DEVICE_CODE_NOSORT_ACCUMV_HPP */
