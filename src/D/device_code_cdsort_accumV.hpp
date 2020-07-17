#ifndef DEVICE_CODE_CDSORT_ACCUMV_HPP
#define DEVICE_CODE_CDSORT_ACCUMV_HPP

MYKERN __launch_bounds__(HZ_L1_MAX_THREADS_PER_BLOCK, HZ_L1_MIN_BLOCKS_PER_SM)
  dHZ_L1_sv(const unsigned step)
{
  __shared__ double shMem[3u * 1024u];

  const unsigned
    x = threadIdx.x,
    y0 = threadIdx.y,
    iBlk = _strat1[step][blockIdx.x][0u],
    jBlk = _strat1[step][blockIdx.x][1u],
    y1 = y0 + 16u;

  const unsigned
    ix = iBlk * 16u + y0,
    jx = jBlk * 16u + y0;

  double
    *const F0 = _F + ix * _ldF,
    *const F1 = _F + jx * _ldF,
    *const G0 = _G + ix * _ldG,
    *const G1 = _G + jx * _ldG,
    *const V0 = _V + ix * _ldV,
    *const V1 = _V + jx * _ldV;
  volatile double
    *const F = shMem;
  volatile double
    *const G = shMem + 1024u;
  volatile double
    *const V = shMem + 2048u;

#if (defined(PROFILE) && (PROFILE == 0))
  const unsigned bix2 = (unsigned)(blockIdx.x) << C_SHIFTR;
  __syncthreads();
  unsigned long long t = static_cast<unsigned long long>(clock64());
#endif /* ?PROFILE */

#ifdef USE_QR
  dFactorize(F0, F1, F, G, _nRowF, x, y0, y1);
#if (defined(PROFILE) && (PROFILE == 0))
  t = static_cast<unsigned long long>(clock64()) - t;
  (void)atomicMax((_C + bix2) + C_SUBPHASE_1, t);
  __syncthreads();
  t = static_cast<unsigned long long>(clock64());
#endif /* ?PROFILE */
  dFactorize(G0, G1, G, V, _nRowG, x, y0, y1);
#else /* !USE_QR */
  dAtA(F0, F1, F, _nRowF, x, y0, y1);
  dAtA(G0, G1, G, _nRowG, x, y0, y1);
#if (defined(PROFILE) && (PROFILE == 0))
  t = static_cast<unsigned long long>(clock64()) - t;
  (void)atomicMax((_C + bix2) + C_SUBPHASE_1, t);
  __syncthreads();
  t = static_cast<unsigned long long>(clock64());
#endif /* ?PROFILE */
  dCholesky32(F, x, y0, y1);
  dCholesky32(G, x, y0, y1);
#endif /* ?USE_QR */

#if (defined(PROFILE) && (PROFILE == 0))
  t = static_cast<unsigned long long>(clock64()) - t;
  (void)atomicMax((_C + bix2) + C_SUBPHASE_2, t);
  __syncthreads();
  t = static_cast<unsigned long long>(clock64());
#endif /* ?PROFILE */
  (void)dHZ_L0_sv(F, G, V, x, y0);
#if (defined(PROFILE) && (PROFILE == 0))
  t = static_cast<unsigned long long>(clock64()) - t;
  (void)atomicMax((_C + bix2) + C_SUBPHASE_3, t);
  __syncthreads();
  t = static_cast<unsigned long long>(clock64());
#endif /* ?PROFILE */
  dMultV(F0, F1, G0, G1, V0, V1, F, G, V, x, y0, y1);
#if (defined(PROFILE) && (PROFILE == 0))
  t = static_cast<unsigned long long>(clock64()) - t;
  (void)atomicMax((_C + bix2) + C_SUBPHASE_4, t);
#endif /* ?PROFILE */
}

#endif /* !DEVICE_CODE_CDSORT_ACCUMV_HPP */
