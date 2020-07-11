#ifndef DEVICE_CODE_CDSORT_ACCUMV_HPP
#define DEVICE_CODE_CDSORT_ACCUMV_HPP

MYKERN __launch_bounds__(HZ_L1_MAX_THREADS_PER_BLOCK, HZ_L1_MIN_BLOCKS_PER_SM)
  dHZ_L1_sv
 (double *const __restrict__ _F,
  double *const __restrict__ _G)
{
  __shared__ double shMem[3u * 1024u];

  const unsigned
    ix = (blockIdx.x << 5),
    x  = threadIdx.x,
    y0 = threadIdx.y,
    y1 = (y0 + 16u);

  double
    *const dF = _F + ix * _ldF,
    *const dG = _G + ix * _ldG,
    *const dW = _W + ix * _ldW;
  volatile double
    *const F = shMem;
  volatile double
    *const G = shMem + 1024u;
  volatile double
    *const V = shMem + 2048u;

  F32(F, x, y0) = dF[y0 * _ldF + x];
  F32(F, x, y1) = dF[y1 * _ldF + x];
  F32(G, x, y0) = dG[y0 * _ldG + x];
  F32(G, x, y1) = dG[y1 * _ldG + x];
  __syncthreads();

  dCholesky32(F, x, y0, y1);
  dCholesky32(G, x, y0, y1);
  dHZ_L0_sv(F, G, V, x, y0);

  dW[y0 * _ldW + x] = F32(V, x, y0);
  dW[y1 * _ldW + x] = F32(V, x, y1);
  __syncthreads();
}

#endif /* !DEVICE_CODE_CDSORT_ACCUMV_HPP */
