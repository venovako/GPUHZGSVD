#ifndef DEVICE_CODE_ACCUMV_HPP
#define DEVICE_CODE_ACCUMV_HPP

MYDEVFN void zMultV
(cuD *const __restrict__ F0D, cuJ *const __restrict__ F0J,
 cuD *const __restrict__ F1D, cuJ *const __restrict__ F1J,
 cuD *const __restrict__ G0D, cuJ *const __restrict__ G0J,
 cuD *const __restrict__ G1D, cuJ *const __restrict__ G1J,
 cuD *const __restrict__ V0D, cuJ *const __restrict__ V0J,
 cuD *const __restrict__ V1D, cuJ *const __restrict__ V1J,
 volatile cuD *const __restrict__ AD, volatile cuJ *const __restrict__ AJ,
 volatile cuD *const __restrict__ BD, volatile cuJ *const __restrict__ BJ,
 volatile const cuD *const __restrict__ CD, volatile const cuJ *const __restrict__ CJ,
 const unsigned x,
 const unsigned y0,
 const unsigned y1)
{
  zMultAV(F0D, F0J, F1D, F1J, AD, AJ, CD, CJ, x, y0, y1, _nRowF);
  zMultAV(G0D, G0J, G1D, G1J, BD, BJ, CD, CJ, x, y0, y1, _nRowG);
  zMultAV(V0D, V0J, V1D, V1J, AD, AJ, CD, CJ, x, y0, y1, _nRowV);
}

#endif /* !DEVICE_CODE_ACCUMV_HPP */
