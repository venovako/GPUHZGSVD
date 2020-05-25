#ifndef DEVICE_CODE_ACCUMV_HPP
#define DEVICE_CODE_ACCUMV_HPP

MYDEVFN void zMultV
(cuD *const F0D, cuJ *const F0J,
 cuD *const F1D, cuJ *const F1J,
 cuD *const G0D, cuJ *const G0J,
 cuD *const G1D, cuJ *const G1J,
 cuD *const V0D, cuJ *const V0J,
 cuD *const V1D, cuJ *const V1J,
 volatile cuD *const AD, volatile cuJ *const AJ,
 volatile cuD *const BD, volatile cuJ *const BJ,
 volatile const cuD *const CD, volatile const cuJ *const CJ,
 const unsigned x,
 const unsigned y0,
 const unsigned y1)
{
  zMultAV(F0D, F0J, F1D, F1J, AD, AJ, CD, CJ, x, y0, y1, _nRowF);
  zMultAV(G0D, G0J, G1D, G1J, BD, BJ, CD, CJ, x, y0, y1, _nRowG);
  zMultAV(V0D, V0J, V1D, V1J, AD, AJ, CD, CJ, x, y0, y1, _nRowV);
}

#endif /* !DEVICE_CODE_ACCUMV_HPP */
