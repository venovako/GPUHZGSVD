#ifndef DEVICE_CODE_ACCUMV_HPP
#define DEVICE_CODE_ACCUMV_HPP

MYDEVFN void dMultV
(double *const __restrict__ F0,
 double *const __restrict__ F1,
 double *const __restrict__ G0,
 double *const __restrict__ G1,
 double *const __restrict__ V0,
 double *const __restrict__ V1,
 volatile double *const __restrict__ A,
 volatile double *const __restrict__ B,
 volatile const double *const __restrict__ C,
 const unsigned x,
 const unsigned y0,
 const unsigned y1)
{
  dMultAV(F0, F1, A, C, x, y0, y1, _nRowF);
  dMultAV(G0, G1, B, C, x, y0, y1, _nRowG);
  dMultAV(V0, V1, A, C, x, y0, y1, _nRowV);
}

#endif /* !DEVICE_CODE_ACCUMV_HPP */
