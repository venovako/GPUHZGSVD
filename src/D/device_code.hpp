#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP

#include "defines.hpp"

EXTERN_C void HZ_L1_sv(const unsigned step) throw();
EXTERN_C void initS(const int full, const unsigned nRank) throw();
EXTERN_C void initV(const int sclV, const unsigned nRank
#ifdef USE_MPI
  , const unsigned ifc0, const unsigned ifc1
#endif // USE_MPI
) throw();
EXTERN_C void initSymbols(double *const F, double *const G, double *const V, double *const S, double *const H, double *const K, unsigned long long *const C, const unsigned nRowF, const unsigned nRowG, const unsigned nRowV, const unsigned nRank, const unsigned ldF, const unsigned ldG, const unsigned ldV, const unsigned nSwp) throw();

#endif // !DEVICE_CODE_HPP
