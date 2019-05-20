#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP

#include "cuda_helper.hpp"

extern void HZ_L1_sv(const unsigned step) throw();
extern void initS(const int full, const unsigned nRank) throw();
extern void initV(const int sclV, const unsigned nRank, const unsigned ifc0, const unsigned ifc1) throw();
extern void initSymbols(double *const F, double *const G, double *const V, double *const S, double *const H, double *const K, const unsigned nRowF, const unsigned nRowG, const unsigned nRank, const unsigned ldF, const unsigned ldG, const unsigned ldV, const unsigned nSwp) throw();

#endif // !DEVICE_CODE_HPP
