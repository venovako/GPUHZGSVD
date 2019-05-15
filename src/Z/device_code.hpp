#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP

#include "cuda_helper.hpp"

extern void HZ_L1_sv(const unsigned step) throw();
extern void initS(const int full, const unsigned nRank) throw();
extern void initV(const int sclV, const unsigned nRank) throw();
extern void initSymbols(cuD *const FD, cuJ *const FJ, cuD *const GD, cuJ *const GJ, cuD *const VD, cuJ *const VJ, double *const S, double *const H, double *const K, const unsigned nRowF, const unsigned nRowG, const unsigned nRank, const unsigned ldF, const unsigned ldG, const unsigned ldV, const unsigned nSwp) throw();

#endif // !DEVICE_CODE_HPP
