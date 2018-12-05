#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP

#include "cuda_helper.hpp"

extern void HZ_L1_sv(const unsigned step) throw();
extern void initS(const int full, const unsigned nRank, const cudaStream_t s) throw();
extern void initV(const int sclV, const unsigned nRank, const cudaStream_t s) throw();
extern void initSymbols(double *const F, double *const G, double *const V, double *const S, double *const H, double *const K, volatile unsigned Long *const cvg, const unsigned nRow, const unsigned nRank, const unsigned ldF, const unsigned ldG, const unsigned ldV, const unsigned nSwp) throw();

#endif // !DEVICE_CODE_HPP
