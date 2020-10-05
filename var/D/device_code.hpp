#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP

#include "defines.hpp"

extern void HZ_L1_sv(double *const F, double *const G, const cudaStream_t s = 0) throw();
extern void initS(double *const F, double *const G, double *const V, double *const S, double *const H, double *const K, const unsigned nRank, const cudaStream_t s = 0) throw();
extern void initS(double *const F, double *const G, double *const V, const unsigned nRank, const cudaStream_t s = 0) throw();
extern void initV(double *const F, double *const G, double *const V, const unsigned nRank, const cudaStream_t s = 0) throw();
extern void initSymbols(double *const W, unsigned long long *const C, const unsigned nRowF, const unsigned nRowG, const unsigned nRowV, const unsigned nRowW, const unsigned ldF, const unsigned ldG, const unsigned ldV, const unsigned ldW, const unsigned nRank, const unsigned nSwp, const cudaStream_t s = 0) throw();

#endif /* !DEVICE_CODE_HPP */
