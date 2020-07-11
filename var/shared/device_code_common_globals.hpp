#ifndef DEVICE_CODE_COMMON_GLOBALS_HPP
#define DEVICE_CODE_COMMON_GLOBALS_HPP

__constant__ double *_W;
__constant__ unsigned long long *_C;

__constant__ unsigned _nRowF;
__constant__ unsigned _nRowG;
__constant__ unsigned _nRowV;
__constant__ unsigned _nRowW;

__constant__ unsigned _ldF;
__constant__ unsigned _ldG;
__constant__ unsigned _ldV;
__constant__ unsigned _ldW;

__constant__ unsigned _nRank;
__constant__ unsigned _nSwp;

__constant__ unsigned _STRAT0_STEPS;
__constant__ unsigned _STRAT0_PAIRS;

STRAT0_STORAGE unsigned STRAT0_DTYPE _strat0[STRAT0_MAX_STEPS][STRAT0_MAX_PAIRS][2u];

#endif /* !DEVICE_CODE_COMMON_GLOBALS_HPP */
