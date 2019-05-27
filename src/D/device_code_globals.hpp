#ifndef DEVICE_CODE_GLOBALS_HPP
#define DEVICE_CODE_GLOBALS_HPP

__constant__
double *_F, *_G, *_V;

__constant__
double *_S, *_H, *_K;

__constant__
unsigned _nRowF, _nRowG, _nRowV, _nRank, _ldF, _ldG, _ldV, _nSwp;

__constant__
unsigned _STRAT0_STEPS, _STRAT0_PAIRS;

STRAT0_STORAGE
unsigned STRAT0_DTYPE _strat0[STRAT0_MAX_STEPS][STRAT0_MAX_PAIRS][2u];

STRAT1_STORAGE
unsigned STRAT1_DTYPE _strat1[STRAT1_MAX_STEPS][STRAT1_MAX_PAIRS][2u];

#endif // !DEVICE_CODE_GLOBALS_HPP
