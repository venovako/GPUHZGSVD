#ifndef HZ_L_HPP
#define HZ_L_HPP

// maximal number of sweeps per Jacobi process
#ifndef HZ_NSWEEP
#define HZ_NSWEEP 99u
#endif // !HZ_NSWEEP

// HEPS = DBL_EPSILON / 2 = 2^(-53) (* Lapack/RN Epsilon *)

// sqrt(32) * HEPS
#ifndef HZ_MYTOL
#define HZ_MYTOL 6.28036983473510067E-16
#endif // !HZ_MYTOL

#ifndef HZ_BLK_ORI
#define HZ_BLK_ORI 8u
#else // HZ_BLK_ORI
#error HZ_BLK_ORI not definable externally
#endif // !HZ_BLK_ORI

#ifndef HZ_L1_NCOLB
#define HZ_L1_NCOLB 16u
#else // HZ_L1_NCOLB
#error HZ_L1_NCOLB not definable externally
#endif // !HZ_L1_NCOLB

#ifndef STRAT_MMSTEP
#define STRAT_MMSTEP 1u
#else // STRAT_MMSTEP
#error STRAT_MMSTEP not definable externally
#endif // !STRAT_MMSTEP

#ifndef STRAT_BRENTL
#define STRAT_BRENTL 2u
#else // STRAT_BRENTL
#error STRAT_BRENTL not definable externally
#endif // !STRAT_BRENTL

#ifndef STRAT_COLCYC
#define STRAT_COLCYC 3u
#else // STRAT_COLCYC
#error STRAT_COLCYC not definable externally
#endif // !STRAT_COLCYC

#ifndef STRAT_CYCLOC
#define STRAT_CYCLOC 4u
#else // STRAT_CYCLOC
#error STRAT_CYCLOC not definable externally
#endif // !STRAT_CYCLOC

#ifndef STRAT_ROWCYC
#define STRAT_ROWCYC 5u
#else // STRAT_ROWCYC
#error STRAT_ROWCYC not definable externally
#endif // !STRAT_ROWCYC

#ifndef STRAT_CYCWOR
#define STRAT_CYCWOR 6u
#else // STRAT_CYCWOR
#error STRAT_CYCWOR not definable externally
#endif // !STRAT_CYCWOR

#ifndef STRAT_BLKREC
#define STRAT_BLKREC 7u
#else // STRAT_BLKREC
#error STRAT_BLKREC not definable externally
#endif // !STRAT_BLKREC

// n-1 for cyclic, n for quasi-cyclic
#ifndef STRAT0_MAX_STEPS
#define STRAT0_MAX_STEPS 32u
#endif // !STRAT0_MAX_STEPS
#ifndef STRAT1_MAX_STEPS
#define STRAT1_MAX_STEPS 1024u
#endif // !STRAT1_MAX_STEPS

// n/2 for even n
#ifndef STRAT0_MAX_PAIRS
#define STRAT0_MAX_PAIRS 16u
#endif // !STRAT0_MAX_PAIRS
#ifndef STRAT1_MAX_PAIRS
#define STRAT1_MAX_PAIRS 512u
#endif // !STRAT1_MAX_PAIRS

#ifndef STRAT0_STORAGE
#define STRAT0_STORAGE __constant__
#endif // !STRAT0_STORAGE
#ifndef STRAT1_STORAGE
#define STRAT1_STORAGE __device__
#endif // !STRAT1_STORAGE

#ifndef STRAT0_DTYPE
#define STRAT0_DTYPE char
#endif // !STRAT0_DTYPE
#ifndef STRAT1_DTYPE
#define STRAT1_DTYPE short
#endif // !STRAT1_DTYPE

extern unsigned STRAT0, STRAT0_STEPS, STRAT0_PAIRS;
extern unsigned STRAT1, STRAT1_STEPS, STRAT1_PAIRS;

extern unsigned STRAT0_DTYPE strat0[STRAT0_MAX_STEPS][STRAT0_MAX_PAIRS][2u];
extern unsigned STRAT1_DTYPE strat1[STRAT1_MAX_STEPS][STRAT1_MAX_PAIRS][2u];

extern void init_strats(const char *const sdy, const char *const snp0, const unsigned n0, const char *const snp1, const unsigned n1) throw();

#endif // !HZ_L_HPP