#ifndef HZ_L_HPP
#define HZ_L_HPP

#include "defines.hpp"

// maximal number of sweeps per Jacobi process
#ifndef HZ_NSWEEP
#define HZ_NSWEEP 30u
#endif /* !HZ_NSWEEP */

// HEPS = DBL_EPSILON / 2 = 2^(-53) (* Lapack/RN Epsilon *)

// sqrt(HEPS)
#ifndef SQRT_HEPS
#define SQRT_HEPS 1.05367121277235087E-08
#else /* SQRT_HEPS */
#error SQRT_HEPS not definable externally
#endif /* ?SQRT_HEPS */

// sqrt(2 / HEPS)
#ifndef SQRT_2_HEPS
#define SQRT_2_HEPS 1.34217728000000000E+08
#else /* SQRT_2_HEPS */
#error SQRT_2_HEPS not definable externally
#endif /* ?SQRT_2_HEPS */

// sqrt(32) * HEPS
#ifndef HZ_MYTOL
#define HZ_MYTOL 6.28036983473510067E-16
#endif /* !HZ_MYTOL */

#ifndef HZ_BO_1
#define HZ_BO_1 8u
#else /* HZ_BO_1 */
#error HZ_BO_1 not definable externally
#endif /* ?HZ_BO_1 */

#ifndef HZ_BO_2
#define HZ_BO_2 4u
#else /* HZ_BO_2 */
#error HZ_BO_2 not definable externally
#endif /* ?HZ_BO_2 */

#ifndef HZ_L1_NCOLB
#define HZ_L1_NCOLB 16u
#else /* HZ_L1_NCOLB */
#error HZ_L1_NCOLB not definable externally
#endif /* ?HZ_L1_NCOLB */

#include "jstrat.h"

#ifndef STRAT_CYCWOR
#define STRAT_CYCWOR 2u
#else /* STRAT_CYCWOR */
#error STRAT_CYCWOR not definable externally
#endif /* ?STRAT_CYCWOR */

#ifndef STRAT_MMSTEP
#define STRAT_MMSTEP 4u
#else /* STRAT_MMSTEP */
#error STRAT_MMSTEP not definable externally
#endif /* ?STRAT_MMSTEP */

// n/2 for even n
#ifndef STRAT0_MAX_PAIRS
#define STRAT0_MAX_PAIRS 16u
#else /* STRAT0_MAX_PAIRS */
#error STRAT0_MAX_PAIRS not definable externally
#endif /* ?STRAT0_MAX_PAIRS */

// n-1 for cyclic, n for quasi-cyclic
#ifndef STRAT0_MAX_STEPS
#define STRAT0_MAX_STEPS ((STRAT0_MAX_PAIRS) * 2u)
#else /* STRAT0_MAX_STEPS */
#error STRAT0_MAX_STEPS not definable externally
#endif /* ?STRAT0_MAX_STEPS */

#ifndef STRAT0_STORAGE
#define STRAT0_STORAGE __constant__
#endif /* !STRAT0_STORAGE */

#ifndef STRAT0_DTYPE
#define STRAT0_DTYPE char
#endif /* !STRAT0_DTYPE */

#ifndef STRAT1_MAX_PAIRS
#define STRAT1_MAX_PAIRS 1024u
#endif /* !STRAT1_MAX_PAIRS */

#ifndef STRAT1_MAX_STEPS
#define STRAT1_MAX_STEPS ((STRAT1_MAX_PAIRS) * 2u)
#else /* STRAT1_MAX_STEPS */
#error STRAT1_MAX_STEPS not definable externally
#endif /* ?STRAT1_MAX_STEPS */

#ifndef STRAT1_STORAGE
#define STRAT1_STORAGE __device__
#endif /* !STRAT1_STORAGE */

#ifndef STRAT1_DTYPE
#define STRAT1_DTYPE short
#endif /* !STRAT1_DTYPE */

extern unsigned STRAT0, STRAT0_STEPS, STRAT0_PAIRS;
extern unsigned STRAT1, STRAT1_STEPS, STRAT1_PAIRS;

extern unsigned STRAT0_DTYPE strat0[STRAT0_MAX_STEPS][STRAT0_MAX_PAIRS][2u];
extern unsigned STRAT1_DTYPE strat1[STRAT1_MAX_STEPS][STRAT1_MAX_PAIRS][2u];

extern jstrat_common js0, js1;

#ifndef C_ELEMS_PER_BLOCK
#if (defined(PROFILE) && (PROFILE == 0))
#define C_ELEMS_PER_BLOCK 8u
#else /* !PROFILE || PROFILE != 0 */
#define C_ELEMS_PER_BLOCK 2u
#endif /* ?PROFILE */
#else /* C_ELEMS_PER_BLOCK */
#error C_ELEMS_PER_BLOCK not definable externally
#endif /* ?C_ELEMS_PER_BLOCK */

#ifndef C_SHIFTR
#if (defined(PROFILE) && (PROFILE == 0))
#define C_SHIFTR 3u
#else /* !PROFILE || PROFILE != 0 */
#define C_SHIFTR 1u
#endif /* ?PROFILE */
#else /* C_SHIFTR */
#error C_SHIFTR not definable externally
#endif /* ?C_SHIFTR */

#ifndef C_SMALL
#define C_SMALL 0u
#else /* C_SMALL */
#error C_SMALL not definable externally
#endif /* ?C_SMALL */

#ifndef C_BIG
#define C_BIG 1u
#else /* C_BIG */
#error C_BIG not definable externally
#endif /* ?C_BIG */

#ifndef C_SUBPHASE_1
#define C_SUBPHASE_1 2u
#else /* C_SUBPHASE_1 */
#error C_SUBPHASE_1 not definable externally
#endif /* ?C_SUBPHASE_1 */

#ifndef C_SUBPHASE_2
#define C_SUBPHASE_2 3u
#else /* C_SUBPHASE_2 */
#error C_SUBPHASE_2 not definable externally
#endif /* ?C_SUBPHASE_2 */

#ifndef C_SUBPHASE_3
#define C_SUBPHASE_3 4u
#else /* C_SUBPHASE_3 */
#error C_SUBPHASE_3 not definable externally
#endif /* ?C_SUBPHASE_3 */

#ifndef C_SUBPHASE_4
#define C_SUBPHASE_4 5u
#else /* C_SUBPHASE_4 */
#error C_SUBPHASE_4 not definable externally
#endif /* ?C_SUBPHASE_4 */

#ifdef USE_MPI
#ifndef HZ_MAX_DEVICES
#define HZ_MAX_DEVICES 512u
#endif /* !HZ_MAX_DEVICES */

#ifndef STRAT2_MAX_PAIRS
#define STRAT2_MAX_PAIRS HZ_MAX_DEVICES
#else /* STRAT2_MAX_PAIRS */
#error STRAT2_MAX_PAIRS not definable externally
#endif /* ?STRAT2_MAX_PAIRS */

#ifndef STRAT2_MAX_STEPS
#define STRAT2_MAX_STEPS ((STRAT2_MAX_PAIRS) * 2u)
#else /* STRAT2_MAX_STEPS */
#error STRAT2_MAX_STEPS not definable externally
#endif /* ?STRAT2_MAX_STEPS */

#ifndef STRAT2_DTYPE
#define STRAT2_DTYPE short
#endif /* !STRAT2_DTYPE */

extern unsigned STRAT2, STRAT2_STEPS, STRAT2_PAIRS;
extern STRAT2_DTYPE strat2[STRAT2_MAX_STEPS][STRAT2_MAX_PAIRS][2u][2u];
extern jstrat_common js2;

EXTERN_C void init_strats(const unsigned snp0, const unsigned n0, const unsigned snp1, const unsigned n1, const unsigned snp2, const unsigned n2) throw();
#else /* !USE_MPI */
EXTERN_C void init_strats(const unsigned snp0, const unsigned n0, const unsigned snp1, const unsigned n1) throw();
#endif /* ?USE_MPI */

EXTERN_C void free_strats() throw();

#endif /* !HZ_L_HPP */
