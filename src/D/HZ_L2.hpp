#ifndef HZ_L2_HPP
#define HZ_L2_HPP

#include "defines.hpp"

#ifdef ANIMATE
#ifdef USE_MPI
#error Animation not supported with MPI
#else /* !USE_MPI */
#include "vn_lib.h"
#endif /* ?USE_MPI */
#endif /* ANIMATE */

EXTERN_C int // 0 if OK, < 0 if invalid argument, > 0 if error
HZ_L2_gpu
(const unsigned routine,    // IN, routine ID, <= 15, (B___)_2,
 // B: block-oriented (else, full-block);
 const unsigned ncol,       // IN, number of columns, <= min(nrowF, nrowG), == 0 (mod 32);
#ifdef ANIMATE
 const unsigned nrowF,      // IN, number of rows of F, == 0 (mod 64);
 const unsigned nrowG,      // IN, number of rows of G, == 0 (mod 64);
 double *const hF,          // INOUT, ldhF x ncol host array in Fortran order;
 const unsigned ldhF,       // IN, leading dimension of hF, >= nrowF;
 double *const dF,          // INOUT, ldhF x ncol device array in Fortran order;
 const unsigned lddF,       // IN, leading dimension of dF, >= nrowF;
 double *const hG,          // INOUT, ldhG x ncol host array in Fortran order;
 const unsigned ldhG,       // IN, leading dimension of fG, >= nrowG;
 double *const dG,          // INOUT, ldhG x ncol device array in Fortran order;
 const unsigned lddG,       // IN, leading dimension of dG, >= nrowG;
#endif /* ANIMATE */
 unsigned long long *const hC, // OUT, convergence vector
 unsigned long long *const dC, // OUT, convergence vector
 unsigned &glbSwp,          // OUT, number of sweeps at the outermost level;
 unsigned long long &glb_s, // OUT, number of rotations;
 unsigned long long &glb_b  // OUT, number of ``big'' rotations;
#ifdef ANIMATE
#if (ANIMATE == 1)
 , vn_mtxvis_ctx *const ctx
#elif (ANIMATE == 2)
 , vn_mtxvis_ctx *const ctxF
 , vn_mtxvis_ctx *const ctxG
#endif /* ?ANIMATE */
#endif /* ANIMATE */
) throw();

EXTERN_C int // 0 if OK, < 0 if invalid argument, > 0 if error
HZ_L2
(const unsigned routine,    // IN, routine ID, <= 15, (B___)_2,
 // B: block-oriented (else, full-block);
 const unsigned nrowF,      // IN, number of rows of F, == 0 (mod 64);
 const unsigned nrowG,      // IN, number of rows of G, == 0 (mod 64);
 const unsigned ncol,       // IN, number of columns, <= min(nrowF, nrowG), == 0 (mod 32);
 double *const hF,          // INOUT, ldhF x ncol host array in Fortran order;
 const unsigned ldhF,       // IN, leading dimension of F, >= nrowF;
 double *const hG,          // INOUT, ldhG x ncol host array in Fortran order;
 const unsigned ldhG,       // IN, leading dimension of G, >= nrowG;
 double *const hV,          // INOUT, ldhV x ncol host array in Fortran order;
 const unsigned ldhV,       // IN, leading dimension of V, >= ncol;
 double *const hS,          // OUT, the generalized singular values, optionally sorted in descending order;
 double *const hH,          // ||F_i||_2/sqrt(||F_i||_2^2 + ||G_i||_2^2);
 double *const hK,          // ||G_i||_2/sqrt(||F_i||_2^2 + ||G_i||_2^2);
 unsigned &glbSwp,          // OUT, number of sweeps at the outermost level;
 unsigned long long &glb_s, // OUT, number of rotations;
 unsigned long long &glb_b, // OUT, number of ``big'' rotations;
 double *const timing       // OUT, optional, in seconds, double[4] ==
 // WALL, SETUP & HOST ==> GPUs, COMPUTATION, CLEANUP & GPUs ==> HOST;
) throw();

#endif /* !HZ_L2_HPP */
