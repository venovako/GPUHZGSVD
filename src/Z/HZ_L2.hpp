#ifndef HZ_L2_HPP
#define HZ_L2_HPP

#include "defines.hpp"

extern int // 0 if OK, < 0 if invalid argument, > 0 if error
HZ_L2
(const unsigned routine,          // IN, routine ID, <= 15, (B___)_2
 // B: block-oriented or full-block
 const unsigned nrowF,            // IN, number of rows of F, == 0 (mod 64)
 const unsigned nrowG,            // IN, number of rows of G, == 0 (mod 64)
 const unsigned ncol,             // IN, number of columns of <= min(nrowF, nrowG), == 0 (mod 32)
 cuD *const hFD,                  // INOUT, ldhF x ncol host array in Fortran order,
 cuJ *const hFJ,                  // INOUT, ldhF x ncol host array in Fortran order,
 const unsigned ldhF,             // IN, leading dimension of F, >= nrowF
 cuD *const hGD,                  // INOUT, ldhG x ncol host array in Fortran order,
 cuJ *const hGJ,                  // INOUT, ldhG x ncol host array in Fortran order,
 const unsigned ldhG,             // IN, leading dimension of G, >= nrowG
 cuD *const hVD,                  // OUT, ldhV x ncol host array in Fortran order,
 cuJ *const hVJ,                  // OUT, ldhV x ncol host array in Fortran order,
 const unsigned ldhV,             // IN, leading dimension of V, >= ncol
 double *const hS,                // OUT, the generalized singular values, optionally sorted in descending order
 double *const hH,                // ||F_i||_2/sqrt(||F_i||_2^2 + ||G_i||_2^2)
 double *const hK,                // ||G_i||_2/sqrt(||F_i||_2^2 + ||G_i||_2^2)
 unsigned *const glbSwp,          // OUT, number of sweeps at the outermost level
 unsigned long long *const glb_s, // OUT, number of rotations
 unsigned long long *const glb_b, // OUT, number of ``big'' rotations
 double *const timing             // OUT, optional, in seconds, double[4] ==
 // WALL, SETUP & HOST ==> GPUs, COMPUTATION, CLEANUP & GPUs ==> HOST
) throw();

#endif // !HZ_L2_HPP
