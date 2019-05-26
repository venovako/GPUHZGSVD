#ifndef HZ_L2_HPP
#define HZ_L2_HPP

#include "defines.hpp"

#ifdef ANIMATE
#ifdef USE_MPI
#error Animation not supported with MPI
#else // !USE_MPI
#include "vn_lib.h"
#endif // ?USE_MPI
#endif // ANIMATE

EXTERN_C int // 0 if OK, < 0 if invalid argument, > 0 if error
HZ_L2_gpu
(unsigned &alg,             // IN, routine ID, <= 15, (B__I)_2,
 // B: block-oriented (else, full-block), I: init symbols (else, keep the previous ones);
 const unsigned nrowF,      // IN, number of rows of F, == 0 (mod 64);
 const unsigned nrowG,      // IN, number of rows of G, == 0 (mod 64);
 const unsigned ncol,       // IN, number of columns, <= min(nrowF, nrowG), == 0 (mod 32);
#ifdef USE_MPI
 const unsigned ifc0,       // IN, index of the first column in the first block column;
 const unsigned ifc1,       // IN, index of the first column in the second block column;
#endif // USE_MPI
 cuD *const hFD,            // INOUT, ldhF x ncol host array in Fortran order;
 cuJ *const hFJ,            // INOUT, ldhF x ncol host array in Fortran order;
 const unsigned ldhF,       // IN, leading dimension of hF, >= nrowF;
 cuD *const dFD,            // INOUT, lddF x ncol device array in Fortran order;
 cuJ *const dFJ,            // INOUT, lddF x ncol device array in Fortran order;
 const unsigned lddF,       // IN, leading dimension of dF, >= nrowF;
 cuD *const hGD,            // INOUT, ldhG x ncol host array in Fortran order;
 cuJ *const hGJ,            // INOUT, ldhG x ncol host array in Fortran order;
 const unsigned ldhG,       // IN, leading dimension of hG, >= nrowG;
 cuD *const dGD,            // INOUT, lddG x ncol device array in Fortran order;
 cuJ *const dGJ,            // INOUT, lddG x ncol device array in Fortran order;
 const unsigned lddG,       // IN, leading dimension of dG, >= nrowG;
 cuD *const hVD,            // OUT, ldhV x ncol host array in Fortran order;
 cuJ *const hVJ,            // OUT, ldhV x ncol host array in Fortran order;
 const unsigned ldhV,       // IN, leading dimension of hV, >= ncol;
 cuD *const dVD,            // OUT, lddV x ncol host array in Fortran order;
 cuJ *const dVJ,            // OUT, lddV x ncol host array in Fortran order;
 const unsigned lddV,       // IN, leading dimension of dV, >= ncol;
 double *const hS,          // OUT, the generalized singular values, optionally sorted in descending order;
 double *const dS,          // OUT, the generalized singular values, optionally sorted in descending order;
 double *const hH,          // ||F_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2);
 double *const dH,          // ||F_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2);
 double *const hK,          // ||G_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2);
 double *const dK,          // ||G_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2);
 unsigned &glbSwp,          // OUT, number of sweeps at the outermost level;
 unsigned long long &glb_s, // OUT, number of rotations;
 unsigned long long &glb_b  // OUT, number of ``big'' rotations;
#ifdef ANIMATE
 , vn_cmplxvis_ctx *const ctx
 , std::complex<double> *const hDJ
 , const size_t nrow
#endif // ANIMATE
) throw();

EXTERN_C int // 0 if OK, < 0 if invalid argument, > 0 if error
HZ_L2
(const unsigned routine,    // IN, routine ID, <= 15, (B___)_2,
 // B: block-oriented (else, full-block);
 const unsigned nrowF,      // IN, number of rows of F, == 0 (mod 64);
 const unsigned nrowG,      // IN, number of rows of G, == 0 (mod 64);
 const unsigned ncol,       // IN, number of columns, <= min(nrowF, nrowG), == 0 (mod 32);
 cuD *const hFD,            // INOUT, ldhF x ncol host array in Fortran order;
 cuJ *const hFJ,            // INOUT, ldhF x ncol host array in Fortran order;
 const unsigned ldhF,       // IN, leading dimension of F, >= nrowF;
 cuD *const hGD,            // INOUT, ldhG x ncol host array in Fortran order;
 cuJ *const hGJ,            // INOUT, ldhG x ncol host array in Fortran order;
 const unsigned ldhG,       // IN, leading dimension of G, >= nrowG;
 cuD *const hVD,            // OUT, ldhV x ncol host array in Fortran order;
 cuJ *const hVJ,            // OUT, ldhV x ncol host array in Fortran order;
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

#endif // !HZ_L2_HPP
