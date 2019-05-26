#include "HZ_L3.hpp"

#include "HZ_L2.hpp"

int HZ_L3
(const unsigned routine,    // IN, routine ID, <= 15, (Bb__)_2,
 // bits B, b: block-oriented (else, full-block), level 1 and 2;
 const size_t gpu,          // IN, GPU ID (0 <= gpu < gpus);
 const size_t gpus,         // IN, number of GPUs;
 const size_t mF,           // IN, number of rows of F, == 0 (mod 64);
 const size_t mG,           // IN, number of rows of G, == 0 (mod 64);
 const size_t n,            // IN, number of columns, <= min(mF, mG), == 0 (mod 32);
 const size_t n_gpu,        // IN, number of columns per GPU (2 * n_col);
 const size_t n_col,        // IN, number of columns in a block column;
 cuD *const hFD,            // INOUT, ldhF x n_gpu host array in Fortran order;
 cuJ *const hFJ,            // INOUT, ldhF x n_gpu host array in Fortran order;
 const size_t ldhF,         // IN, leading dimension of F, >= mF;
 cuD *const hGD,            // INOUT, ldhG x n_gpu host array in Fortran order;
 cuJ *const hGJ,            // INOUT, ldhG x n_gpu host array in Fortran order;
 const size_t ldhG,         // IN, leading dimension of G, >= mG;
 cuD *const hVD,            // OUT, ldhV x n_gpu host array in Fortran order;
 cuJ *const hVJ,            // OUT, ldhV x n_gpu host array in Fortran order;
 const size_t ldhV,         // IN, leading dimension of V, >= n;
 double *const hS,          // OUT, the generalized singular values, optionally sorted in descending order;
 double *const hH,          // ||F_i||_2/sqrt(||F_i||_2^2 + ||G_i||_2^2);
 double *const hK,          // ||G_i||_2/sqrt(||F_i||_2^2 + ||G_i||_2^2);
 unsigned &glbSwp,          // OUT, number of sweeps at the outermost level;
 unsigned long long &glb_s, // OUT, number of rotations;
 unsigned long long &glb_b, // OUT, number of ``big'' rotations;
 double *const timing       // OUT, optional, in seconds, double[4];
) throw()
{
  return 0;
}
