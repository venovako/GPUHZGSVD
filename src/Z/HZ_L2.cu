#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L2.hpp"

#include "device_code.hpp"
#include "cuda_memory_helper.hpp"

int // 0 if OK, < 0 if invalid argument, > 0 if error
HZ_L2_gpu
(const unsigned routine,    // IN, routine ID, <= 15, (B_N_)_2,
 // B: block-oriented (else, full-block), N: no sort;
 const unsigned ncol,       // IN, number of columns, <= min(nrowF, nrowG), == 0 (mod 32);
#ifdef ANIMATE
 const unsigned nrowF,      // IN, number of rows of F, == 0 (mod 64);
 const unsigned nrowG,      // IN, number of rows of G, == 0 (mod 64);
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
#endif // ANIMATE
 unsigned long long *const hC, // OUT, convergence vector
 unsigned long long *const dC, // OUT, convergence vector
 unsigned &glbSwp,          // OUT, number of sweeps at the outermost level;
 unsigned long long &glb_s, // OUT, number of rotations;
 unsigned long long &glb_b  // OUT, number of ``big'' rotations;
#ifdef ANIMATE
 , vn_cmplxvis_ctx *const ctx
 , std::complex<double> *const hDJ
 , const size_t nrow
#endif // ANIMATE
 ) throw()
{
  void (*const HZ_L1)(const unsigned) = ((routine & 2u) ? HZ_L1_v : HZ_L1_sv);

  const unsigned swp = ((routine & HZ_BO_2) ? 1u : HZ_NSWEEP);
  // stats per thread block
  const unsigned spb = 2u;
  // stats count
  const unsigned sc = STRAT1_PAIRS * spb;
  // stats len
  const size_t sl = sc * sizeof(unsigned long long);

  glb_s = 0ull;
  glb_b = 0ull;
#ifndef USE_MPI
  long long swp_tim = 0ll;
  stopwatch_reset(swp_tim);
#endif // !USE_MPI

  unsigned blk_swp = 0u;
  while (blk_swp < swp) {
    CUDA_CALL(cudaMemset(dC, 0, sl));
    CUDA_CALL(cudaDeviceSynchronize());
    for (unsigned blk_stp = 0u; blk_stp < STRAT1_STEPS; ++blk_stp) {
      if (blk_stp)
        CUDA_CALL(cudaDeviceSynchronize());
      HZ_L1(blk_stp);
#ifdef ANIMATE
      if (ctx) {
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy2D(hFD, ldhF * sizeof(double), dFD, lddF * sizeof(cuD), nrowF * sizeof(cuD), ncol, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy2D(hFJ, ldhF * sizeof(double), dFJ, lddF * sizeof(cuJ), nrowF * sizeof(cuJ), ncol, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy2D(hGD, ldhG * sizeof(double), dGD, lddG * sizeof(cuD), nrowG * sizeof(cuD), ncol, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy2D(hGJ, ldhG * sizeof(double), dGJ, lddG * sizeof(cuJ), nrowG * sizeof(cuJ), ncol, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());

        for (unsigned j = 0u; j < ncol; ++j) {
          const size_t offDJ = ldhDJ * j;
          const size_t offhF = ldhF * j;
          for (unsigned i = 0u; i < nrow; ++i) {
            const size_t ixDJ = offDJ + i;
            const size_t ixhF = offhF + i;
            hDJ[ixDJ].real(hFD[ixhF]);
            hDJ[ixDJ].imag(hFJ[ixhF]);
          }
        }
        SYSI_CALL(vn_cmplxvis_frame(ctx, (const vn_complex*)hDJ, nrow));

        for (unsigned j = 0u; j < ncol; ++j) {
          const size_t offDJ = ldhDJ * j;
          const size_t offhG = ldhG * j;
          for (unsigned i = 0u; i < nrow; ++i) {
            const size_t ixDJ = offDJ + i;
            const size_t ixhG = offhG + i;
            hDJ[ixDJ].real(hGD[ixhG]);
            hDJ[ixDJ].imag(hGJ[ixhG]);
          }
        }
        SYSI_CALL(vn_cmplxvis_frame(ctx, (const vn_complex*)hDJ, nrow));
      }
#endif // ANIMATE
    }

    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(hC, dC, sl, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    unsigned long long cvg_s = 0ull;
    unsigned long long cvg_b = 0ull;
    for (unsigned i = 0u; i < sc; i += spb) {
      cvg_s += hC[i];
      cvg_b += hC[i + 1u];
    }
    glb_s += cvg_s;
    glb_b += cvg_b;

#ifndef USE_MPI
    const double tim_s = stopwatch_lap(swp_tim) * TS2S;
    (void)fprintf(stdout, "BLK_SWP(%2u), ROT_S(%13llu), ROT_B(%13llu), TIME(%#14.6f s)\n", blk_swp, cvg_s, cvg_b, tim_s);
    (void)fflush(stdout);
#endif // !USE_MPI
    if (!cvg_b)
      break;
    ++blk_swp;
    initS(0, ncol);
    CUDA_CALL(cudaDeviceSynchronize());
#ifdef ANIMATE
    if (ctx) {
      CUDA_CALL(cudaMemcpy2D(hFD, ldhF * sizeof(double), dFD, lddF * sizeof(cuD), nrowF * sizeof(cuD), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy2D(hFJ, ldhF * sizeof(double), dFJ, lddF * sizeof(cuJ), nrowF * sizeof(cuJ), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy2D(hGD, ldhG * sizeof(double), dGD, lddG * sizeof(cuD), nrowG * sizeof(cuD), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy2D(hGJ, ldhG * sizeof(double), dGJ, lddG * sizeof(cuJ), nrowG * sizeof(cuJ), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());

      for (unsigned j = 0u; j < ncol; ++j) {
        const size_t offDJ = ldhDJ * j;
        const size_t offhF = ldhF * j;
        for (unsigned i = 0u; i < nrow; ++i) {
          const size_t ixDJ = offDJ + i;
          const size_t ixhF = offhF + i;
          hDJ[ixDJ].real(hFD[ixhF]);
          hDJ[ixDJ].imag(hFJ[ixhF]);
        }
      }
      SYSI_CALL(vn_cmplxvis_frame(ctx, (const vn_complex*)hDJ, nrow));

      for (unsigned j = 0u; j < ncol; ++j) {
        const size_t offDJ = ldhDJ * j;
        const size_t offhG = ldhG * j;
        for (unsigned i = 0u; i < nrow; ++i) {
          const size_t ixDJ = offDJ + i;
          const size_t ixhG = offhG + i;
          hDJ[ixDJ].real(hGD[ixhG]);
          hDJ[ixDJ].imag(hGJ[ixhG]);
        }
      }
      SYSI_CALL(vn_cmplxvis_frame(ctx, (const vn_complex*)hDJ, nrow));
    }
#endif // ANIMATE
  }

  if (blk_swp < swp)
    glbSwp = (blk_swp + 1u);
  else
    glbSwp = blk_swp;
#ifdef USE_MPI
  if (blk_swp < swp)
    initS(0, ncol);
#else // !USE_MPI
  initS(1, ncol);
#endif // !USE_MPI
  CUDA_CALL(cudaDeviceSynchronize());
  return 0;
}

int // 0 if OK, < 0 if invalid argument, > 0 if error
HZ_L2
(const unsigned routine,    // IN, routine ID, <= 15, (B_N_)_2,
 // B: block-oriented (else, full-block), N: no sort;
 const unsigned nrowF,      // IN, number of rows of F, == 0 (mod 64);
 const unsigned nrowG,      // IN, number of rows of G, == 0 (mod 64);
 const unsigned ncol,       // IN, number of columns, <= min(nrowF, nrowG), == 0 (mod 32);
 cuD *const hFD,            // INOUT, ldhF x ncol host array in Fortran order;
 cuJ *const hFJ,            // INOUT, ldhF x ncol host array in Fortran order;
 const unsigned ldhF,       // IN, leading dimension of F, >= nrowF;
 cuD *const hGD,            // INOUT, ldhG x ncol host array in Fortran order;
 cuJ *const hGJ,            // INOUT, ldhG x ncol host array in Fortran order;
 const unsigned ldhG,       // IN, leading dimension of G, >= nrowG;
 cuD *const hVD,            // INOUT, ldhV x ncol host array in Fortran order;
 cuJ *const hVJ,            // INOUT, ldhV x ncol host array in Fortran order;
 const unsigned ldhV,       // IN, leading dimension of V, >= ncol;
 double *const hS,          // OUT, the generalized singular values, optionally sorted in descending order;
 double *const hH,          // ||F_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2);
 double *const hK,          // ||G_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2);
 unsigned &glbSwp,          // OUT, number of sweeps at the outermost level;
 unsigned long long &glb_s, // OUT, number of rotations;
 unsigned long long &glb_b, // OUT, number of ``big'' rotations;
 double *const timing       // OUT, optional, in seconds, double[4] ==
 // WALL, SETUP & HOST ==> GPUs, COMPUTATION, CLEANUP & GPUs ==> HOST;
) throw()
{
  long long timers[4] = { 0ll };
  stopwatch_reset(timers[0]);

  if (routine >= 16u)
    return -1;

  if (!nrowF || (nrowF % 64u))
    return -2;
  if (!nrowG || (nrowG % 64u))
    return -3;
  if (!ncol || (ncol > nrowF) || (ncol > nrowG) || (ncol % 32u))
    return -4;

  if (!hFD)
    return -5;
  if (!hFJ)
    return -6;
  if (ldhF < nrowF)
    return -7;

  if (!hGD)
    return -8;
  if (!hGJ)
    return -9;
  if (ldhG < nrowG)
    return -10;

  if (!hVD)
    return -11;
  if (!hVJ)
    return -12;
  if (ldhV < ncol)
    return -13;

  if (!hS)
    return -14;
  if (!hH)
    return -15;
  if (!hK)
    return -16;

  stopwatch_reset(timers[3]);

  size_t lddF = static_cast<size_t>(nrowF);
  cuD *const dFD = allocDeviceMtx<cuD>(lddF, static_cast<size_t>(nrowF), static_cast<size_t>(ncol), true);
  cuJ *const dFJ = allocDeviceMtx<cuJ>(lddF, static_cast<size_t>(nrowF), static_cast<size_t>(ncol), true);

  size_t lddG = static_cast<size_t>(nrowG);
  cuD *const dGD = allocDeviceMtx<cuD>(lddG, static_cast<size_t>(nrowG), static_cast<size_t>(ncol), true);
  cuJ *const dGJ = allocDeviceMtx<cuJ>(lddG, static_cast<size_t>(nrowG), static_cast<size_t>(ncol), true);

  size_t lddV = static_cast<size_t>(ncol);
  cuD *const dVD = allocDeviceMtx<cuD>(lddV, static_cast<size_t>(ncol), static_cast<size_t>(ncol), true);
  cuJ *const dVJ = allocDeviceMtx<cuJ>(lddV, static_cast<size_t>(ncol), static_cast<size_t>(ncol), true);

  double *const dS = allocDeviceVec<double>(static_cast<size_t>(ncol));
  double *const dH = allocDeviceVec<double>(static_cast<size_t>(ncol));
  double *const dK = allocDeviceVec<double>(static_cast<size_t>(ncol));

  unsigned long long *const dC = allocDeviceVec<unsigned long long>(static_cast<size_t>(STRAT1_PAIRS) * 2u);
  unsigned long long *const hC = allocHostVec<unsigned long long>(static_cast<size_t>(STRAT1_PAIRS) * 2u);

  initSymbols(dFD,dFJ, dGD,dGJ, dVD,dVJ, dS,dH,dK, dC, nrowF,nrowG,ncol,ncol, lddF,lddG,lddV, ((routine & HZ_BO_1) ? 1u : HZ_NSWEEP));
  CUDA_CALL(cudaMemcpy2D(dFD, lddF * sizeof(cuD), hFD, ldhF * sizeof(double), nrowF * sizeof(cuD), ncol, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy2D(dFJ, lddF * sizeof(cuJ), hFJ, ldhF * sizeof(double), nrowF * sizeof(cuJ), ncol, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy2D(dGD, lddG * sizeof(cuD), hGD, ldhG * sizeof(double), nrowG * sizeof(cuD), ncol, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy2D(dGJ, lddG * sizeof(cuJ), hGJ, ldhG * sizeof(double), nrowG * sizeof(cuJ), ncol, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy2D(dVD, lddV * sizeof(cuD), hVD, ldhV * sizeof(double), ncol * sizeof(cuD), ncol, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy2D(dVJ, lddV * sizeof(cuJ), hVJ, ldhV * sizeof(double), ncol * sizeof(cuJ), ncol, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaDeviceSynchronize());
#ifdef USE_MPI
  const unsigned ifc0 = 0u;
  const unsigned ifc1 = (ncol >> 1u);
  initV(((CVG == 0) || (CVG == 1) || (CVG == 4) || (CVG == 5)), ncol, ifc0, ifc1);
#else // !USE_MPI
  initV(((CVG == 0) || (CVG == 1) || (CVG == 4) || (CVG == 5)), ncol);
#endif // USE_MPI
  CUDA_CALL(cudaDeviceSynchronize());

#ifdef ANIMATE
  vn_cmplxvis_ctx *ctx = static_cast<vn_cmplxvis_ctx*>(NULL);
  std::complex<double> *hDJ = static_cast<std::complex<double>>(NULL);
  size_t nrow = 0u;
  // it is meant to work only for nrowF == nrowG
  if (nrowF == nrowG) {
    nrow = nrowF;
    hDJ = allocHostMtx<std::complex<double>>(nrow, nrow, static_cast<size_t>(ncol), true);
  }
  if (ncol < 10000u) {
    char fname[8] = { '\0' };
    (void)sprintf(fname, "FG%x%04u", routine, ncol);
    if (hDJ)
      SYSI_CALL(vn_cmplxvis_start(&ctx, fname, (VN_CMPLXVIS_OP_AhA | VN_CMPLXVIS_FN_Lg), ncol, ncol, 1, 1, 7));
    if (ctx) {
      CUDA_CALL(cudaMemcpy2D(hFD, ldhF * sizeof(double), dFD, lddF * sizeof(cuD), nrowF * sizeof(cuD), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy2D(hFJ, ldhF * sizeof(double), dFJ, lddF * sizeof(cuJ), nrowF * sizeof(cuJ), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy2D(hGD, ldhG * sizeof(double), dGD, lddG * sizeof(cuD), nrowG * sizeof(cuD), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy2D(hGJ, ldhG * sizeof(double), dGJ, lddG * sizeof(cuJ), nrowG * sizeof(cuJ), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());

      for (unsigned j = 0u; j < ncol; ++j) {
        const size_t offDJ = ldhDJ * j;
        const size_t offhF = ldhF * j;
        for (unsigned i = 0u; i < nrow; ++i) {
          const size_t ixDJ = offDJ + i;
          const size_t ixhF = offhF + i;
          hDJ[ixDJ].real(hFD[ixhF]);
          hDJ[ixDJ].imag(hFJ[ixhF]);
        }
      }
      SYSI_CALL(vn_cmplxvis_frame(ctx, (const vn_complex*)hDJ, nrow));

      for (unsigned j = 0u; j < ncol; ++j) {
        const size_t offDJ = ldhDJ * j;
        const size_t offhG = ldhG * j;
        for (unsigned i = 0u; i < nrow; ++i) {
          const size_t ixDJ = offDJ + i;
          const size_t ixhG = offhG + i;
          hDJ[ixDJ].real(hGD[ixhG]);
          hDJ[ixDJ].imag(hGJ[ixhG]);
        }
      }
      SYSI_CALL(vn_cmplxvis_frame(ctx, (const vn_complex*)hDJ, nrow));
    }
  }
#endif // ANIMATE

  timers[1] = stopwatch_lap(timers[3]);
  const int ret = HZ_L2_gpu
    (routine,ncol,
#ifdef ANIMATE
     nrowF,nrowG, hFD,hFJ,ldhF, dFD,dFJ,lddF, hGD,hGJ,ldhG, dGD,dGJ,lddG,
#endif // ANIMATE
     hC,dC, glbSwp,glb_s,glb_b
#ifdef ANIMATE
     , ctx,hDJ,nrow
#endif // ANIMATE
     );
  timers[2] = stopwatch_lap(timers[3]);

  CUDA_CALL(cudaMemcpy2D(hFD, ldhF * sizeof(double), dFD, lddF * sizeof(cuD), nrowF * sizeof(cuD), ncol, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy2D(hFJ, ldhF * sizeof(double), dFJ, lddF * sizeof(cuJ), nrowF * sizeof(cuJ), ncol, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy2D(hGD, ldhG * sizeof(double), dGD, lddG * sizeof(cuD), nrowG * sizeof(cuD), ncol, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy2D(hGJ, ldhG * sizeof(double), dGJ, lddG * sizeof(cuJ), nrowG * sizeof(cuJ), ncol, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy2D(hVD, ldhV * sizeof(double), dVD, lddV * sizeof(cuD), ncol * sizeof(cuD), ncol, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy2D(hVJ, ldhV * sizeof(double), dVJ, lddV * sizeof(cuJ), ncol * sizeof(cuJ), ncol, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hS, dS, ncol * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hH, dH, ncol * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hK, dK, ncol * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaDeviceSynchronize());

#ifdef ANIMATE
  if (ctx) {
    for (unsigned j = 0u; j < ncol; ++j) {
      const size_t offDJ = ldhDJ * j;
      const size_t offhF = ldhF * j;
      for (unsigned i = 0u; i < nrow; ++i) {
        const size_t ixDJ = offDJ + i;
        const size_t ixhF = offhF + i;
        hDJ[ixDJ].real(hFD[ixhF]);
        hDJ[ixDJ].imag(hFJ[ixhF]);
      }
    }
    SYSI_CALL(vn_cmplxvis_frame(ctx, (const vn_complex*)hDJ, nrow));

    for (unsigned j = 0u; j < ncol; ++j) {
      const size_t offDJ = ldhDJ * j;
      const size_t offhG = ldhG * j;
      for (unsigned i = 0u; i < nrow; ++i) {
        const size_t ixDJ = offDJ + i;
        const size_t ixhG = offhG + i;
        hDJ[ixDJ].real(hGD[ixhG]);
        hDJ[ixDJ].imag(hGJ[ixhG]);
      }
    }
    SYSI_CALL(vn_cmplxvis_frame(ctx, (const vn_complex*)hDJ, nrow));

    SYSI_CALL(vn_cmplxvis_stop(ctx));
    CUDA_CALL(cudaFreeHost((void*)hDJ));
  }
#endif // ANIMATE

  CUDA_CALL(cudaFreeHost(hC));
  CUDA_CALL(cudaFree(dC));
  CUDA_CALL(cudaFree(dK));
  CUDA_CALL(cudaFree(dH));
  CUDA_CALL(cudaFree(dS));
  CUDA_CALL(cudaFree(dVJ));
  CUDA_CALL(cudaFree(dVD));
  CUDA_CALL(cudaFree(dGJ));
  CUDA_CALL(cudaFree(dGD));
  CUDA_CALL(cudaFree(dFJ));
  CUDA_CALL(cudaFree(dFD));
  CUDA_CALL(cudaDeviceSynchronize());

  timers[3] = stopwatch_lap(timers[3]);
  timers[0] = stopwatch_lap(timers[0]);

  if (timing)
    for (unsigned i = 0u; i < 4u; ++i)
      timing[i] = timers[i] * TS2S;

  return ret;
}
