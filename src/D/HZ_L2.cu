#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L2.hpp"

#include "device_code.hpp"
#include "cuda_helper.hpp"
#include "cuda_memory_helper.hpp"
#include "my_utils.hpp"

int // 0 if OK, < 0 if invalid argument, > 0 if error
HZ_L2_gpu
(const unsigned routine,          // IN, routine ID, <= 15, (B___)_2
 // B: block-oriented or full-block
 const unsigned nrowF,            // IN, number of rows of F, == 0 (mod 64)
 const unsigned nrowG,            // IN, number of rows of G, == 0 (mod 64)
 const unsigned ncol,             // IN, number of columns <= min(nrowF, nrowG), == 0 (mod 32)
 double *const hF,                // INOUT, ldhF x ncol host array in Fortran order,
 const unsigned ldhF,             // IN, leading dimension of hF, >= nrowF
 double *const dF,                // INOUT, ldhF x ncol device array in Fortran order,
 const unsigned lddF,             // IN, leading dimension of dF, >= nrowF
 double *const hG,                // INOUT, ldhG x ncol host array in Fortran order,
 const unsigned ldhG,             // IN, leading dimension of fG, >= nrowG
 double *const dG,                // INOUT, ldhG x ncol device array in Fortran order,
 const unsigned lddG,             // IN, leading dimension of dG, >= nrowG
 double *const hV,                // OUT, ldhV x ncol host array in Fortran order,
 const unsigned ldhV,             // IN, leading dimension of hV, >= ncol
 double *const dV,                // OUT, ldhV x ncol device array in Fortran order,
 const unsigned lddV,             // IN, leading dimension of dV, >= ncol
 double *const hS,                // OUT, the generalized singular values, optionally sorted in descending order
 double *const dS,                // OUT, the generalized singular values, optionally sorted in descending order
 double *const hH,                // ||F_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2)
 double *const dH,                // ||F_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2)
 double *const hK,                // ||G_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2)
 double *const dK,                // ||G_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2)
 unsigned *const glbSwp,          // OUT, number of sweeps at the outermost level
 unsigned long long *const glb_s, // OUT, number of rotations
 unsigned long long *const glb_b  // OUT, number of ``big'' rotations
#ifdef ANIMATE
#if (ANIMATE == 1)
 , vn_mtxvis_ctx *const ctx
#elif (ANIMATE == 2)
 , vn_mtxvis_ctx *const ctxF
 , vn_mtxvis_ctx *const ctxG
#endif // ?ANIMATE
#endif // ANIMATE
) throw()
{
  const bool blk_ori = (routine & HZ_BLK_ORI);
  initSymbols(dF, dG, dV, dS, dH, dK, nrowF, nrowG, ncol, static_cast<unsigned>(lddF), static_cast<unsigned>(lddG), static_cast<unsigned>(lddV), (blk_ori ? 1u : HZ_NSWEEP));
  CUDA_CALL(cudaDeviceSynchronize());

  const int sclV = ((CVG == 0) || (CVG == 1) || (CVG == 4) || (CVG == 5));
  initV(sclV, ncol);
  CUDA_CALL(cudaDeviceSynchronize());

  void (*const HZ_L1)(const unsigned) = HZ_L1_sv;

  *glb_s = 0ull;
  *glb_b = 0ull;
  long long swp_tim = 0ll;
  stopwatch_reset(swp_tim);

  const unsigned swp = HZ_NSWEEP;
  unsigned blk_swp = 0u;

  // stats per thread block
  const unsigned spb = 2u;
  // stats count
  const unsigned sc = STRAT1_PAIRS * spb;

  while (blk_swp < swp) {
    for (unsigned blk_stp = 0u; blk_stp < STRAT1_STEPS; ++blk_stp) {
      if (blk_stp)
        CUDA_CALL(cudaDeviceSynchronize());
      HZ_L1(blk_stp);

#ifdef ANIMATE
#if (ANIMATE == 1)
      if (ctx) {
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
        SYSI_CALL(vn_mtxvis_frame(ctx, hF, ldhF));
        SYSI_CALL(vn_mtxvis_frame(ctx, hG, ldhG));
      }
#elif (ANIMATE == 2)
      if (ctxF) {
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
        SYSI_CALL(vn_mtxvis_frame(ctxF, hF, ldhF));
      }
      if (ctxG) {
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
        SYSI_CALL(vn_mtxvis_frame(ctxG, hG, ldhG));
      }
#endif // ?ANIMATE
#endif // ANIMATE
    }

    ++blk_swp;
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(hS, dS, sc * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    unsigned long long cvg_s = 0ull;
    unsigned long long cvg_b = 0ull;
    for (unsigned i = 0u; i < sc; i += spb) {
      cvg_s += ((const unsigned long long*)hS)[i];
      cvg_b += ((const unsigned long long*)hS)[i + 1u];
    }
    *glb_s += cvg_s;
    *glb_b += cvg_b;

    const double tim_s = stopwatch_lap(swp_tim) * TS2S;
    (void)fprintf(stdout, "BLK_SWP(%2u), ROT_S(%10llu), ROT_B(%10llu), TIME(%#12.6f s)\n", blk_swp, cvg_s, cvg_b, tim_s);
    (void)fflush(stdout);
    if (!cvg_b)
      break;

    initS(0, ncol);
    CUDA_CALL(cudaDeviceSynchronize());
  }

  *glbSwp = blk_swp;
  initS(1, ncol);
  CUDA_CALL(cudaDeviceSynchronize());
  return 0;
}

int // 0 if OK, < 0 if invalid argument, > 0 if error
HZ_L2
(const unsigned routine,          // IN, routine ID, <= 15, (B___)_2
 // B: block-oriented or full-block
 const unsigned nrowF,            // IN, number of rows of F, == 0 (mod 64)
 const unsigned nrowG,            // IN, number of rows of G, == 0 (mod 64)
 const unsigned ncol,             // IN, number of columns <= min(nrowF, nrowG), == 0 (mod 32)
 double *const hF,                // INOUT, ldhF x ncol host array in Fortran order,
 const unsigned ldhF,             // IN, leading dimension of F, >= nrowF
 double *const hG,                // INOUT, ldhG x ncol host array in Fortran order,
 const unsigned ldhG,             // IN, leading dimension of G, >= nrowG
 double *const hV,                // OUT, ldhV x ncol host array in Fortran order,
 const unsigned ldhV,             // IN, leading dimension of V, >= ncol
 double *const hS,                // OUT, the generalized singular values, optionally sorted in descending order
 double *const hH,                // ||F_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2)
 double *const hK,                // ||G_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2)
 unsigned *const glbSwp,          // OUT, number of sweeps at the outermost level
 unsigned long long *const glb_s, // OUT, number of rotations
 unsigned long long *const glb_b, // OUT, number of ``big'' rotations
 double *const timing             // OUT, optional, in seconds, double[4] ==
 // WALL, SETUP & HOST ==> GPUs, COMPUTATION, CLEANUP & GPUs ==> HOST
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

  if (!hF)
    return -5;
  if (ldhF < nrowF)
    return -6;

  if (!hG)
    return -7;
  if (ldhG < nrowG)
    return -8;

  if (!hV)
    return -9;
  if (ldhV < ncol)
    return -10;

  if (!hS)
    return -11;
  if (!hH)
    return -12;
  if (!hK)
    return -13;

  if (!glbSwp)
    return -14;
  if (!glb_s)
    return -15;
  if (!glb_b)
    return -16;

  stopwatch_reset(timers[3]);

  size_t lddF = static_cast<size_t>(nrowF);
  double *const dF = allocDeviceMtx<double>(lddF, static_cast<size_t>(nrowF), static_cast<size_t>(ncol), true);

  size_t lddG = static_cast<size_t>(nrowG);
  double *const dG = allocDeviceMtx<double>(lddG, static_cast<size_t>(nrowG), static_cast<size_t>(ncol), true);

  size_t lddV = static_cast<size_t>(ncol);
  double *const dV = allocDeviceMtx<double>(lddV, static_cast<size_t>(ncol), static_cast<size_t>(ncol), true);
  
  double *const dS = allocDeviceVec<double>(static_cast<size_t>(ncol));
  double *const dH = allocDeviceVec<double>(static_cast<size_t>(ncol));
  double *const dK = allocDeviceVec<double>(static_cast<size_t>(ncol));

  CUDA_CALL(cudaMemcpy2DAsync(dF, lddF * sizeof(double), hF, ldhF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy2DAsync(dG, lddG * sizeof(double), hG, ldhG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemset2DAsync(dV, lddV * sizeof(double), 0, ncol * sizeof(double), ncol));
  CUDA_CALL(cudaMemsetAsync(dS, 0, ncol * sizeof(double)));
  CUDA_CALL(cudaMemsetAsync(dH, 0, ncol * sizeof(double)));
  CUDA_CALL(cudaMemsetAsync(dK, 0, ncol * sizeof(double)));
  CUDA_CALL(cudaDeviceSynchronize());

#ifdef ANIMATE
#if (ANIMATE == 1)
  vn_mtxvis_ctx *ctx = static_cast<vn_mtxvis_ctx*>(NULL);
  if (ncol < 10000u) {
    char fname[8] = { '\0' };
    (void)sprintf(fname, "FG%x%04u", routine, ncol);
    SYSI_CALL(vn_mtxvis_start(&ctx, fname, (VN_MTXVIS_OP_AtA | VN_MTXVIS_FN_Lg | VN_MTXVIS_FF_Bin), ncol, ncol, 1, 1, 7));
    if (ctx) {
      SYSI_CALL(vn_mtxvis_frame(ctx, hF, ldhF));
      SYSI_CALL(vn_mtxvis_frame(ctx, hG, ldhG));
      CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());
      SYSI_CALL(vn_mtxvis_frame(ctx, hF, ldhF));
      SYSI_CALL(vn_mtxvis_frame(ctx, hG, ldhG));
    }
  }
#elif (ANIMATE == 2)
  vn_mtxvis_ctx *ctxF = static_cast<vn_mtxvis_ctx*>(NULL);
  vn_mtxvis_ctx *ctxG = static_cast<vn_mtxvis_ctx*>(NULL);
  if (ncol < 10000u) {
    char fname[8] = { '\0' };
    (void)sprintf(fname, "%c%x_%04u", 'F', routine, ncol);
    SYSI_CALL(vn_mtxvis_start(&ctxF, fname, (VN_MTXVIS_OP_AtA | VN_MTXVIS_FN_Lg | VN_MTXVIS_FF_Bin), nrowF, ncol, 1, 1, 7));
    if (ctxF) {
      SYSI_CALL(vn_mtxvis_frame(ctxF, hF, ldhF));
      CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());
      SYSI_CALL(vn_mtxvis_frame(ctxF, hF, ldhF));
    }
    (void)sprintf(fname, "%c%x_%04u", 'G', routine, ncol);
    SYSI_CALL(vn_mtxvis_start(&ctxG, fname, (VN_MTXVIS_OP_AtA | VN_MTXVIS_FN_Lg | VN_MTXVIS_FF_Bin), nrowG, ncol, 1, 1, 7));
    if (ctxG) {
      SYSI_CALL(vn_mtxvis_frame(ctxG, hG, ldhG));
      CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaDeviceSynchronize());
      SYSI_CALL(vn_mtxvis_frame(ctxG, hG, ldhG));
    }
  }
#endif // ?ANIMATE
#endif // ANIMATE

  timers[1] = stopwatch_lap(timers[3]);
  const int ret = HZ_L2_gpu
    (routine, nrowF,nrowG,ncol, hF,ldhF, dF,lddF, hG,ldhG, dG,lddG, hV,ldhV, dV,lddV, hS,dS, hH,dH, hK,dK, glbSwp,glb_s,glb_b
#ifdef ANIMATE
#if (ANIMATE == 1)
     , ctx
#elif (ANIMATE == 2)
     , ctxF,ctxG
#endif // ?ANIMATE
#endif // ANIMATE
     );
  timers[2] = stopwatch_lap(timers[3]);

  CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy2DAsync(hV, ldhV * sizeof(double), dV, lddV * sizeof(double), ncol * sizeof(double), ncol, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpyAsync(hS, dS, ncol * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpyAsync(hH, dH, ncol * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpyAsync(hK, dK, ncol * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaDeviceSynchronize());

#ifdef ANIMATE
#if (ANIMATE == 1)
  if (ctx) {
    SYSI_CALL(vn_mtxvis_frame(ctx, hF, ldhF));
    SYSI_CALL(vn_mtxvis_frame(ctx, hG, ldhG));
    SYSI_CALL(vn_mtxvis_stop(ctx));
  }
#elif (ANIMATE == 2)
  if (ctxF) {
    SYSI_CALL(vn_mtxvis_frame(ctxF, hF, ldhF));
    SYSI_CALL(vn_mtxvis_stop(ctxF));
  }
  if (ctxG) {
    SYSI_CALL(vn_mtxvis_frame(ctxG, hG, ldhG));
    SYSI_CALL(vn_mtxvis_stop(ctxG));
  }
#endif // ?ANIMATE
#endif // ANIMATE

  CUDA_CALL(cudaFree(static_cast<void*>(dK)));
  CUDA_CALL(cudaFree(static_cast<void*>(dH)));
  CUDA_CALL(cudaFree(static_cast<void*>(dS)));
  CUDA_CALL(cudaFree(static_cast<void*>(dV)));
  CUDA_CALL(cudaFree(static_cast<void*>(dG)));
  CUDA_CALL(cudaFree(static_cast<void*>(dF)));

  timers[3] = stopwatch_lap(timers[3]);
  timers[0] = stopwatch_lap(timers[0]);

  if (timing)
    for (unsigned i = 0u; i < 4u; ++i)
      timing[i] = timers[i] * TS2S;

  return ret;
}
