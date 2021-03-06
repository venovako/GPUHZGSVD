#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L2.hpp"

#include "device_code.hpp"
#include "cuda_helper.hpp"
#include "cuda_memory_helper.hpp"
#include "my_utils.hpp"

int // 0 if OK, < 0 if invalid argument, > 0 if error
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
 , const cudaStream_t s
) throw()
{
  void (*const HZ_L1)(const unsigned, const cudaStream_t) = HZ_L1_sv;

  const unsigned swp = ((routine & HZ_BO_2) ? 1u : HZ_NSWEEP);
  // stats count
  const unsigned sc = STRAT1_PAIRS * C_ELEMS_PER_BLOCK;
  // stats len
  const size_t sl = sc * sizeof(unsigned long long);

  glb_s = 0ull;
  glb_b = 0ull;
#if (defined(PROFILE) && (PROFILE == 0))
  unsigned long long CLK_1 = 0ull;
  unsigned long long CLK_2 = 0ull;
  unsigned long long CLK_3 = 0ull;
  unsigned long long CLK_4 = 0ull;
#endif /* ?PROFILE */

#ifndef USE_MPI
  long long swp_tim = 0ll;
  stopwatch_reset(swp_tim);
#endif /* !USE_MPI */

  unsigned blk_swp = 0u;
  while (blk_swp < swp) {
    CUDA_CALL(cudaMemsetAsync(dC, 0, sl, s));
    CUDA_CALL(cudaStreamSynchronize(s));
    for (unsigned blk_stp = 0u; blk_stp < STRAT1_STEPS; ++blk_stp) {
      if (blk_stp)
        CUDA_CALL(cudaStreamSynchronize(s));
      HZ_L1(blk_stp, s);

#ifdef ANIMATE
#if (ANIMATE == 1)
      if (ctx) {
        CUDA_CALL(cudaStreamSynchronize(s));
        CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
        CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
        CUDA_CALL(cudaStreamSynchronize(s));
        SYSI_CALL(vn_mtxvis_frame(ctx, hF, ldhF));
        SYSI_CALL(vn_mtxvis_frame(ctx, hG, ldhG));
      }
#elif (ANIMATE == 2)
      if (ctxF) {
        CUDA_CALL(cudaStreamSynchronize(s));
        CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
        CUDA_CALL(cudaStreamSynchronize(s));
        SYSI_CALL(vn_mtxvis_frame(ctxF, hF, ldhF));
      }
      if (ctxG) {
        CUDA_CALL(cudaStreamSynchronize(s));
        CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
        CUDA_CALL(cudaStreamSynchronize(s));
        SYSI_CALL(vn_mtxvis_frame(ctxG, hG, ldhG));
      }
#endif /* ?ANIMATE */
#endif /* ANIMATE */
    }

    CUDA_CALL(cudaStreamSynchronize(s));
    CUDA_CALL(cudaMemcpyAsync(hC, dC, sl, cudaMemcpyDeviceToHost, s));
    CUDA_CALL(cudaStreamSynchronize(s));

    unsigned long long cvg_s = 0ull;
    unsigned long long cvg_b = 0ull;
#if (defined(PROFILE) && (PROFILE == 0))
    unsigned long long clk_1 = 0ull;
    unsigned long long clk_2 = 0ull;
    unsigned long long clk_3 = 0ull;
    unsigned long long clk_4 = 0ull;
#endif /* ?PROFILE */
    for (unsigned i = 0u; i < sc; i += C_ELEMS_PER_BLOCK) {
      cvg_s += hC[i + C_SMALL];
      cvg_b += hC[i + C_BIG];
#if (defined(PROFILE) && (PROFILE == 0))
      if (clk_1 < hC[i + C_SUBPHASE_1])
        clk_1 = hC[i + C_SUBPHASE_1];
      if (clk_2 < hC[i + C_SUBPHASE_2])
        clk_2 = hC[i + C_SUBPHASE_2];
      if (clk_3 < hC[i + C_SUBPHASE_3])
        clk_3 = hC[i + C_SUBPHASE_3];
      if (clk_4 < hC[i + C_SUBPHASE_4])
        clk_4 = hC[i + C_SUBPHASE_4];
#endif /* ?PROFILE */
    }
    glb_s += cvg_s;
    glb_b += cvg_b;
#if (defined(PROFILE) && (PROFILE == 0))
    CLK_1 += clk_1;
    CLK_2 += clk_2;
    CLK_3 += clk_3;
    CLK_4 += clk_4;
#endif /* ?PROFILE */

#ifndef USE_MPI
    const double tim_s = stopwatch_lap(swp_tim) * TS2S;
    (void)fprintf(stdout, "BLK_SWP(%2u), ROT_S(%13llu), ROT_B(%13llu), TIME(%#14.6f s)", blk_swp, cvg_s, cvg_b, tim_s);
#if (defined(PROFILE) && (PROFILE == 0))
    (void)fprintf(stdout, ", clk_1(%11llu), clk_2(%11llu), clk_3(%11llu), clk_4(%11llu)", clk_1, clk_2, clk_3, clk_4);
#endif /* ?PROFILE */
    (void)fprintf(stdout, "\n");
    (void)fflush(stdout);
#endif /* !USE_MPI */
    if (!cvg_b)
      break;
    ++blk_swp;
    initS(0, ncol, s);
    CUDA_CALL(cudaStreamSynchronize(s));
  }

  if (blk_swp < swp)
    glbSwp = (blk_swp + 1u);
  else
    glbSwp = blk_swp;
#ifdef USE_MPI
  if (blk_swp < swp)
    initS(0, ncol, s);
#else /* !USE_MPI */
  initS(1, ncol, s);
#endif /* ?USE_MPI */
  CUDA_CALL(cudaStreamSynchronize(s));

#if (defined(PROFILE) && (PROFILE == 0))
  (void)fprintf(stdout, "CLK_1(%13llu), CLK_2(%13llu), CLK_3(%13llu), CLK_4(%13llu)\n", CLK_1, CLK_2, CLK_3, CLK_4);
  (void)fflush(stdout);
#endif /* ?PROFILE */

  return 0;
}

int // 0 if OK, < 0 if invalid argument, > 0 if error
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
 double *const hH,          // ||F_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2);
 double *const hK,          // ||G_i||_F/sqrt(||F_i||_F^2 + ||G_i||_F^2);
 unsigned &glbSwp,          // OUT, number of sweeps at the outermost level;
 unsigned long long &glb_s, // OUT, number of rotations;
 unsigned long long &glb_b, // OUT, number of ``big'' rotations;
 double *const timing,      // OUT, optional, in seconds, double[4] ==
 // WALL, SETUP & HOST ==> GPUs, COMPUTATION, CLEANUP & GPUs ==> HOST;
 const cudaStream_t s
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

  stopwatch_reset(timers[3]);

  size_t lddF = static_cast<size_t>(nrowF);
  double *const dF = allocDeviceMtx<double>(lddF, static_cast<size_t>(nrowF), static_cast<size_t>(ncol), true, s);

  size_t lddG = static_cast<size_t>(nrowG);
  double *const dG = allocDeviceMtx<double>(lddG, static_cast<size_t>(nrowG), static_cast<size_t>(ncol), true, s);

  size_t lddV = static_cast<size_t>(ncol);
  double *const dV = allocDeviceMtx<double>(lddV, static_cast<size_t>(ncol), static_cast<size_t>(ncol), true, s);
  
  double *const dS = allocDeviceVec<double>(static_cast<size_t>(ncol), s);
  double *const dH = allocDeviceVec<double>(static_cast<size_t>(ncol), s);
  double *const dK = allocDeviceVec<double>(static_cast<size_t>(ncol), s);

  unsigned long long *const dC = allocDeviceVec<unsigned long long>((static_cast<size_t>(STRAT1_PAIRS) * C_ELEMS_PER_BLOCK), s);
  unsigned long long *const hC = allocHostVec<unsigned long long>(static_cast<size_t>(STRAT1_PAIRS) * C_ELEMS_PER_BLOCK);

  initSymbols(dF,dG,dV, dS,dH,dK, dC, nrowF,nrowG,ncol,ncol, lddF,lddG,lddV, ((routine & HZ_BO_1) ? 1u : HZ_NSWEEP), s);
  CUDA_CALL(cudaMemcpy2DAsync(dF, lddF * sizeof(double), hF, ldhF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpy2DAsync(dG, lddG * sizeof(double), hG, ldhG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpy2DAsync(dV, lddV * sizeof(double), hV, ldhV * sizeof(double), ncol * sizeof(double), ncol, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaStreamSynchronize(s));
#ifndef USE_MPI
  cuda_prof_start();
#endif /* !USE_MPI */

#ifdef USE_MPI
  const unsigned ifc0 = 0u;
  const unsigned ifc1 = (ncol >> 1u);
  initV(((CVG == 0) || (CVG == 1) || (CVG == 4) || (CVG == 5)), ncol, ifc0, ifc1, s);
#else /* !USE_MPI */
  initV(((CVG == 0) || (CVG == 1) || (CVG == 4) || (CVG == 5)), ncol, s);
#endif /* ?USE_MPI */
  CUDA_CALL(cudaStreamSynchronize(s));

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
      CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
      CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
      CUDA_CALL(cudaStreamSynchronize(s));
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
      CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
      CUDA_CALL(cudaStreamSynchronize(s));
      SYSI_CALL(vn_mtxvis_frame(ctxF, hF, ldhF));
    }
    (void)sprintf(fname, "%c%x_%04u", 'G', routine, ncol);
    SYSI_CALL(vn_mtxvis_start(&ctxG, fname, (VN_MTXVIS_OP_AtA | VN_MTXVIS_FN_Lg | VN_MTXVIS_FF_Bin), nrowG, ncol, 1, 1, 7));
    if (ctxG) {
      SYSI_CALL(vn_mtxvis_frame(ctxG, hG, ldhG));
      CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
      CUDA_CALL(cudaStreamSynchronize(s));
      SYSI_CALL(vn_mtxvis_frame(ctxG, hG, ldhG));
    }
  }
#endif /* ?ANIMATE */
#endif /* ANIMATE */

  timers[1] = stopwatch_lap(timers[3]);
  const int ret = HZ_L2_gpu
    (routine,ncol
#ifdef ANIMATE
     , nrowF,nrowG, hF,ldhF, dF,lddF, hG,ldhG, dG,lddG,
#endif /* ANIMATE */
     , hC,dC, glbSwp,glb_s,glb_b
#ifdef ANIMATE
#if (ANIMATE == 1)
     , ctx
#elif (ANIMATE == 2)
     , ctxF,ctxG
#endif /* ?ANIMATE */
#endif /* ANIMATE */
     , s);
  timers[2] = stopwatch_lap(timers[3]);
#ifndef USE_MPI
  cuda_prof_stop();
#endif /* !USE_MPI */

  CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaMemcpy2DAsync(hV, ldhV * sizeof(double), dV, lddV * sizeof(double), ncol * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaMemcpyAsync(hS, dS, ncol * sizeof(double), cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaMemcpyAsync(hH, dH, ncol * sizeof(double), cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaMemcpyAsync(hK, dK, ncol * sizeof(double), cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaStreamSynchronize(s));

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
#endif /* ?ANIMATE */
#endif /* ANIMATE */

  CUDA_CALL(cudaFreeHost(hC));
  CUDA_CALL(cudaFree(dC));
  CUDA_CALL(cudaFree(dK));
  CUDA_CALL(cudaFree(dH));
  CUDA_CALL(cudaFree(dS));
  CUDA_CALL(cudaFree(dV));
  CUDA_CALL(cudaFree(dG));
  CUDA_CALL(cudaFree(dF));

  timers[3] = stopwatch_lap(timers[3]);
  timers[0] = stopwatch_lap(timers[0]);

  if (timing)
    for (unsigned i = 0u; i < 4u; ++i)
      timing[i] = timers[i] * TS2S;

  return ret;
}
