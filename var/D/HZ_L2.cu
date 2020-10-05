#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L2.hpp"

#include "device_code.hpp"
#include "cuda_memory_helper.hpp"
#include "my_utils.hpp"

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
 cublasHandle_t handle      // IN, CUBLAS handle
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

  cudaStream_t s = 0;
  CUBLAS_CALL(cublasGetStream(handle, &s));

  size_t lddF = static_cast<size_t>(nrowF);
  double *const dF[2u] =
    { allocDeviceMtx<double>(lddF, static_cast<size_t>(nrowF), static_cast<size_t>(ncol), true, s),
      allocDeviceMtx<double>(lddF, static_cast<size_t>(nrowF), static_cast<size_t>(ncol), true, s)
    };

  size_t lddG = static_cast<size_t>(nrowG);
  double *const dG[2u] =
    { allocDeviceMtx<double>(lddG, static_cast<size_t>(nrowG), static_cast<size_t>(ncol), true, s),
      allocDeviceMtx<double>(lddG, static_cast<size_t>(nrowG), static_cast<size_t>(ncol), true, s)
    };

  const unsigned nrowV = ncol;
  size_t lddV = static_cast<size_t>(nrowV);
  double *const dV[2u] =
    { allocDeviceMtx<double>(lddV, static_cast<size_t>(nrowV), static_cast<size_t>(ncol), true, s),
      allocDeviceMtx<double>(lddV, static_cast<size_t>(nrowV), static_cast<size_t>(ncol), true, s)
    };

  const unsigned nrowW = (HZ_L1_NCOLB << 1u);
  size_t lddW = static_cast<size_t>(nrowW);
  double *const dW = allocDeviceMtx<double>(lddW, static_cast<size_t>(nrowW), static_cast<size_t>(ncol), true, s);

  double *const dS = allocDeviceVec<double>(static_cast<size_t>(ncol), s);
  double *const dH = allocDeviceVec<double>(static_cast<size_t>(ncol), s);
  double *const dK = allocDeviceVec<double>(static_cast<size_t>(ncol), s);

  // stats count
  const unsigned sc = STRAT1_PAIRS * C_ELEMS_PER_BLOCK;
  // stats len
  const size_t sl = sc * sizeof(unsigned long long);

  unsigned long long *const dC = allocDeviceVec<unsigned long long>(static_cast<size_t>(sc), s);
  unsigned long long *const hC = allocHostVec<unsigned long long>(static_cast<size_t>(sc));

  const size_t bc1 = (static_cast<size_t>(STRAT1_PAIRS) << 1u);
  const size_t bc0 = (bc1 + STRAT1_PAIRS);

  double **const pA = allocHostVec<double*>(bc0);
  double **const pB = allocHostVec<double*>(bc0);
  double **const pC = allocHostVec<double*>(bc0);

  initSymbols(dW,dC, nrowF,nrowG,nrowV,nrowW, lddF,lddG,lddV,lddW, ncol,((routine & HZ_BO_1) ? 1u : HZ_NSWEEP), s);

  const size_t lddFb = (HZ_L1_NCOLB * lddF);
  const size_t lddGb = (HZ_L1_NCOLB * lddG);
  const size_t lddVb = (HZ_L1_NCOLB * lddV);
  const size_t lddWb = (HZ_L1_NCOLB * lddW);

  const double one = 1.0;
  const double zero = 0.0;

  unsigned s0 = 0u;
  unsigned s1 = 1u;

  CUDA_CALL(cudaMemcpy2DAsync(dF[s0], lddF * sizeof(double), hF, ldhF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpy2DAsync(dG[s0], lddG * sizeof(double), hG, ldhG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaMemcpy2DAsync(dV[s0], lddV * sizeof(double), hV, ldhV * sizeof(double), nrowV * sizeof(double), ncol, cudaMemcpyHostToDevice, s));
  CUDA_CALL(cudaStreamSynchronize(s));
  cuda_prof_start();

  initV(dF[s0], dG[s0], dV[s0], ncol, s);
  CUDA_CALL(cudaStreamSynchronize(s));

  const unsigned swp = HZ_NSWEEP;
  timers[1] = stopwatch_lap(timers[3]);
  glb_s = 0ull;
  glb_b = 0ull;

  long long swp_tim = 0ll;
  stopwatch_reset(swp_tim);

  unsigned blk_swp = 0u;
  while (blk_swp < swp) {
    CUDA_CALL(cudaMemsetAsync(dC, 0, sl, s));
    CUDA_CALL(cudaStreamSynchronize(s));
    for (unsigned blk_stp = 0u; blk_stp < STRAT1_STEPS; ++blk_stp) {
      if (blk_stp) {
        CUDA_CALL(cudaStreamSynchronize(s));
      }
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif /* _OPENMP */
      for (unsigned k = 0u; k < STRAT1_PAIRS; ++k) {
        const unsigned p = static_cast<unsigned>(strat1[blk_stp][k][0u]);
        const unsigned q = static_cast<unsigned>(strat1[blk_stp][k][1u]);
        const unsigned i = (k << 1u);
        unsigned j = (k * 3u);
        // (p,p) @ F
        pB[j] = pA[j] = (dF[s0] + p * lddFb);
        pC[j] = (dF[s1] + i * lddFb);
        ++j;
        // (q,p) @ F
        pA[j] = (dF[s0] + q * lddFb);
        pB[j] = (dF[s0] + p * lddFb);
        pC[j] = (dF[s1] + i * lddFb + HZ_L1_NCOLB);
        ++j;
        // (q,q) @ F
        pB[j] = pA[j] = (dF[s0] + q * lddFb);
        pC[j] = (dF[s1] + (i + 1u) * lddFb + HZ_L1_NCOLB);
      }
      CUBLAS_CALL(cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(HZ_L1_NCOLB), static_cast<int>(HZ_L1_NCOLB), static_cast<int>(nrowF), &one, pA, static_cast<int>(lddF), pB, static_cast<int>(lddF), &zero, pC, static_cast<int>(lddF), static_cast<int>(bc0)));
      CUDA_CALL(cudaStreamSynchronize(s));
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif /* _OPENMP */
      for (unsigned k = 0u; k < STRAT1_PAIRS; ++k) {
        const unsigned p = static_cast<unsigned>(strat1[blk_stp][k][0u]);
        const unsigned q = static_cast<unsigned>(strat1[blk_stp][k][1u]);
        const unsigned i = (k << 1u);
        unsigned j = (k * 3u);
        // (p,p) @ G
        pB[j] = pA[j] = (dG[s0] + p * lddGb);
        pC[j] = (dG[s1] + i * lddGb);
        ++j;
        // (q,p) @ G
        pA[j] = (dG[s0] + q * lddGb);
        pB[j] = (dG[s0] + p * lddGb);
        pC[j] = (dG[s1] + i * lddGb + HZ_L1_NCOLB);
        ++j;
        // (q,q) @ G
        pB[j] = pA[j] = (dG[s0] + q * lddGb);
        pC[j] = (dG[s1] + (i + 1u) * lddGb + HZ_L1_NCOLB);
      }
      CUBLAS_CALL(cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(HZ_L1_NCOLB), static_cast<int>(HZ_L1_NCOLB), static_cast<int>(nrowG), &one, pA, static_cast<int>(lddG), pB, static_cast<int>(lddG), &zero, pC, static_cast<int>(lddG), static_cast<int>(bc0)));
      CUDA_CALL(cudaStreamSynchronize(s));
      HZ_L1_sv(dF[s1], dG[s1], s);
      CUDA_CALL(cudaStreamSynchronize(s));
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif /* _OPENMP */
      for (unsigned k = 0u; k < STRAT1_PAIRS; ++k) {
        const unsigned p = static_cast<unsigned>(strat1[blk_stp][k][0u]);
        const unsigned q = static_cast<unsigned>(strat1[blk_stp][k][1u]);
        unsigned i = (k << 1u);
        // p @ F
        pA[i] = (dF[s0] + p * lddFb);
        pB[i] = (dW + i * lddWb);
        pC[i] = (dF[s1] + p * lddFb);
        ++i;
        // q @ F
        pA[i] = (dF[s0] + p * lddFb);
        pB[i] = (dW + i * lddWb);
        pC[i] = (dF[s1] + q * lddFb);
      }
      CUBLAS_CALL(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(nrowF), static_cast<int>(HZ_L1_NCOLB), static_cast<int>(HZ_L1_NCOLB), &one, pA, static_cast<int>(lddF), pB, static_cast<int>(lddW), &zero, pC, static_cast<int>(lddF), static_cast<int>(bc1)));
      CUDA_CALL(cudaStreamSynchronize(s));
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif /* _OPENMP */
      for (unsigned k = 0u; k < STRAT1_PAIRS; ++k) {
        const unsigned p = static_cast<unsigned>(strat1[blk_stp][k][0u]);
        const unsigned q = static_cast<unsigned>(strat1[blk_stp][k][1u]);
        unsigned i = (k << 1u);
        // p @ G
        pA[i] = (dG[s0] + p * lddGb);
        pB[i] = (dW + i * lddWb);
        pC[i] = (dG[s1] + p * lddGb);
        ++i;
        // q @ G
        pA[i] = (dG[s0] + p * lddGb);
        pB[i] = (dW + i * lddWb);
        pC[i] = (dG[s1] + q * lddGb);
      }
      CUBLAS_CALL(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(nrowG), static_cast<int>(HZ_L1_NCOLB), static_cast<int>(HZ_L1_NCOLB), &one, pA, static_cast<int>(lddG), pB, static_cast<int>(lddW), &zero, pC, static_cast<int>(lddG), static_cast<int>(bc1)));
      CUDA_CALL(cudaStreamSynchronize(s));
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif /* _OPENMP */
      for (unsigned k = 0u; k < STRAT1_PAIRS; ++k) {
        const unsigned p = static_cast<unsigned>(strat1[blk_stp][k][0u]);
        const unsigned q = static_cast<unsigned>(strat1[blk_stp][k][1u]);
        unsigned i = (k << 1u);
        // p @ V
        pA[i] = (dV[s0] + p * lddVb);
        pB[i] = (dW + i * lddWb);
        pC[i] = (dV[s1] + p * lddVb);
        ++i;
        // q @ V
        pA[i] = (dV[s0] + p * lddVb);
        pB[i] = (dW + i * lddWb);
        pC[i] = (dV[s1] + q * lddVb);
      }
      CUBLAS_CALL(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(nrowV), static_cast<int>(HZ_L1_NCOLB), static_cast<int>(HZ_L1_NCOLB), &one, pA, static_cast<int>(lddV), pB, static_cast<int>(lddW), &zero, pC, static_cast<int>(lddV), static_cast<int>(bc1)));
      CUDA_CALL(cudaStreamSynchronize(s));
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif /* _OPENMP */
      for (unsigned k = 0u; k < STRAT1_PAIRS; ++k) {
        const unsigned p = static_cast<unsigned>(strat1[blk_stp][k][0u]);
        const unsigned q = static_cast<unsigned>(strat1[blk_stp][k][1u]);
        unsigned i = (k << 1u);
        // p @ F
        pA[i] = (dF[s0] + q * lddFb);
        pB[i] = (dW + i * lddWb + HZ_L1_NCOLB);
        pC[i] = (dF[s1] + p * lddFb);
        ++i;
        // q @ F
        pA[i] = (dF[s0] + q * lddFb);
        pB[i] = (dW + i * lddWb + HZ_L1_NCOLB);
        pC[i] = (dF[s1] + q * lddFb);
      }
      CUBLAS_CALL(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(nrowF), static_cast<int>(HZ_L1_NCOLB), static_cast<int>(HZ_L1_NCOLB), &one, pA, static_cast<int>(lddF), pB, static_cast<int>(lddW), &one, pC, static_cast<int>(lddF), static_cast<int>(bc1)));
      CUDA_CALL(cudaStreamSynchronize(s));
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif /* _OPENMP */
      for (unsigned k = 0u; k < STRAT1_PAIRS; ++k) {
        const unsigned p = static_cast<unsigned>(strat1[blk_stp][k][0u]);
        const unsigned q = static_cast<unsigned>(strat1[blk_stp][k][1u]);
        unsigned i = (k << 1u);
        // p @ G
        pA[i] = (dG[s0] + q * lddGb);
        pB[i] = (dW + i * lddWb + HZ_L1_NCOLB);
        pC[i] = (dG[s1] + p * lddGb);
        ++i;
        // q @ G
        pA[i] = (dG[s0] + q * lddGb);
        pB[i] = (dW + i * lddWb + HZ_L1_NCOLB);
        pC[i] = (dG[s1] + q * lddGb);
      }
      CUBLAS_CALL(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(nrowG), static_cast<int>(HZ_L1_NCOLB), static_cast<int>(HZ_L1_NCOLB), &one, pA, static_cast<int>(lddG), pB, static_cast<int>(lddW), &one, pC, static_cast<int>(lddG), static_cast<int>(bc1)));
      CUDA_CALL(cudaStreamSynchronize(s));
#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif /* _OPENMP */
      for (unsigned k = 0u; k < STRAT1_PAIRS; ++k) {
        const unsigned p = static_cast<unsigned>(strat1[blk_stp][k][0u]);
        const unsigned q = static_cast<unsigned>(strat1[blk_stp][k][1u]);
        unsigned i = (k << 1u);
        // p @ V
        pA[i] = (dV[s0] + q * lddVb);
        pB[i] = (dW + i * lddWb + HZ_L1_NCOLB);
        pC[i] = (dV[s1] + p * lddVb);
        ++i;
        // q @ V
        pA[i] = (dV[s0] + q * lddVb);
        pB[i] = (dW + i * lddWb + HZ_L1_NCOLB);
        pC[i] = (dV[s1] + q * lddVb);
      }
      CUBLAS_CALL(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(nrowV), static_cast<int>(HZ_L1_NCOLB), static_cast<int>(HZ_L1_NCOLB), &one, pA, static_cast<int>(lddV), pB, static_cast<int>(lddW), &one, pC, static_cast<int>(lddV), static_cast<int>(bc1)));
      const unsigned s = s0;
      s0 = s1;
      s1 = s;
    }

    CUDA_CALL(cudaStreamSynchronize(s));
    CUDA_CALL(cudaMemcpyAsync(hC, dC, sl, cudaMemcpyDeviceToHost, s));
    CUDA_CALL(cudaStreamSynchronize(s));

    unsigned long long cvg_s = 0ull;
    unsigned long long cvg_b = 0ull;
    for (unsigned i = 0u; i < sc; i += C_ELEMS_PER_BLOCK) {
      cvg_s += hC[i + C_SMALL];
      cvg_b += hC[i + C_BIG];
    }
    glb_s += cvg_s;
    glb_b += cvg_b;

    const double tim_s = stopwatch_lap(swp_tim) * TS2S;
    (void)fprintf(stdout, "BLK_SWP(%2u), ROT_S(%13llu), ROT_B(%13llu), TIME(%#14.6f s)\n", blk_swp, cvg_s, cvg_b, tim_s);
    (void)fflush(stdout);
    if (!cvg_b)
      break;
    ++blk_swp;
    initS(dF[s0], dG[s0], dV[s0], ncol, s);
    CUDA_CALL(cudaStreamSynchronize(s));
  }

  if (blk_swp < swp)
    glbSwp = (blk_swp + 1u);
  else
    glbSwp = blk_swp;
  initS(dF[s0], dG[s0], dV[s0], dS, dH, dK, ncol, s);
  CUDA_CALL(cudaStreamSynchronize(s));

  timers[2] = stopwatch_lap(timers[3]);
  cuda_prof_stop();

  CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF[s0], lddF * sizeof(double), nrowF * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG[s0], lddG * sizeof(double), nrowG * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaMemcpy2DAsync(hV, ldhV * sizeof(double), dV[s0], lddV * sizeof(double), nrowV * sizeof(double), ncol, cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaMemcpyAsync(hS, dS, ncol * sizeof(double), cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaMemcpyAsync(hH, dH, ncol * sizeof(double), cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaMemcpyAsync(hK, dK, ncol * sizeof(double), cudaMemcpyDeviceToHost, s));
  CUDA_CALL(cudaStreamSynchronize(s));

  CUDA_CALL(cudaFreeHost(pC));
  CUDA_CALL(cudaFreeHost(pB));
  CUDA_CALL(cudaFreeHost(pA));  
  CUDA_CALL(cudaFreeHost(hC));
  CUDA_CALL(cudaFree(dC));
  CUDA_CALL(cudaFree(dK));
  CUDA_CALL(cudaFree(dH));
  CUDA_CALL(cudaFree(dS));
  CUDA_CALL(cudaFree(dW));
  CUDA_CALL(cudaFree(dV[1u]));
  CUDA_CALL(cudaFree(dV[0u]));
  CUDA_CALL(cudaFree(dG[1u]));
  CUDA_CALL(cudaFree(dG[0u]));
  CUDA_CALL(cudaFree(dF[1u]));
  CUDA_CALL(cudaFree(dF[0u]));

  timers[3] = stopwatch_lap(timers[3]);
  timers[0] = stopwatch_lap(timers[0]);

  if (timing)
    for (unsigned i = 0u; i < 4u; ++i)
      timing[i] = timers[i] * TS2S;

  return 0;
}
