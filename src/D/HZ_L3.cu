#include "HZ_L3.hpp"

#include "cuda_memory_helper.hpp"
#include "HZ_L.hpp"
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
 double *const hF,          // INOUT, ldhF x n_gpu host array in Fortran order;
 const size_t ldhF,         // IN, leading dimension of F, >= mF;
 double *const hG,          // INOUT, ldhG x n_gpu host array in Fortran order;
 const size_t ldhG,         // IN, leading dimension of G, >= mG;
 double *const hV,          // OUT, ldhV x n_gpu host array in Fortran order;
 const size_t ldhV,         // IN, leading dimension of V, >= n;
 double *const hS,          // OUT, the generalized singular values, optionally sorted in descending order;
 double *const hH,          // ||F_i||_2/sqrt(||F_i||_2^2 + ||G_i||_2^2);
 double *const hK,          // ||G_i||_2/sqrt(||F_i||_2^2 + ||G_i||_2^2);
 unsigned &glbSwp,          // OUT, number of sweeps at the outermost level;
 unsigned long long &glb_s, // OUT, number of rotations;
 unsigned long long &glb_b, // OUT, number of ``big'' rotations;
 double &timing             // OUT, in seconds;
) throw()
{
  if (routine >= 16u)
    return -1;

  if (gpu >= gpus)
    return -2;
  if (!gpus)
    return -3;

  if (!mF)
    return -4;
  if (!mG)
    return -5;
  if (!n)
    return -6;
  if (!n_gpu)
    return -7;
  if (!n_col)
    return -8;

  if (!hF)
    return -9;
  if (ldhF < mF)
    return -10;

  if (!hG)
    return -11;
  if (ldhG < mG)
    return -12;

  if (!hV)
    return -13;
  if (ldhV < n)
    return -14;

  if (!hS)
    return -15;
  if (!hH)
    return -16;
  if (!hK)
    return -17;

  size_t lddF = mF;
  double *const dF = allocDeviceMtx<double>(lddF, mF, n_gpu, true);
  if (lddF != ldhF) {
    DIE("lddF != ldhF");
  }

  size_t lddG = mG;
  double *const dG = allocDeviceMtx<double>(lddG, mG, n_gpu, true);
  if (lddG != ldhG) {
    DIE("lddG != ldhG");
  }

  size_t lddV = n;
  double *const dV = allocDeviceMtx<double>(lddV, n, n_gpu, true);
  if (lddV != ldhV) {
    DIE("lddV != ldhV");
  }

  double *const dS = allocDeviceVec<double>(n_gpu);
  double *const dH = allocDeviceVec<double>(n_gpu);
  double *const dK = allocDeviceVec<double>(n_gpu);

  CUDA_CALL(cudaDeviceSynchronize());
  if (MPI_Barrier(MPI_COMM_WORLD)) {
    DIE("MPI_Barrier(init)");
  }
  long long all_tim = 0ll, swp_tim = 0ll;
  stopwatch_reset(all_tim);
  glb_s = 0ull;
  glb_b = 0ull;
  glbSwp = 0u;
  timing = 0.0;
  unsigned alg = (routine | 1u);

  CUDA_CALL(cudaMemset2DAsync(dV, lddV * sizeof(double), 0, n * sizeof(double), n_gpu));
  CUDA_CALL(cudaMemsetAsync(dH, 0, n_gpu * sizeof(double)));
  CUDA_CALL(cudaMemsetAsync(dK, 0, n_gpu * sizeof(double)));
  CUDA_CALL(cudaDeviceSynchronize());

  stopwatch_reset(swp_tim);

  while (glbSwp < HZ_NSWEEP) {
    unsigned swp_swp = 0u;
    unsigned long long swp_rot[2u] = { 0ull, 0ull };
    for (unsigned stp = 0u; stp < STRAT2_STEPS; ++stp) {
      CUDA_CALL(cudaMemcpy2DAsync(dF, lddF * sizeof(double), hF, ldhF * sizeof(double), mF * sizeof(double), n_gpu, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy2DAsync(dG, lddG * sizeof(double), hG, ldhG * sizeof(double), mG * sizeof(double), n_gpu, cudaMemcpyHostToDevice));
      if (stp || glbSwp) {
        CUDA_CALL(cudaMemcpy2DAsync(dV, lddV * sizeof(double), hV, ldhV * sizeof(double), n * sizeof(double), n_gpu, cudaMemcpyHostToDevice));
      }
      CUDA_CALL(cudaDeviceSynchronize());

      int sp = static_cast<int>(strat2[stp][gpu][1u][0u]);
      const int tp = (sp ? ((sp < 0) ? 0 : 3) : -1);
      sp = abs(sp) - 1;

      int sq = static_cast<int>(strat2[stp][gpu][1u][1u]);
      const int tq = (sq ? ((sq < 0) ? 0 : 3) : -1);
      sq = abs(sq) - 1;

      MPI_Request r[12u] =
        { MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
          MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
          MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
          MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL };

      if (MPI_Irecv(hF, (ldhF * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, (r + 0u))) {
        DIE("MPI_Irecv(F)p");
      }
      if (MPI_Irecv(hG, (ldhG * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, (r + 1u))) {
        DIE("MPI_Irecv(G)p");
      }
      if (MPI_Irecv(hV, (ldhV * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, (r + 2u))) {
        DIE("MPI_Irecv(V)p");
      }

      if (MPI_Irecv((hF + ldhF * n_col), (ldhF * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, (r + 3u))) {
        DIE("MPI_Irecv(F)q");
      }
      if (MPI_Irecv((hG + ldhG * n_col), (ldhG * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 5, MPI_COMM_WORLD, (r + 4u))) {
        DIE("MPI_Irecv(G)q");
      }
      if (MPI_Irecv((hV + ldhV * n_col), (ldhV * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 6, MPI_COMM_WORLD, (r + 5u))) {
        DIE("MPI_Irecv(V)q");
      }

      const unsigned p = static_cast<unsigned>(strat2[stp][gpu][0u][0u]);
      const unsigned q = static_cast<unsigned>(strat2[stp][gpu][0u][1u]);

      const size_t ifc0 = p * n_col;
      const size_t ifc1 = q * n_col;

      unsigned swp2 = 0u;
      unsigned long long rot2s = 0ull, rot2b = 0ull;
      const int ret = HZ_L2_gpu(alg, mF,mG,n_gpu, ifc0,ifc1, dF,lddF, dG,lddG, dV,lddV, hS,dS,dH,dK, swp2,rot2s,rot2b);
      if (ret) {
        (void)snprintf(err_msg, err_msg_size, "HZ_L2_gpu @GPU(%u) SWP(%u) STP(%u): %d", gpu, glbSwp, stp, ret);
        DIE(err_msg);
      }
      if (swp2 > swp_swp)
        swp_swp = swp2;
      swp_rot[0u] += rot2s;
      swp_rot[1u] += rot2b;
   
      if (MPI_Isend(dF, (lddF * n_col), MPI_DOUBLE, sp, (1 + tp), MPI_COMM_WORLD, (r + 6u))) {
        DIE("MPI_Isend(F)p");
      }
      if (MPI_Isend(dG, (lddG * n_col), MPI_DOUBLE, sp, (2 + tp), MPI_COMM_WORLD, (r + 7u))) {
        DIE("MPI_Isend(G)p");
      }
      if (MPI_Isend(dV, (lddV * n_col), MPI_DOUBLE, sp, (3 + tp), MPI_COMM_WORLD, (r + 8u))) {
        DIE("MPI_Isend(V)p");
      }

      if (MPI_Isend((dF + lddF * n_col), (lddF * n_col), MPI_DOUBLE, sq, (1 + tq), MPI_COMM_WORLD, (r + 9u))) {
        DIE("MPI_Isend(F)q");
      }
      if (MPI_Isend((dG + lddG * n_col), (lddG * n_col), MPI_DOUBLE, sq, (2 + tq), MPI_COMM_WORLD, (r + 10u))) {
        DIE("MPI_Isend(G)q");
      }
      if (MPI_Isend((dV + lddV * n_col), (lddV * n_col), MPI_DOUBLE, sq, (3 + tq), MPI_COMM_WORLD, (r + 11u))) {
        DIE("MPI_Isend(V)q");
      }

      if (MPI_Waitall(12, r, MPI_STATUSES_IGNORE)) {
        DIE("MPI_Waitall");
      }
      CUDA_CALL(cudaDeviceSynchronize());
    }
    unsigned max_swp = 0u;
    if (MPI_Allreduce(&swp_swp, &max_swp, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD)) {
      DIE("MPI_Allreduce(max_swp)");
    }
    unsigned long long all_rot[2u] = { 0ull, 0ull };
    if (MPI_Allreduce(swp_rot, all_rot, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD)) {
      DIE("MPI_Allreduce(all_rot)");
    }
    glb_s += all_rot[0u];
    glb_b += all_rot[1u];
    ++glbSwp;

    if (!gpu) {
      (void)fprintf(stdout, "MAX2SWP(%2u), ROT_S(%10llu), ROT_B(%10llu), TIME(%#12.6f s)\n", max_swp, all_rot[0u], all_rot[1u], (stopwatch_lap(swp_tim) * TS2S));
      (void)fflush(stdout);
    }
    if (!all_rot[1u])
      break;
  }

  // CUDA_CALL(cudaMemcpy2DAsync(hF, ldhF * sizeof(double), dF, lddF * sizeof(double), mF * sizeof(double), n_gpu, cudaMemcpyDeviceToHost));
  // CUDA_CALL(cudaMemcpy2DAsync(hG, ldhG * sizeof(double), dG, lddG * sizeof(double), mG * sizeof(double), n_gpu, cudaMemcpyDeviceToHost));
  // CUDA_CALL(cudaMemcpy2DAsync(hV, ldhV * sizeof(double), dV, lddV * sizeof(double), n * sizeof(double), n_gpu, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpyAsync(hS, dS, n_gpu * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpyAsync(hH, dH, n_gpu * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpyAsync(hK, dK, n_gpu * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaDeviceSynchronize());

  if (MPI_Barrier(MPI_COMM_WORLD)) {
    DIE("MPI_Barrier(fini)");
  }
  timing = (stopwatch_lap(all_tim) * TS2S);

  CUDA_CALL(cudaFree(static_cast<void*>(dK)));
  CUDA_CALL(cudaFree(static_cast<void*>(dH)));
  CUDA_CALL(cudaFree(static_cast<void*>(dS)));
  CUDA_CALL(cudaFree(static_cast<void*>(dV)));
  CUDA_CALL(cudaFree(static_cast<void*>(dG)));
  CUDA_CALL(cudaFree(static_cast<void*>(dF)));
  CUDA_CALL(cudaDeviceSynchronize());

  return 0;
}
