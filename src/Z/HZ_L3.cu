#include "HZ_L3.hpp"

#include "HZ_L.hpp"
#include "HZ_L2.hpp"
#include "cuda_memory_helper.hpp"

int HZ_L3
(const unsigned routine,    // IN, routine ID, <= 15, (BbN_)_2,
 // bits B, b: block-oriented (else, full-block), level 1 and 2, N: no sort;
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
 double &timing             // OUT, in seconds;
) throw()
{
  if (routine >= 16)
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

  if (!hFD)
    return -9;
  if (!hFJ)
    return -10;
  if (ldhF < mF)
    return -11;

  if (!hGD)
    return -12;
  if (!hGJ)
    return -13;
  if (ldhG < mG)
    return -14;

  if (!hVD)
    return -15;
  if (!hVJ)
    return -16;
  if (ldhV < n)
    return -17;

  if (!hS)
    return -18;
  if (!hH)
    return -19;
  if (!hK)
    return -20;

  size_t lddF = mF;
  cuD *const dFD = allocDeviceMtx<cuD>(lddF, mF, n_gpu, true);
  cuJ *const dFJ = allocDeviceMtx<cuJ>(lddF, mF, n_gpu, true);
  if (lddF != ldhF) {
    DIE("lddF != ldhF");
  }
  
  size_t lddG = mG;
  cuD *const dGD = allocDeviceMtx<cuD>(lddG, mG, n_gpu, true);
  cuJ *const dGJ = allocDeviceMtx<cuJ>(lddG, mG, n_gpu, true);
  if (lddG != ldhG) {
    DIE("lddG != ldhG");
  }

  size_t lddV = n;
  cuD *const dVD = allocDeviceMtx<cuD>(lddV, n, n_gpu, true);
  cuJ *const dVJ = allocDeviceMtx<cuJ>(lddV, n, n_gpu, true);
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

  CUDA_CALL(cudaMemset2DAsync(dVD, lddV * sizeof(cuD), 0, n * sizeof(cuD), n_gpu));
  CUDA_CALL(cudaMemset2DAsync(dVJ, lddV * sizeof(cuJ), 0, n * sizeof(cuJ), n_gpu));
  CUDA_CALL(cudaMemsetAsync(dH, 0, n_gpu * sizeof(double)));
  CUDA_CALL(cudaMemsetAsync(dK, 0, n_gpu * sizeof(double)));
  CUDA_CALL(cudaDeviceSynchronize());

  stopwatch_reset(swp_tim);

  while (glbSwp < HZ_NSWEEP) {
    unsigned swp_swp = 0u;
    unsigned long long swp_rot[2u] = { 0ull, 0ull };
    for (unsigned stp = 0u; stp < STRAT2_STEPS; ++stp) {
      CUDA_CALL(cudaMemcpy2DAsync(dFD, lddF * sizeof(cuD), hFD, ldhF * sizeof(double), mF * sizeof(cuD), n_gpu, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy2DAsync(dFJ, lddF * sizeof(cuJ), hFJ, ldhF * sizeof(double), mF * sizeof(cuJ), n_gpu, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy2DAsync(dGD, lddG * sizeof(cuD), hGD, ldhG * sizeof(double), mG * sizeof(cuD), n_gpu, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy2DAsync(dGJ, lddG * sizeof(cuJ), hGJ, ldhG * sizeof(double), mG * sizeof(cuJ), n_gpu, cudaMemcpyHostToDevice));
      if (stp || glbSwp) {
        CUDA_CALL(cudaMemcpy2DAsync(dVD, lddG * sizeof(cuD), hVD, ldhV * sizeof(double), n * sizeof(cuD), n_gpu, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy2DAsync(dVJ, lddG * sizeof(cuJ), hVJ, ldhV * sizeof(double), n * sizeof(cuJ), n_gpu, cudaMemcpyHostToDevice));
      }
      CUDA_CALL(cudaDeviceSynchronize());

      int sp = static_cast<int>(strat2[stp][gpu][1u][0u]);
      const int tp = (sp ? ((sp < 0) ? 0 : 6) : -1);
      sp = abs(sp) - 1;

      int sq = static_cast<int>(strat2[stp][gpu][1u][1u]);
      const int tq = (sq ? ((sq < 0) ? 0 : 6) : -1);
      sq = abs(sq) - 1;

      MPI_Request r[24u] =
        { MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
          MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
          MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
          MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL };

      if (MPI_Irecv(hFD, (ldhF * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, (r + 0u))) {
        DIE("MPI_Irecv(FD)p");
      }
      if (MPI_Irecv(hFJ, (ldhF * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, (r + 1u))) {
        DIE("MPI_Irecv(FJ)p");
      }
      if (MPI_Irecv(hGD, (ldhG * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, (r + 2u))) {
        DIE("MPI_Irecv(GD)p");
      }
      if (MPI_Irecv(hGJ, (ldhG * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, (r + 3u))) {
        DIE("MPI_Irecv(GJ)p");
      }
      if (MPI_Irecv(hVD, (ldhV * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 5, MPI_COMM_WORLD, (r + 4u))) {
        DIE("MPI_Irecv(VD)p");
      }
      if (MPI_Irecv(hVJ, (ldhV * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 6, MPI_COMM_WORLD, (r + 5u))) {
        DIE("MPI_Irecv(VJ)p");
      }

      if (MPI_Irecv((hFD + ldhF * n_col), (ldhF * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 7, MPI_COMM_WORLD, (r + 6u))) {
        DIE("MPI_Irecv(FD)q");
      }
      if (MPI_Irecv((hFJ + ldhF * n_col), (ldhF * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 8, MPI_COMM_WORLD, (r + 7u))) {
        DIE("MPI_Irecv(FJ)q");
      }
      if (MPI_Irecv((hGD + ldhG * n_col), (ldhG * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 9, MPI_COMM_WORLD, (r + 8u))) {
        DIE("MPI_Irecv(GD)q");
      }
      if (MPI_Irecv((hGJ + ldhG * n_col), (ldhG * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, (r + 9u))) {
        DIE("MPI_Irecv(GJ)q");
      }
      if (MPI_Irecv((hVD + ldhV * n_col), (ldhV * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 11, MPI_COMM_WORLD, (r + 10u))) {
        DIE("MPI_Irecv(VD)q");
      }
      if (MPI_Irecv((hVJ + ldhV * n_col), (ldhV * n_col), MPI_DOUBLE, MPI_ANY_SOURCE, 12, MPI_COMM_WORLD, (r + 11u))) {
        DIE("MPI_Irecv(VJ)q");
      }

      const unsigned p = static_cast<unsigned>(strat2[stp][gpu][0u][0u]);
      const unsigned q = static_cast<unsigned>(strat2[stp][gpu][0u][1u]);

      const size_t ifc0 = p * n_col;
      const size_t ifc1 = q * n_col;

      unsigned swp2 = 0u;
      unsigned long long rot2s = 0ull, rot2b = 0ull;
      const int ret = HZ_L2_gpu(alg, mF,mG,n_gpu, ifc0,ifc1, dFD,dFJ,lddF, dGD,dGJ,lddG, dVD,dVJ,lddV, hS,dS,dH,dK, swp2,rot2s,rot2b);
      if (ret) {
        (void)snprintf(err_msg, err_msg_size, "HZ_L2_gpu @GPU(%u) SWP(%u) STP(%u): %d", gpu, glbSwp, stp, ret);
        DIE(err_msg);
      }
      if (swp2 > swp_swp)
        swp_swp = swp2;
      swp_rot[0u] += rot2s;
      swp_rot[1u] += rot2b;
   
      if (MPI_Isend(dFD, (lddF * n_col), MPI_DOUBLE, sp, (1 + tp), MPI_COMM_WORLD, (r + 12u))) {
        DIE("MPI_Isend(FD)p");
      }
      if (MPI_Isend(dFJ, (lddF * n_col), MPI_DOUBLE, sp, (2 + tp), MPI_COMM_WORLD, (r + 13u))) {
        DIE("MPI_Isend(FJ)p");
      }
      if (MPI_Isend(dGD, (lddG * n_col), MPI_DOUBLE, sp, (3 + tp), MPI_COMM_WORLD, (r + 14u))) {
        DIE("MPI_Isend(GD)p");
      }
      if (MPI_Isend(dGJ, (lddG * n_col), MPI_DOUBLE, sp, (4 + tp), MPI_COMM_WORLD, (r + 15u))) {
        DIE("MPI_Isend(GJ)p");
      }
      if (MPI_Isend(dVD, (lddV * n_col), MPI_DOUBLE, sp, (5 + tp), MPI_COMM_WORLD, (r + 16u))) {
        DIE("MPI_Isend(VD)p");
      }
      if (MPI_Isend(dVJ, (lddV * n_col), MPI_DOUBLE, sp, (6 + tp), MPI_COMM_WORLD, (r + 17u))) {
        DIE("MPI_Isend(VJ)p");
      }

      if (MPI_Isend((dFD + lddF * n_col), (lddF * n_col), MPI_DOUBLE, sq, (1 + tq), MPI_COMM_WORLD, (r + 18u))) {
        DIE("MPI_Isend(FD)q");
      }
      if (MPI_Isend((dFJ + lddF * n_col), (lddF * n_col), MPI_DOUBLE, sq, (2 + tq), MPI_COMM_WORLD, (r + 19u))) {
        DIE("MPI_Isend(FJ)q");
      }
      if (MPI_Isend((dGD + lddG * n_col), (lddG * n_col), MPI_DOUBLE, sq, (3 + tq), MPI_COMM_WORLD, (r + 20u))) {
        DIE("MPI_Isend(GD)q");
      }
      if (MPI_Isend((dGJ + lddG * n_col), (lddG * n_col), MPI_DOUBLE, sq, (4 + tq), MPI_COMM_WORLD, (r + 21u))) {
        DIE("MPI_Isend(GJ)q");
      }
      if (MPI_Isend((dVD + lddV * n_col), (lddV * n_col), MPI_DOUBLE, sq, (5 + tq), MPI_COMM_WORLD, (r + 22u))) {
        DIE("MPI_Isend(VD)q");
      }
      if (MPI_Isend((dVJ + lddV * n_col), (lddV * n_col), MPI_DOUBLE, sq, (6 + tq), MPI_COMM_WORLD, (r + 23u))) {
        DIE("MPI_Isend(VJ)q");
      }

      if (MPI_Waitall(24, r, MPI_STATUSES_IGNORE)) {
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

  // CUDA_CALL(cudaMemcpy2DAsync(hFD, ldhF * sizeof(double), dFD, lddF * sizeof(cuD), mF * sizeof(cuD), n_gpu, cudaMemcpyDeviceToHost));
  // CUDA_CALL(cudaMemcpy2DAsync(hFJ, ldhF * sizeof(double), dFJ, lddF * sizeof(cuJ), mF * sizeof(cuJ), n_gpu, cudaMemcpyDeviceToHost));
  // CUDA_CALL(cudaMemcpy2DAsync(hGD, ldhG * sizeof(double), dGD, lddG * sizeof(cuD), mG * sizeof(cuD), n_gpu, cudaMemcpyDeviceToHost));
  // CUDA_CALL(cudaMemcpy2DAsync(hGJ, ldhG * sizeof(double), dGJ, lddG * sizeof(cuJ), mG * sizeof(cuJ), n_gpu, cudaMemcpyDeviceToHost));
  // CUDA_CALL(cudaMemcpy2DAsync(hVD, ldhV * sizeof(double), dVD, lddV * sizeof(cuD), n * sizeof(cuD), n_gpu, cudaMemcpyDeviceToHost));
  // CUDA_CALL(cudaMemcpy2DAsync(hVJ, ldhV * sizeof(double), dVJ, lddV * sizeof(cuJ), n * sizeof(cuJ), n_gpu, cudaMemcpyDeviceToHost));
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
  CUDA_CALL(cudaFree(static_cast<void*>(dVJ)));
  CUDA_CALL(cudaFree(static_cast<void*>(dVD)));
  CUDA_CALL(cudaFree(static_cast<void*>(dGJ)));
  CUDA_CALL(cudaFree(static_cast<void*>(dGD)));
  CUDA_CALL(cudaFree(static_cast<void*>(dFJ)));
  CUDA_CALL(cudaFree(static_cast<void*>(dFD)));
  CUDA_CALL(cudaDeviceSynchronize());

  return 0;
}
