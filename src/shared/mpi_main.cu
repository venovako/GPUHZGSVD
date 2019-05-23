// mpi_main.cu: test driver.

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L3.hpp"

#include "my_utils.hpp"
#include "cuda_memory_helper.hpp"

int main(int argc, char *argv[])
{
  if (9 != argc) {
    DIE("Arguments: SNP0 SNP1 SNP2 ALG MF MG N FN");
  }

  const char *const ca_exe = argv[0];
  const char *const ca_snp0 = argv[1];
  const char *const ca_snp1 = argv[2];
  const char *const ca_snp2 = argv[3];
  const char *const ca_alg = argv[4];
  const char *const ca_mF = argv[5];
  const char *const ca_mG = argv[6];
  const char *const ca_n = argv[7];
  const char *const ca_fn = argv[8];

  const unsigned routine = static_cast<unsigned>(atou(ca_alg));
  if (routine && (routine != 8u)) {
    DIE("ALG \\notin { 0, 8 }");
  }

  const unsigned nrowF = static_cast<unsigned>(atou(ca_mF));
  const unsigned nrowG = static_cast<unsigned>(atou(ca_mG));
  const unsigned ncol = static_cast<unsigned>(atou(ca_n));
  if (ncol > nrowF) {
    DIE("N > MF");
  }
  if (ncol > nrowG) {
    DIE("N > MG");
  }

  const unsigned snp0 = static_cast<unsigned>(atou(ca_snp0));
  if ((snp0 != STRAT_CYCWOR) && (snp0 != STRAT_MMSTEP)) {
    DIE("SNP0 \\notin { 2, 4 }");
  }
  const unsigned snp1 = static_cast<unsigned>(atou(ca_snp1));
  if ((snp1 != STRAT_CYCWOR) && (snp1 != STRAT_MMSTEP)) {
    DIE("SNP1 \\notin { 2, 4 }");
  }
  const unsigned snp2 = static_cast<unsigned>(atou(ca_snp2));
  if ((snp2 != (STRAT_CYCWOR + 1u)) && (snp2 != (STRAT_MMSTEP + 1u))) {
    DIE("SNP2 \\notin { 3, 5 }");
  }
  if (!*ca_fn) {
    DIE("invalid argument FN");
  }

  if (init_MPI(&argc, &argv)) {
    (void)fprintf(stderr, "%s[%d] init_MPI failed\n", ca_exe, mpi_rank);
    return fini_MPI();
  }
  const unsigned gpu = static_cast<unsigned>(mpi_rank);
  if (!mpi_cuda_aware) {
    if (!gpu)
      (void)fprintf(stderr, "MPI is not CUDA aware\n");
    return fini_MPI();
  }
  const unsigned gpus = static_cast<unsigned>(mpi_size);
  if (gpus < 2u) {
    if (!gpu)
      (void)fprintf(stderr, "MPI_COMM_WORLD size (%u) < 2\n", gpus);
    return fini_MPI();
  }
  const unsigned n2 = (gpus << 1u);
  if (ncol < n2) {
    if (!gpu)
      (void)fprintf(stderr, "N(%u) < n2(%u)\n", ncol, n2);
    return fini_MPI();
  }

  const int dev = assign_dev2host();
  if (dev < 0) {
    if (!gpu)
      (void)fprintf(stderr, "assign_dev2host failed (%d)\n", dev);
    return fini_MPI();
  }

  const int dcc = configureGPU(dev);
#ifndef NDEBUG
  (void)fprintf(stdout, "[%u] device(%d) has CC(%d)\n", gpu, dev, dcc);
  (void)fflush(stdout);
#endif // !NDEBUG

  unsigned nrowF_ = 0u, nrowG_ = 0u, ncol_ = 0u;
  border_sizes(gpus, nrowF, nrowG, ncol, nrowF_, nrowG_, ncol_);
  const unsigned ncol_gpu = ncol_ / gpus;

  const unsigned n0 = (HZ_L1_NCOLB << 1u);
  const unsigned n1 = ncol_gpu / HZ_L1_NCOLB;
  if (ncol_gpu % HZ_L1_NCOLB) {
    if (!gpu)
      (void)fprintf(stderr, "ncol_gpu(%u)\n", ncol_gpu);
    return fini_MPI();
  }
  init_strats(snp0, n0, snp1, n1, snp2, n2);

  const unsigned p = strat2[0u][gpu][0u][0u];
  const unsigned q = strat2[0u][gpu][0u][1u];

  const size_t mF = static_cast<size_t>(nrowF);
  const size_t mF_ = static_cast<size_t>(nrowF_);
  const size_t mG = static_cast<size_t>(nrowG);
  const size_t mG_ = static_cast<size_t>(nrowG_);
  const size_t n = static_cast<size_t>(ncol);
  const size_t n_ = static_cast<size_t>(ncol_);
  const size_t n_gpu = static_cast<size_t>(ncol_gpu);
  const size_t n_col = (n_gpu >> 1u);

  unsigned
    ldhF = nrowF_,
    ldhG = nrowG_,
    ldhV = ncol_;

  const size_t p_ = p * n_col;
  const size_t n_p = ((p_ >= n) ? static_cast<size_t>(0u) : (((p_ + n_col) > n) ? (n - p_) : n_col));
  const long opF = static_cast<long>((p_ * mF)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );
  const long opG = static_cast<long>((p_ * mG)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );

  const size_t q_ = q * n_col; 
  const size_t n_q = ((q_ >= n) ? static_cast<size_t>(0u) : (((q_ + n_col) > n) ? (n - q_) : n_col));
  const long oqF = static_cast<long>((q_ * mF)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );
  const long oqG = static_cast<long>((q_ * mG)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );

  char *const buf = static_cast<char*>(calloc(strlen(ca_fn) + 4u, sizeof(char)));
  SYSP_CALL(buf);
  FILE *f = static_cast<FILE*>(NULL);

  size_t ldA = static_cast<size_t>(0u);

  ldA = static_cast<size_t>(ldhF);
#ifdef USE_COMPLEX
  cuD *const hFD = allocHostMtx<cuD>(ldA, mF_, n_gpu, true);
  SYSP_CALL(hFD);
  cuJ *const hFJ = allocHostMtx<cuJ>(ldA, mF_, n_gpu, true);
  SYSP_CALL(hFJ);
#else // !USE_COMPLEX
  double *const hF = allocHostMtx<double>(ldA, mF_, n_gpu, true);
  SYSP_CALL(hF);
#endif // ?USE_COMPLEX
  ldhF = static_cast<unsigned>(ldA);

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".Y"), "rb"));
#ifdef USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mF, n_p, hFD, ldA, opF, 2l));
  SYSI_CALL(fread_bycol(f, mF, n_p, hFJ, ldA, (opF + 1l), 2l));
  SYSI_CALL(fread_bycol(f, mF, n_q, (hFD + ldA * n_col), ldA, oqF, 2l));
  SYSI_CALL(fread_bycol(f, mF, n_q, (hFJ + ldA * n_col), ldA, (oqF + 1l), 2l));
#else // !USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mF, n_p, hF, ldA, opF));
  SYSI_CALL(fread_bycol(f, mF, n_q, (hF + ldA * n_col), ldA, oqF));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));

  ldA = static_cast<size_t>(ldhG);
#ifdef USE_COMPLEX
  cuD *const hGD = allocHostMtx<cuD>(ldA, mG_, n_gpu, true);
  SYSP_CALL(hGD);
  cuJ *const hGJ = allocHostMtx<cuJ>(ldA, mG_, n_gpu, true);
  SYSP_CALL(hGJ);
#else // !USE_COMPLEX
  double *const hG = allocHostMtx<double>(ldA, mG_, n_gpu, true);
  SYSP_CALL(hG);
#endif // ?USE_COMPLEX
  ldhG = static_cast<unsigned>(ldA);

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".W"), "rb"));
#ifdef USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mG, n_p, hGD, ldA, opG, 2l));
  SYSI_CALL(fread_bycol(f, mG, n_p, hGJ, ldA, (opG + 1l), 2l));
  SYSI_CALL(fread_bycol(f, mG, n_q, (hGD + ldA * n_col), ldA, oqG, 2l));
  SYSI_CALL(fread_bycol(f, mG, n_q, (hGJ + ldA * n_col), ldA, (oqG + 1l), 2l));
#else // !USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mG, n_p, hG, ldA, opG));
  SYSI_CALL(fread_bycol(f, mG, n_q, (hG + ldA * n_col), ldA, oqG));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));

  ldA = static_cast<size_t>(ldhV);
#ifdef USE_COMPLEX
  cuD *const hVD = allocHostMtx<cuD>(ldA, n_gpu, n_gpu, true);
  SYSP_CALL(hVD);
  cuJ *const hVJ = allocHostMtx<cuJ>(ldA, n_gpu, n_gpu, true);
  SYSP_CALL(hVJ);
#else // !USE_COMPLEX
  double *const hV = allocHostMtx<double>(ldA, n_gpu, n_gpu, true);
  SYSP_CALL(hV);
#endif // ?USE_COMPLEX
  ldhV = static_cast<unsigned>(ldA);

  double *const hS = allocHostVec<double>(n_gpu);
  SYSP_CALL(hS);
  double *const hH = allocHostVec<double>(n_gpu);
  SYSP_CALL(hH);
  double *const hK = allocHostVec<double>(n_gpu);
  SYSP_CALL(hK);

  unsigned glbSwp = 0u;
  unsigned long long glb_s = 0ull, glb_b = 0ull;
  double timing[4u] = { -0.0, -0.0, -0.0, -0.0 };

  if (!gpu) {
    SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".YU"), "wb"));
    ldA = (mF * n * sizeof(double))
#ifdef USE_COMPLEX
      * 2u
#endif // USE_COMPLEX
    ;
    SYSI_CALL(fresize(f, ldA));
    SYSI_CALL(fclose(f));

    SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".WV"), "wb"));
    ldA = (mG * n * sizeof(double))
#ifdef USE_COMPLEX
      * 2u
#endif // USE_COMPLEX
    ;
    SYSI_CALL(fresize(f, ldA));
    SYSI_CALL(fclose(f));

    SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".Z"), "wb"));
    ldA = (n * n * sizeof(double))
#ifdef USE_COMPLEX
      * 2u
#endif // USE_COMPLEX
    ;
    SYSI_CALL(fresize(f, ldA));
    SYSI_CALL(fclose(f));

    SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".SS"), "wb"));
    ldA = (n * sizeof(double));
    SYSI_CALL(fresize(f, ldA));
    SYSI_CALL(fclose(f));

    SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".SY"), "wb"));
    ldA = (n * sizeof(double));
    SYSI_CALL(fresize(f, ldA));
    SYSI_CALL(fclose(f));

    SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".SW"), "wb"));
    ldA = (n * sizeof(double));
    SYSI_CALL(fresize(f, ldA));
    SYSI_CALL(fclose(f));
  }
  if (MPI_Barrier(MPI_COMM_WORLD))
    return fini_MPI();

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".YU"), "wb"));
#ifdef USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mF, n_p, hFD, ldhF, opF, 2l));
  SYSI_CALL(fwrite_bycol(f, mF, n_p, hFJ, ldhF, (opF + 1l), 2l));
  SYSI_CALL(fwrite_bycol(f, mF, n_q, (hFD + ldhF * n_col), ldhF, oqF, 2l));
  SYSI_CALL(fwrite_bycol(f, mF, n_q, (hFJ + ldhF * n_col), ldhF, (oqF + 1l), 2l));
#else // !USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mF, n_p, hF, ldhF, opF));
  SYSI_CALL(fwrite_bycol(f, mF, n_q, (hF + ldhF * n_col), ldhF, oqF));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));
  if (MPI_Barrier(MPI_COMM_WORLD))
    return fini_MPI();

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".WV"), "wb"));
#ifdef USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mG, n_p, hGD, ldhG, opG, 2l));
  SYSI_CALL(fwrite_bycol(f, mG, n_p, hGJ, ldhG, (opG + 1l), 2l));
  SYSI_CALL(fwrite_bycol(f, mG, n_q, (hGD + ldhG * n_col), ldhG, oqG, 2l));
  SYSI_CALL(fwrite_bycol(f, mG, n_q, (hGJ + ldhG * n_col), ldhG, (oqG + 1l), 2l));
#else // !USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mG, n_p, hG, ldhG, opG));
  SYSI_CALL(fwrite_bycol(f, mG, n_q, (hG + ldhG * n_col), ldhG, oqG));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));
  if (MPI_Barrier(MPI_COMM_WORLD))
    return fini_MPI();

  const long opV = static_cast<long>((p_ * n)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );
  const long oqV = static_cast<long>((q_ * n)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".Z"), "wb"));
#ifdef USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, n, n_p, hVD, ldhV, opV, 2l));
  SYSI_CALL(fwrite_bycol(f, n, n_p, hVJ, ldhV, (opV + 1l), 2l));
  SYSI_CALL(fwrite_bycol(f, n, n_q, (hVD + ldhV * n_col), ldhV, oqV, 2l));
  SYSI_CALL(fwrite_bycol(f, n, n_q, (hVJ + ldhV * n_col), ldhV, (oqV + 1l), 2l));
#else // !USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, n, n_p, hV, ldhV, opV));
  SYSI_CALL(fwrite_bycol(f, n, n_q, (hV + ldhV * n_col), ldhV, oqV));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));
  if (MPI_Barrier(MPI_COMM_WORLD))
    return fini_MPI();

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".SS"), "wb"));
  SYSI_CALL(fwrite_bycol(f, n_p, 1u, hS, n_gpu, static_cast<long>(p_)));
  SYSI_CALL(fwrite_bycol(f, n_q, 1u, (hS + n_col), n_gpu, static_cast<long>(q_)));
  SYSI_CALL(fclose(f));
  if (MPI_Barrier(MPI_COMM_WORLD))
    return fini_MPI();

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".SY"), "wb"));
  SYSI_CALL(fwrite_bycol(f, n_p, 1u, hH, n_gpu, static_cast<long>(p_)));
  SYSI_CALL(fwrite_bycol(f, n_q, 1u, (hH + n_col), n_gpu, static_cast<long>(q_)));
  SYSI_CALL(fclose(f));
  if (MPI_Barrier(MPI_COMM_WORLD))
    return fini_MPI();

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".SW"), "wb"));
  SYSI_CALL(fwrite_bycol(f, n_p, 1u, hK, n_gpu, static_cast<long>(p_)));
  SYSI_CALL(fwrite_bycol(f, n_q, 1u, (hK + n_col), n_gpu, static_cast<long>(q_)));
  SYSI_CALL(fclose(f));
  if (MPI_Barrier(MPI_COMM_WORLD))
    return fini_MPI();

  if (hK)
    CUDA_CALL(cudaFreeHost(hK));
  if (hH)
    CUDA_CALL(cudaFreeHost(hH));
  if (hS)
    CUDA_CALL(cudaFreeHost(hS));
#ifdef USE_COMPLEX
  if (hVJ)
    CUDA_CALL(cudaFreeHost(hVJ));
  if (hVD)
    CUDA_CALL(cudaFreeHost(hVD));
  if (hGJ)
    CUDA_CALL(cudaFreeHost(hGJ));
  if (hGD)
    CUDA_CALL(cudaFreeHost(hGD));
  if (hFJ)
    CUDA_CALL(cudaFreeHost(hFJ));
  if (hFD)
    CUDA_CALL(cudaFreeHost(hFD));
#else // !USE_COMPLEX
  if (hV)
    CUDA_CALL(cudaFreeHost(hV));
  if (hG)
    CUDA_CALL(cudaFreeHost(hG));
  if (hF)
    CUDA_CALL(cudaFreeHost(hF));
#endif // ?USE_COMPLEX

  free(buf);  
  free_strats();

  // for profiling
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaDeviceReset());

  return fini_MPI();
}
