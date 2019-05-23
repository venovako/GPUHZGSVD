// main.cu: test driver.

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L2.hpp"

#include "cuda_memory_helper.hpp"
#include "my_utils.hpp"

int main(int argc, char *argv[])
{
  if (9 != argc) {
    (void)fprintf(stderr, "%s DEV SNP0 SNP1 ALG MF MG N FN\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char *const ca_exe = argv[0];
  const char *const ca_dev = argv[1];
  const char *const ca_snp0 = argv[2];
  const char *const ca_snp1 = argv[3];
  const char *const ca_alg = argv[4];
  const char *const ca_mF = argv[5];
  const char *const ca_mG = argv[6];
  const char *const ca_n = argv[7];
  const char *const ca_fn = argv[8];

  const int dev = atoi(ca_dev);
  if (dev < 0) {
    (void)fprintf(stderr, "DEV(%d) < 0\n", dev);
    return EXIT_FAILURE;
  }

  const unsigned routine = static_cast<unsigned>(atou(ca_alg));
  if (routine && (routine != 8u)) {
    (void)fprintf(stderr, "ALG(%d) \\notin { 0, 8 }\n", routine);
    return EXIT_FAILURE;
  }

  const unsigned nrowF = static_cast<unsigned>(atou(ca_mF));
  if (!nrowF)
    return EXIT_SUCCESS;
  const unsigned nrowG = static_cast<unsigned>(atou(ca_mG));
  if (!nrowG)
    return EXIT_SUCCESS;
  const unsigned ncol = static_cast<unsigned>(atou(ca_n));
  if (!ncol)
    return EXIT_SUCCESS;
  if (ncol > nrowF) {
    (void)fprintf(stderr, "N(%u) > MF(%u)\n", ncol, nrowF);
    return EXIT_FAILURE;
  }
  if (ncol > nrowG) {
    (void)fprintf(stderr, "N(%u) > MG(%u)\n", ncol, nrowG);
    return EXIT_FAILURE;
  }

  const unsigned snp0 = static_cast<unsigned>(atou(ca_snp0));
  if ((snp0 != STRAT_CYCWOR) && (snp0 != STRAT_MMSTEP)) {
    (void)fprintf(stderr, "SNP0(%u) \\notin { 2, 4 }\n", snp0);
    return EXIT_FAILURE;
  }
  const unsigned snp1 = static_cast<unsigned>(atou(ca_snp1));
  if ((snp1 != STRAT_CYCWOR) && (snp1 != STRAT_MMSTEP)) {
    (void)fprintf(stderr, "SNP1(%u) \\notin { 2, 4 }\n", snp1);
    return EXIT_FAILURE;
  }
  if (!*ca_fn)
    return EXIT_FAILURE;

  const int dcc = configureGPU(dev);
#ifndef NDEBUG
  (void)fprintf(stdout, "device(%d) has CC(%d)\n", dev, dcc);
  (void)fflush(stdout);
#endif // !NDEBUG

  unsigned nrowF_ = 0u, nrowG_ = 0u, ncol_ = 0u;
  border_sizes(1u, nrowF, nrowG, ncol, nrowF_, nrowG_, ncol_);

  const unsigned n0 = (HZ_L1_NCOLB << 1u);
  const unsigned n1 = udiv_ceil(ncol_, HZ_L1_NCOLB);
  init_strats(snp0, n0, snp1, n1);

  const size_t mF = static_cast<size_t>(nrowF);
  const size_t mF_ = static_cast<size_t>(nrowF_);
  const size_t mG = static_cast<size_t>(nrowG);
  const size_t mG_ = static_cast<size_t>(nrowG_);
  const size_t n = static_cast<size_t>(ncol);
  const size_t n_ = static_cast<size_t>(ncol_);

  unsigned
    ldhF = nrowF_,
    ldhG = nrowG_,
    ldhV = ncol_;

  char *const buf = static_cast<char*>(calloc(strlen(ca_fn) + 4u, sizeof(char)));
  SYSP_CALL(buf);
  size_t ldA = static_cast<size_t>(0u);
  FILE *f = static_cast<FILE*>(NULL);

  ldA = static_cast<size_t>(ldhF);
#ifdef USE_COMPLEX
  cuD *const hFD = allocHostMtx<cuD>(ldA, mF_, n_, true);
  SYSP_CALL(hFD);
  cuJ *const hFJ = allocHostMtx<cuJ>(ldA, mF_, n_, true);
  SYSP_CALL(hFJ);
#else // !USE_COMPLEX
  double *const hF = allocHostMtx<double>(ldA, mF_, n_, true);
  SYSP_CALL(hF);
#endif // ?USE_COMPLEX
  ldhF = static_cast<unsigned>(ldA);

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".Y"), "rb"));
#ifdef USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mF, n, hFD, ldA, 0l, 2l));
  SYSI_CALL(fread_bycol(f, mF, n, hFJ, ldA, 1l, 2l));
#else // !USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mF, n, hF, ldA));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));
#ifdef USE_COMPLEX
  SYSI_CALL(bdinit(mF, n, n_, hFD, ldA));
#else // !USE_COMPLEX
  SYSI_CALL(bdinit(mF, n, n_, hF, ldA));
#endif // ?USE_COMPLEX

  ldA = static_cast<size_t>(ldhG);
#ifdef USE_COMPLEX
  cuD *const hGD = allocHostMtx<cuD>(ldA, mG_, n_, true);
  SYSP_CALL(hGD);
  cuJ *const hGJ = allocHostMtx<cuJ>(ldA, mG_, n_, true);
  SYSP_CALL(hGJ);
#else // !USE_COMPLEX
  double *const hG = allocHostMtx<double>(ldA, mG_, n_, true);
  SYSP_CALL(hG);
#endif // ?USE_COMPLEX
  ldhG = static_cast<unsigned>(ldA);

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".W"), "rb"));
#ifdef USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mG, n, hGD, ldA, 0l, 2l));
  SYSI_CALL(fread_bycol(f, mG, n, hGJ, ldA, 1l, 2l));
#else // !USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mG, n, hG, ldA));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));
#ifdef USE_COMPLEX
  SYSI_CALL(bdinit(mG, n, n_, hGD, ldA));
#else // !USE_COMPLEX
  SYSI_CALL(bdinit(mG, n, n_, hG, ldA));
#endif // ?USE_COMPLEX

  ldA = static_cast<size_t>(ldhV);
#ifdef USE_COMPLEX
  cuD *const hVD = allocHostMtx<cuD>(ldA, n_, n_, true);
  SYSP_CALL(hVD);
  cuJ *const hVJ = allocHostMtx<cuJ>(ldA, n_, n_, true);
  SYSP_CALL(hVJ);
#else // !USE_COMPLEX
  double *const hV = allocHostMtx<double>(ldA, n_, n_, true);
  SYSP_CALL(hV);
#endif // ?USE_COMPLEX
  ldhV = static_cast<unsigned>(ldA);

  double *const hS = allocHostVec<double>(n_);
  SYSP_CALL(hS);
  double *const hH = allocHostVec<double>(n_);
  SYSP_CALL(hH);
  double *const hK = allocHostVec<double>(n_);
  SYSP_CALL(hK);

  unsigned glbSwp = 0u;
  unsigned long long glb_s = 0ull, glb_b = 0ull;
  double timing[4u] = { -0.0, -0.0, -0.0, -0.0 };
#ifdef USE_COMPLEX
  int ret = HZ_L2(routine, nrowF_, nrowG_, ncol_, hFD, hFJ, ldhF, hGD, hGJ, ldhG, hVD, hVJ, ldhV, hS, hH, hK, &glbSwp, &glb_s, &glb_b, timing);
#else // !USE_COMPLEX
  int ret = HZ_L2(routine, nrowF_, nrowG_, ncol_, hF, ldhF, hG, ldhG, hV, ldhV, hS, hH, hK, &glbSwp, &glb_s, &glb_b, timing);
#endif // ?USE_COMPLEX

  if (ret)
    (void)fprintf(stderr, "%s: error %d\n", ca_exe, ret);
  else {
    (void)fprintf(stdout, "GLB_ROT_S(%20llu), GLB_ROT_B(%20llu)\n", glb_s, glb_b);
    (void)fflush(stdout);
    (void)fprintf(stdout, "%#12.6f s %2u sweeps\n", *timing, glbSwp);
    (void)fflush(stdout);
  }

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".YU"), "wb"));
  ldA = (mF * n * sizeof(double))
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
    ;
  SYSI_CALL(fresize(f, ldA));
#ifdef USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mF, n, hFD, ldhF, 0l, 2l));
  SYSI_CALL(fwrite_bycol(f, mF, n, hFJ, ldhF, 1l, 2l));
#else // !USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mF, n, hF, ldhF));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".WV"), "wb"));
  ldA = (mG * n * sizeof(double))
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
    ;
  SYSI_CALL(fresize(f, ldA));
#ifdef USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mG, n, hGD, ldhG, 0l, 2l));
  SYSI_CALL(fwrite_bycol(f, mG, n, hGJ, ldhG, 1l, 2l));
#else // !USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mG, n, hG, ldhG));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".Z"), "wb"));
  ldA = (n * n * sizeof(double))
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
    ;
  SYSI_CALL(fresize(f, ldA));
#ifdef USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, n, n, hVD, ldhV, 0l, 2l));
  SYSI_CALL(fwrite_bycol(f, n, n, hVJ, ldhV, 1l, 2l));
#else // !USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, n, n, hV, ldhV));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".SS"), "wb"));
  SYSI_CALL(n != fwrite(hS, sizeof(*hS), n, f));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".SY"), "wb"));
  SYSI_CALL(n != fwrite(hH, sizeof(*hH), n, f));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".SW"), "wb"));
  SYSI_CALL(n != fwrite(hK, sizeof(*hK), n, f));
  SYSI_CALL(fclose(f));

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

  return ret;
}
