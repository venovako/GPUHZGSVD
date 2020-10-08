// main.cu: test driver.

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L2.hpp"

#include "cuda_memory_helper.hpp"

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

  const size_t mF = atou(ca_mF);
  if (!mF)
    return EXIT_SUCCESS;
  const size_t mG = atou(ca_mG);
  if (!mG)
    return EXIT_SUCCESS;
  const size_t n = atou(ca_n);
  if (!n)
    return EXIT_SUCCESS;
  if (n > mF) {
    (void)fprintf(stderr, "N(%u) > MF(%u)\n", n, mF);
    return EXIT_FAILURE;
  }
  if (n > mG) {
    (void)fprintf(stderr, "N(%u) > MG(%u)\n", n, mG);
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

  cublasHandle_t handle = static_cast<cublasHandle_t>(NULL);
  const int dcc = configureGPU(dev, handle);
  (void)fprintf(stderr, "device(%d) has CC(%d)\n", dev, dcc);
  (void)fflush(stderr);

  size_t mF_ = 0u, mG_ = 0u, n_ = 0u;
  const size_t gpus = static_cast<size_t>(1u);
  border_sizes(gpus, mF, mG, n, mF_, mG_, n_);

  const size_t n0 = (HZ_L1_NCOLB << 1u);
  const size_t n1 = udiv_ceil(n_, static_cast<size_t>(HZ_L1_NCOLB));
  init_strats(snp0, n0, snp1, n1);

  size_t
    ldhF = mF_,
    ldhG = mG_,
    ldhV = n_;

  char *const fn = static_cast<char*>(calloc(strlen(ca_fn) + 4u, sizeof(char)));
  SYSP_CALL(fn);

  FILE *f = static_cast<FILE*>(NULL);
  size_t ldA = static_cast<size_t>(0u);

  ldA = ldhF;
  double *const hF = allocHostMtx<double>(ldA, mF_, n_, true);
  SYSP_CALL(hF);
  ldhF = ldA;

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".Y"), "rb"));
  SYSI_CALL(fread_bycol(f, mF, n, hF, ldA));
  SYSI_CALL(fclose(f));
  SYSI_CALL(bdinit(mF, (n_ - n), (hF + ldA * n), ldA));

  ldA = ldhG;
  double *const hG = allocHostMtx<double>(ldA, mG_, n_, true);
  SYSP_CALL(hG);
  ldhG = ldA;

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".W"), "rb"));
  SYSI_CALL(fread_bycol(f, mG, n, hG, ldA));
  SYSI_CALL(fclose(f));
  SYSI_CALL(bdinit(mG, (n_ - n), (hG + ldA * n), ldA));

  ldA = ldhV;
  double *const hV = allocHostMtx<double>(ldA, n_, n_, true);
  SYSP_CALL(hV);
  ldhV = ldA;

  SYSI_CALL(bdinit(n, (n_ - n), (hV + ldA * n), ldA));

  double *const hS = allocHostVec<double>(n_);
  SYSP_CALL(hS);
  double *const hH = allocHostVec<double>(n_);
  SYSP_CALL(hH);
  double *const hK = allocHostVec<double>(n_);
  SYSP_CALL(hK);

  unsigned glbSwp = 0u;
  unsigned long long glb_s = 0ull, glb_b = 0ull;
  double timing[4u] = { -0.0, -0.0, -0.0, -0.0 };
  const int ret = HZ_L2(routine, mF_, mG_, n_, hF, ldhF, hG, ldhG, hV, ldhV, hS, hH, hK, glbSwp, glb_s, glb_b, timing, handle);

  if (ret)
    (void)fprintf(stderr, "%s: error %d\n", ca_exe, ret);
  else {
    (void)fprintf(stdout, "GLB_ROT_S(%15llu), GLB_ROT_B(%15llu)\n", glb_s, glb_b);
    (void)fprintf(stdout, "%#16.6f s %2u sweeps\n", *timing, glbSwp);
    (void)fflush(stdout);
  }

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".YU"), "wb"));
  ldA = (mF * n * sizeof(double));
  SYSI_CALL(fresize(f, ldA));
  SYSI_CALL(fwrite_bycol(f, mF, n, hF, ldhF));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".WV"), "wb"));
  ldA = (mG * n * sizeof(double));
  SYSI_CALL(fresize(f, ldA));
  SYSI_CALL(fwrite_bycol(f, mG, n, hG, ldhG));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".Z"), "wb"));
  ldA = (n * n * sizeof(double));
  SYSI_CALL(fresize(f, ldA));
  SYSI_CALL(fwrite_bycol(f, n, n, hV, ldhV));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".SS"), "wb"));
  SYSI_CALL(n != fwrite(hS, sizeof(*hS), n, f));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".SY"), "wb"));
  SYSI_CALL(n != fwrite(hH, sizeof(*hH), n, f));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".SW"), "wb"));
  SYSI_CALL(n != fwrite(hK, sizeof(*hK), n, f));
  SYSI_CALL(fclose(f));

  if (hK)
    CUDA_CALL(cudaFreeHost(hK));
  if (hH)
    CUDA_CALL(cudaFreeHost(hH));
  if (hS)
    CUDA_CALL(cudaFreeHost(hS));
  if (hV)
    CUDA_CALL(cudaFreeHost(hV));
  if (hG)
    CUDA_CALL(cudaFreeHost(hG));
  if (hF)
    CUDA_CALL(cudaFreeHost(hF));

  free(fn);
  free_strats();
  freeGPU(handle);

  return ret;
}
