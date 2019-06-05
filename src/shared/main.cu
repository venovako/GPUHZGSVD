// main.cu: test driver.

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L2.hpp"

#include "cuda_memory_helper.hpp"
#include "my_utils.hpp"

static int fresize(FILE *const f, const size_t s) throw()
{
  int e = -1;
  if (!f)
    return e;
  e = ftruncate(fileno(f), static_cast<off_t>(s));
  if (e)
    return e;
  e = fflush(f);
  if (e)
    return e;
  return 0;
}

static int fread_bycol
(FILE *const f, const size_t m, const size_t n,
#ifdef USE_COMPLEX
 double *const buf,
 cuD *const AD,
 cuJ *const AJ,
#else // !USE_COMPLEX
 double *const A,
#endif // ?USE_COMPLEX
 const size_t ldA) throw()
{
  if (!f)
    return -1;
  if (!m)
    return 0;
  if (!n)
    return 0;
#ifdef USE_COMPLEX
  if (!buf)
    return -4;
  if (!AD)
    return -5;
  if (!AJ)
    return -6;
  if (ldA < m)
    return -7;
#else // !USE_COMPLEX
  if (!A)
    return -4;
  if (ldA < m)
    return -5;
#endif // ?USE_COMPLEX

  const long co = ftell(f);
  SYSI_CALL(co < 0l);
  if (co)
    SYSI_CALL(fseek(f, 0l, SEEK_SET));

  for (size_t j = 0u; j < n; ++j) {
#ifdef USE_COMPLEX
    SYSI_CALL(fread(buf, (2u * sizeof(double)), m, f) != m);
    const size_t o = (ldA * j);
    cuD *const cD = (AD + o);
    cuJ *const cJ = (AJ + o);
    for (size_t i = 0u; i < m; ++i) {
      const size_t i2 = (i << 1u);
      cD[i] = static_cast<cuD>(buf[i2]);
      cJ[i] = static_cast<cuJ>(buf[i2 + 1u]);
    }
#else // !USE_COMPLEX
    double *const c = (A + ldA * j);
    SYSI_CALL(fread(c, sizeof(double), m, f) != m);
#endif // ?USE_COMPLEX
  }

  return 0;
}

static int fwrite_bycol
(FILE *const f, const size_t m, const size_t n,
#ifdef USE_COMPLEX
 double *const buf,
 const cuD *const AD,
 const cuJ *const AJ,
#else // !USE_COMPLEX
 const double *const A,
#endif // ?USE_COMPLEX
 const size_t ldA) throw()
{
  if (!f)
    return -1;
  if (!m)
    return 0;
  if (!n)
    return 0;
#ifdef USE_COMPLEX
  if (!buf)
    return -4;
  if (!AD)
    return -5;
  if (!AJ)
    return -6;
  if (ldA < m)
    return -7;
#else // !USE_COMPLEX
  if (!A)
    return -4;
  if (ldA < m)
    return -5;
#endif // ?USE_COMPLEX

  const long co = ftell(f);
  SYSI_CALL(co < 0l);
  if (co)
    SYSI_CALL(fseek(f, 0l, SEEK_SET));

  for (size_t j = 0u; j < n; ++j) {
#ifdef USE_COMPLEX
    const size_t o = (ldA * j);
    const cuD *const cD = (AD + o);
    const cuJ *const cJ = (AJ + o);
    for (size_t i = 0u; i < m; ++i) {
      const size_t i2 = (i << 1u);
      buf[i2] = static_cast<double>(cD[i]);
      buf[i2 + 1u] = static_cast<double>(cJ[i]);
    }
    SYSI_CALL(fwrite(buf, (2u * sizeof(double)), m, f) != m);
#else // !USE_COMPLEX
    const double *const c = (A + ldA * j);
    SYSI_CALL(fwrite(c, sizeof(double), m, f) != m);
#endif // ?USE_COMPLEX
  }

  return 0;
}

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

  const int dcc = configureGPU(dev);
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
#ifdef USE_COMPLEX
  double *const buf = static_cast<double*>(calloc((((mF >= mG) ? mF : mG) * 2u), sizeof(double)));
  SYSP_CALL(buf);
#endif // USE_COMPLEX

  FILE *f = static_cast<FILE*>(NULL);
  size_t ldA = static_cast<size_t>(0u);

  ldA = ldhF;
#ifdef USE_COMPLEX
  cuD *const hFD = allocHostMtx<cuD>(ldA, mF_, n_, true);
  SYSP_CALL(hFD);
  cuJ *const hFJ = allocHostMtx<cuJ>(ldA, mF_, n_, true);
  SYSP_CALL(hFJ);
#else // !USE_COMPLEX
  double *const hF = allocHostMtx<double>(ldA, mF_, n_, true);
  SYSP_CALL(hF);
#endif // ?USE_COMPLEX
  ldhF = ldA;

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".Y"), "rb"));
#ifdef USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mF, n, buf, hFD, hFJ, ldA));
#else // !USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mF, n, hF, ldA));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));
#ifdef USE_COMPLEX
  SYSI_CALL(bdinit(mF, (n_ - n), (hFD + ldA * n), ldA));
#else // !USE_COMPLEX
  SYSI_CALL(bdinit(mF, (n_ - n), (hF + ldA * n), ldA));
#endif // ?USE_COMPLEX

  ldA = ldhG;
#ifdef USE_COMPLEX
  cuD *const hGD = allocHostMtx<cuD>(ldA, mG_, n_, true);
  SYSP_CALL(hGD);
  cuJ *const hGJ = allocHostMtx<cuJ>(ldA, mG_, n_, true);
  SYSP_CALL(hGJ);
#else // !USE_COMPLEX
  double *const hG = allocHostMtx<double>(ldA, mG_, n_, true);
  SYSP_CALL(hG);
#endif // ?USE_COMPLEX
  ldhG = ldA;

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".W"), "rb"));
#ifdef USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mG, n, buf, hGD, hGJ, ldA));
#else // !USE_COMPLEX
  SYSI_CALL(fread_bycol(f, mG, n, hG, ldA));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));
#ifdef USE_COMPLEX
  SYSI_CALL(bdinit(mG, (n_ - n), (hGD + ldA * n), ldA));
#else // !USE_COMPLEX
  SYSI_CALL(bdinit(mG, (n_ - n), (hG + ldA * n), ldA));
#endif // ?USE_COMPLEX

  ldA = ldhV;
#ifdef USE_COMPLEX
  cuD *const hVD = allocHostMtx<cuD>(ldA, n_, n_, true);
  SYSP_CALL(hVD);
  cuJ *const hVJ = allocHostMtx<cuJ>(ldA, n_, n_, true);
  SYSP_CALL(hVJ);
#else // !USE_COMPLEX
  double *const hV = allocHostMtx<double>(ldA, n_, n_, true);
  SYSP_CALL(hV);
#endif // ?USE_COMPLEX
  ldhV = ldA;

#ifdef USE_COMPLEX
  SYSI_CALL(bdinit(n, (n_ - n), (hVD + ldA * n), ldA));
#else // !USE_COMPLEX
  SYSI_CALL(bdinit(n, (n_ - n), (hV + ldA * n), ldA));
#endif // ?USE_COMPLEX

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
  const int ret = HZ_L2(routine, mF_, mG_, n_, hFD, hFJ, ldhF, hGD, hGJ, ldhG, hVD, hVJ, ldhV, hS, hH, hK, glbSwp, glb_s, glb_b, timing);
#else // !USE_COMPLEX
  const int ret = HZ_L2(routine, mF_, mG_, n_, hF, ldhF, hG, ldhG, hV, ldhV, hS, hH, hK, glbSwp, glb_s, glb_b, timing);
#endif // ?USE_COMPLEX

  if (ret)
    (void)fprintf(stderr, "%s: error %d\n", ca_exe, ret);
  else {
    (void)fprintf(stdout, "GLB_ROT_S(%15llu), GLB_ROT_B(%15llu)\n", glb_s, glb_b);
    (void)fprintf(stdout, "%#16.6f s %2u sweeps\n", *timing, glbSwp);
    (void)fflush(stdout);
  }

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".YU"), "wb"));
  ldA = (mF * n * sizeof(double))
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
    ;
  SYSI_CALL(fresize(f, ldA));
#ifdef USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mF, n, buf, hFD, hFJ, ldhF));
#else // !USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mF, n, hF, ldhF));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".WV"), "wb"));
  ldA = (mG * n * sizeof(double))
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
    ;
  SYSI_CALL(fresize(f, ldA));
#ifdef USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mG, n, buf, hGD, hGJ, ldhG));
#else // !USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, mG, n, hG, ldhG));
#endif // ?USE_COMPLEX
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(fn, ca_fn), ".Z"), "wb"));
  ldA = (n * n * sizeof(double))
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
    ;
  SYSI_CALL(fresize(f, ldA));
#ifdef USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, n, n, buf, hVD, hVJ, ldhV));
#else // !USE_COMPLEX
  SYSI_CALL(fwrite_bycol(f, n, n, hV, ldhV));
#endif // ?USE_COMPLEX
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

#ifdef USE_COMPLEX
  free(buf);
#endif // USE_COMPLEX
  free(fn);
  free_strats();

  // for profiling
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaDeviceReset());

  return ret;
}
