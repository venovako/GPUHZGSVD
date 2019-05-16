// main.cu: test driver.

#include "HZ.hpp"
#include "HZ_L2.hpp"

#include "cuda_memory_helper.hpp"
#include "my_utils.hpp"

template <typename CT>
int CT_main(int argc, char *argv[])
{
  if (10 != argc) {
    (void)fprintf(stderr, "%s DEV SDY SNP0 SNP1 ALG MF MG N FN\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char *const ca_exe = argv[0];
  const char *const ca_dev = argv[1];
  const char *const ca_sdy = argv[2];
  const char *const ca_snp0 = argv[3];
  const char *const ca_snp1 = argv[4];
  const char *const ca_alg = argv[5];
  const char *const ca_mF = argv[6];
  const char *const ca_mG = argv[7];
  const char *const ca_n = argv[8];
  const char *const ca_fn = argv[9];

  const unsigned nrowF = static_cast<unsigned>(atoi(ca_mF));
  if (!nrowF)
    return EXIT_SUCCESS;
  const unsigned nrowG = static_cast<unsigned>(atoi(ca_mG));
  if (!nrowG)
    return EXIT_SUCCESS;
  const unsigned ncol = static_cast<unsigned>(atoi(ca_n));
  if (!ncol)
    return EXIT_SUCCESS;

  unsigned nrowF_ = 0u, nrowG_ = 0u, ncol_ = 0u;
  if (border1sz(nrowF, nrowG, ncol, nrowF_, nrowG_, ncol_))
    return EXIT_FAILURE;

  const unsigned routine = static_cast<unsigned>(atoi(ca_alg));

  const int dev = atoi(ca_dev);
  if (dev < 0)
    return EXIT_FAILURE;
  const int dcc = configureGPU(dev);
#ifndef NDEBUG
  (void)fprintf(stdout, "Device %d has CC %d\n", dev, dcc);
  (void)fflush(stdout);
#endif // !NDEBUG

  const unsigned n0 = (HZ_L1_NCOLB << 1u);
  const unsigned n1 = udiv_ceil(ncol_, HZ_L1_NCOLB);
  init_strats(ca_sdy, ca_snp0, n0, ca_snp1, n1);

  const size_t mF = static_cast<size_t>(nrowF);
  const size_t mF_ = static_cast<size_t>(nrowF_);
  const size_t mG = static_cast<size_t>(nrowG);
  const size_t mG_ = static_cast<size_t>(nrowG_);
  const size_t n_ = static_cast<size_t>(ncol_);
  const size_t n = static_cast<size_t>(ncol);

  unsigned
    ldhF = nrowF_,
    ldhG = nrowG_,
    ldhV = ncol_;

  size_t ldA = static_cast<size_t>(0u);
  FILE *f = static_cast<FILE*>(NULL);
  char *const buf = static_cast<char*>(calloc(strlen(ca_fn) + 4u, sizeof(char)));

  ldA = static_cast<size_t>(ldhF);
  CT *const hF = allocHostMtx<CT>(ldA, mF_, n_, true);
  SYSP_CALL(hF);
  ldhF = static_cast<unsigned>(ldA);

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".Y"), "rb"));
  SYSI_CALL(fread_bycol(f, mF, n, hF, ldA));
  SYSI_CALL(fclose(f));
  SYSI_CALL(bdinit(n, n_, hF, ldA));

  ldA = static_cast<size_t>(ldhG);
  CT *const hG = allocHostMtx<CT>(ldA, mG_, n_, true);
  SYSP_CALL(hG);
  ldhG = static_cast<unsigned>(ldA);

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".W"), "rb"));
  SYSI_CALL(fread_bycol(f, mG, n, hG, ldA));
  SYSI_CALL(fclose(f));
  SYSI_CALL(bdinit(n, n_, hG, ldA));

  CT *hV = static_cast<CT*>(NULL);
  ldA = static_cast<size_t>(ldhV);
  hV = allocHostMtx<CT>(ldA, n_, n_, true);
  SYSP_CALL(hV);
  ldhV = static_cast<unsigned>(ldA);
  SYSI_CALL(bdinit(n, n_, hV, ldA));

  double *const hS = allocHostVec<double>(n_);
  SYSP_CALL(hS);
  double *const hH = allocHostVec<double>(n_);
  SYSP_CALL(hH);
  double *const hK = allocHostVec<double>(n_);
  SYSP_CALL(hK);

  unsigned glbSwp = 0u;
  unsigned long long glb_s = 0ull, glb_b = 0ull;
  double timing[4] = { -0.0, -0.0, -0.0, -0.0 };
  int ret = HZ_L2(routine, nrowF_, nrowG_, ncol_, hF, ldhF, hG, ldhG, hV, ldhV, hS, hH, hK, &glbSwp, &glb_s, &glb_b, timing);

  if (ret)
    (void)fprintf(stderr, "%s: error %d\n", ca_exe, ret);
  else {
    (void)fprintf(stdout, "GLB_ROT_S(%20llu), GLB_ROT_B(%20llu)\n", glb_s, glb_b);
    (void)fflush(stdout);
    (void)fprintf(stdout, "%#12.6f s %2u sweeps\n", *timing, glbSwp);
    (void)fflush(stdout);
  }

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".YU"), "wb"));
  SYSI_CALL(fwrite_bycol(f, mF, n, const_cast<const CT*>(hF), static_cast<size_t>(ldhF)));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".WV"), "wb"));
  SYSI_CALL(fwrite_bycol(f, mG, n, const_cast<const CT*>(hG), static_cast<size_t>(ldhG)));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".Z"), "wb"));
  SYSI_CALL(fwrite_bycol(f, n, n, const_cast<const CT*>(hV), static_cast<size_t>(ldhV)));
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
  if (hV)
    CUDA_CALL(cudaFreeHost(hV));
  if (hG)
    CUDA_CALL(cudaFreeHost(hG));
  if (hF)
    CUDA_CALL(cudaFreeHost(hF));
  free(buf);

  // for profiling
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaDeviceReset());

  return ret;
}

#ifdef CT
#error CT already defined
#else // !CT
#ifdef USE_COMPLEX
#define CT std::complex<double>
#else // !USE_COMPLEX
#define CT double
#endif // ?USE_COMPLEX
#endif // ?CT

int main(int argc, char *argv[])
{
  return CT_main<CT>(argc, argv);
}
