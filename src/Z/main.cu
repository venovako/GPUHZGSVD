// main.cu: test driver.

#include "HZ.hpp"
#include "HZ_L2.hpp"

#include "cuda_memory_helper.hpp"
#include "my_utils.hpp"

int main(int argc, char *argv[])
{
  if (9 != argc) {
    (void)fprintf(stderr, "%s DEV SDY SNP0 SNP1 ALG M N FN\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char *const ca_exe = argv[0];
  const char *const ca_dev = argv[1];
  const char *const ca_sdy = argv[2];
  const char *const ca_snp0 = argv[3];
  const char *const ca_snp1 = argv[4];
  const char *const ca_alg = argv[5];
  const char *const ca_m = argv[6];
  const char *const ca_n = argv[7];
  const char *const ca_fn = argv[8];

  const unsigned nrow = static_cast<unsigned>(atoi(ca_m));
  if (!nrow)
    return EXIT_SUCCESS;
  const unsigned ncol = static_cast<unsigned>(atoi(ca_n));
  if (!ncol)
    return EXIT_SUCCESS;
  if (ncol > nrow)
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
  const unsigned n1 = (ncol + HZ_L1_NCOLB - 1u) / HZ_L1_NCOLB;
  init_strats(ca_sdy, ca_snp0, n0, ca_snp1, n1);

  const size_t m = static_cast<size_t>(nrow);
  const size_t n = static_cast<size_t>(ncol);
  const size_t mn = m * n;
  const size_t nn = n * n;

  unsigned
    ldhF = nrow,
    ldhG = nrow,
    ldhV = nrow;

  size_t ldA = static_cast<size_t>(0u);
  FILE *f = static_cast<FILE*>(NULL);
  char *const buf = static_cast<char*>(calloc(strlen(ca_fn) + 4u, sizeof(char)));

  ldA = static_cast<size_t>(ldhF);
  std::complex<double> *const hF = allocHostMtx<std::complex<double>>(ldA, m, n, true);
  SYSP_CALL(hF);
  ldhF = static_cast<unsigned>(ldA);

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".Y"), "rb"));
  SYSI_CALL(mn != fread(hF, sizeof(*hF), mn, f));
  SYSI_CALL(fclose(f));

  ldA = static_cast<size_t>(ldhG);
  std::complex<double> *const hG = allocHostMtx<std::complex<double>>(ldA, m, n, true);
  SYSP_CALL(hG);
  ldhG = static_cast<unsigned>(ldA);

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".W"), "rb"));
  SYSI_CALL(mn != fread(hG, sizeof(*hG), mn, f));
  SYSI_CALL(fclose(f));

  std::complex<double> *hV = static_cast<std::complex<double>*>(NULL);
  ldA = static_cast<size_t>(ldhV);
  hV = allocHostMtx<std::complex<double>>(ldA, n, n, true);
  SYSP_CALL(hV);
  ldhV = static_cast<unsigned>(ldA);

  double *const hS = allocHostVec<double>(n);
  SYSP_CALL(hS);
  double *const hH = allocHostVec<double>(n);
  SYSP_CALL(hH);
  double *const hK = allocHostVec<double>(n);
  SYSP_CALL(hK);

  unsigned glbSwp = 0u;
  unsigned long long glb_s = 0ull, glb_b = 0ull;
  double timing[4] = { -0.0, -0.0, -0.0, -0.0 };
  int ret = HZ_L2(routine, nrow, ncol, hF, ldhF, hG, ldhG, hV, ldhV, hS, hH, hK, &glbSwp, &glb_s, &glb_b, timing);

  if (ret)
    (void)fprintf(stderr, "%s: error %d\n", ca_exe, ret);
  else {
    (void)fprintf(stdout, "GLB_ROT_S(%20llu), GLB_ROT_B(%20llu)\n", glb_s, glb_b);
    (void)fflush(stdout);
    (void)fprintf(stdout, "%#12.6f s %2u sweeps\n", *timing, glbSwp);
    (void)fflush(stdout);
  }

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".YU"), "wb"));
  SYSI_CALL(mn != fwrite(hF, sizeof(*hF), mn, f));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".WV"), "wb"));
  SYSI_CALL(mn != fwrite(hG, sizeof(*hG), mn, f));
  SYSI_CALL(fclose(f));

  SYSP_CALL(f = fopen(strcat(strcpy(buf, ca_fn), ".Z"), "wb"));
  SYSI_CALL(nn != fwrite(hV, sizeof(*hV), nn, f));
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
