#include "cuda_memory_helper.hpp"

int main(int argc, char *argv[])
{
  if ((argc < 5) || (argc > 6)) {
    (void)fprintf(stderr, "%s mF mG n fn [dev]\n", *argv);
    (void)fflush(stderr);
    return EXIT_FAILURE;
  }
  const unsigned mF = static_cast<unsigned>(atoi(argv[1]));
  if (!mF)
    return EXIT_FAILURE;
  const unsigned mG = static_cast<unsigned>(atoi(argv[2]));
  if (!mG)
    return EXIT_FAILURE;
  const unsigned n = static_cast<unsigned>(atoi(argv[3]));
  if (!n)
    return EXIT_FAILURE;
  if (mF < n)
    return EXIT_FAILURE;
  if (mG < n)
    return EXIT_FAILURE;
  const int dev = ((argc == 6) ? atoi(argv[5]) : 0);
  cublasHandle_t cbh = 0;
  cusolverDnHandle_t csh = 0;
  const int dcc = configureGPU(dev, cbh, csh);
#ifndef NDEBUG
  (void)fprintf(stderr, "GPU device %d is of compute capability %d.\n", dev, dcc);
  (void)fflush(stderr);
#endif /* !NDEBUG */
  char *const fn = static_cast<char*>(calloc(strlen(argv[4]) + 3u, sizeof(char)));
  SYSP_CALL(fn);
  FILE *const f = fopen(strcat(strcpy(fn, argv[4]), ".Y"), "rb");
  SYSP_CALL(f);
  size_t ldhF = mF;
  double *const hF = allocHostMtx<double>(ldhF, mF, n, true);
  SYSP_CALL(hF);
  SYSI_CALL(fread_bycol(f, mF, n, hF, ldhF));
  SYSI_CALL(fclose(f));
  FILE *const g = fopen(strcat(strcpy(fn, argv[4]), ".W"), "rb");
  SYSP_CALL(g);
  size_t ldhG = mG;
  double *const hG = allocHostMtx<double>(ldhG, mG, n, true);
  SYSP_CALL(hG);
  SYSI_CALL(fread_bycol(g, mG, n, hG, ldhG));
  SYSI_CALL(fclose(g));
  size_t lddF = mF;
  double *const dF = allocDeviceMtx<double>(lddF, mF, n, true);
  SYSP_CALL(dF);
  size_t lddG = mG;
  double *const dG = allocDeviceMtx<double>(lddG, mG, n, true);
  SYSP_CALL(dG);
  size_t lddA = n;
  double *const dA = allocDeviceMtx<double>(lddA, n, n, true);
  SYSP_CALL(dA);
  size_t lddB = n;
  double *const dB = allocDeviceMtx<double>(lddB, n, n, true);
  SYSP_CALL(dB);
  double *const dW = allocDeviceVec<double>(n + 1u);
  SYSP_CALL(dW);
  int *const dinfo = reinterpret_cast<int*>(dW + n);
  int lwork = 0;
  CUSOLVER_CALL(cusolverDnDsygvd_bufferSize(csh, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, lddA, dB, lddB, dW, &lwork));
  if (lwork < 0)
    return EXIT_FAILURE;
  (void)fprintf(stdout, "%d,", lwork);
  (void)fflush(stdout);
  const size_t mM = ((mF >= mG) ? mF : mG);
  size_t lddM = ((lddF >= lddG) ? lddF : lddG);
  size_t n2 = static_cast<size_t>(n) * 2u;
  if ((lddM * n2) < lwork)
    n2 = ((static_cast<size_t>(lwork) + (lddM - 1u)) / lddM);
  double *const dwork = allocDeviceMtx<double>(lddM, mM, n2, true);
  SYSP_CALL(dwork);
  const size_t lddU = lddM;
  double *const dU = dwork;
  const size_t lddV = lddM;
  double *const dV = dwork + (lddM * n);
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy2D(dF, lddF * sizeof(double), hF, ldhF * sizeof(double), mF * sizeof(double), n, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy2D(dG, lddG * sizeof(double), hG, ldhG * sizeof(double), mG * sizeof(double), n, cudaMemcpyHostToDevice));
  const double alpha = 1.0;
  const double beta = 0.0;
  long long sw = 0ll, t = 0ll, tt = 0ll;
  CUDA_CALL(cudaDeviceSynchronize());
  stopwatch_reset(sw);
  CUBLAS_CALL(cublasDsyrk(cbh, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, mF, &alpha, dF, lddF, &beta, dA, lddA));
  CUBLAS_CALL(cublasDsyrk(cbh, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, mG, &alpha, dG, lddG, &beta, dB, lddB));
  CUDA_CALL(cudaDeviceSynchronize());
  (void)fprintf(stdout, "%lld,", (t = stopwatch_lap(sw)));
  (void)fflush(stdout);
  tt += t;
  CUSOLVER_CALL(cusolverDnDsygvd(csh, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, lddA, dB, lddB, dW, dwork, lwork, dinfo));
  CUDA_CALL(cudaDeviceSynchronize());
  (void)fprintf(stdout, "%lld,", (t = stopwatch_lap(sw)));
  (void)fflush(stdout);
  tt += t;
  CUBLAS_CALL(cublasDgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, mF, n, n, &alpha, dF, lddF, dA, lddA, &beta, dU, lddU));
  CUBLAS_CALL(cublasDgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, mG, n, n, &alpha, dG, lddG, dA, lddA, &beta, dV, lddV));
  CUDA_CALL(cudaDeviceSynchronize());
  (void)fprintf(stdout, "%lld,", (t = stopwatch_lap(sw)));
  (void)fflush(stdout);
  tt += t;
  CUDA_CALL(cudaFree(dG));
  CUDA_CALL(cudaFree(dF));
  double *const hW = allocHostVec<double>(n + 1u);
  SYSP_CALL(hW);
  CUDA_CALL(cudaMemcpy(hW, dW, (n + 1u) * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy2D(hF, ldhF * sizeof(double), dU, lddU * sizeof(double), mF * sizeof(double), n, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy2D(hG, ldhG * sizeof(double), dV, lddV * sizeof(double), mG * sizeof(double), n, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaFree(dwork));
  CUDA_CALL(cudaFree(dW));
  (void)fprintf(stdout, "%lld,%d\n", tt, *reinterpret_cast<const int*>(hW + n));
  (void)fflush(stdout);
  FILE *const l = fopen(strcat(strcpy(fn, argv[4]), ".L"), "wb");
  SYSP_CALL(l);
  SYSI_CALL(fwrite(hW, sizeof(double), n, l) != n);
  SYSI_CALL(fclose(l));
  CUDA_CALL(cudaFreeHost(hW));
  FILE *const v = fopen(strcat(strcpy(fn, argv[4]), ".V"), "wb");
  SYSP_CALL(v);
  SYSI_CALL(fresize(v, mG * (n * sizeof(double))));
  SYSI_CALL(fwrite_bycol(v, mG, n, hG, ldhG));
  SYSI_CALL(fclose(v));
  CUDA_CALL(cudaMemcpy2D(hG, ldhG * sizeof(double), dB, lddB * sizeof(double), n * sizeof(double), n, cudaMemcpyDeviceToHost));
  FILE *const u = fopen(strcat(strcpy(fn, argv[4]), ".U"), "wb");
  SYSP_CALL(u);
  SYSI_CALL(fresize(u, mF * (n * sizeof(double))));
  SYSI_CALL(fwrite_bycol(u, mF, n, hF, ldhF));
  SYSI_CALL(fclose(u));
  CUDA_CALL(cudaMemcpy2D(hF, ldhF * sizeof(double), dA, lddA * sizeof(double), n * sizeof(double), n, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaFree(dB));
  CUDA_CALL(cudaFree(dA));
  for (unsigned j = 1u; j < n; ++j) {
    double *const Gj = hG + j * ldhG;
    for (unsigned i = 0u; i < j; ++i)
      Gj[i] = beta;
  }
  FILE *const b = fopen(strcat(strcpy(fn, argv[4]), ".B"), "wb");
  SYSP_CALL(b);
  SYSI_CALL(fresize(b, n * (n * sizeof(double))));
  SYSI_CALL(fwrite_bycol(b, n, n, hG, ldhG));
  SYSI_CALL(fclose(b));
  CUDA_CALL(cudaFreeHost(hG));
  FILE *const e = fopen(strcat(strcpy(fn, argv[4]), ".E"), "wb");
  SYSP_CALL(e);
  SYSI_CALL(fresize(e, n * (n * sizeof(double))));
  SYSI_CALL(fwrite_bycol(e, n, n, hF, ldhF));
  SYSI_CALL(fclose(e));
  CUDA_CALL(cudaFreeHost(hF));
  free(fn);
  freeGPU(cbh, csh);
  return EXIT_SUCCESS;
}
