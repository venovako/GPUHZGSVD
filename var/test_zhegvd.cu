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
  cuDoubleComplex *const hF = allocHostMtx<cuDoubleComplex>(ldhF, mF, n, true);
  SYSP_CALL(hF);
  SYSI_CALL(fread_bycol(f, mF, n, hF, ldhF));
  SYSI_CALL(fclose(f));
  FILE *const g = fopen(strcat(strcpy(fn, argv[4]), ".W"), "rb");
  SYSP_CALL(g);
  size_t ldhG = mG;
  cuDoubleComplex *const hG = allocHostMtx<cuDoubleComplex>(ldhG, mG, n, true);
  SYSP_CALL(hG);
  SYSI_CALL(fread_bycol(g, mG, n, hG, ldhG));
  SYSI_CALL(fclose(g));
  size_t lddA = n;
  cuDoubleComplex *const dA = allocDeviceMtx<cuDoubleComplex>(lddA, n, n, true);
  SYSP_CALL(dA);
  size_t lddB = n;
  cuDoubleComplex *const dB = allocDeviceMtx<cuDoubleComplex>(lddB, n, n, true);
  SYSP_CALL(dB);
  double *const dW = allocDeviceVec<double>(n + 1u);
  SYSP_CALL(dW);
  int *const dinfo = reinterpret_cast<int*>(dW + n);
  int lwork = 0;
  CUSOLVER_CALL(cusolverDnZhegvd_bufferSize(csh, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, lddA, dB, lddB, dW, &lwork));
  if (lwork < 0)
    return EXIT_FAILURE;
  (void)fprintf(stdout, "%d,", lwork);
  (void)fflush(stdout);
  const unsigned mM = ((mF >= mG) ? mF : mG);
  size_t lddM = mM;
  size_t n2 = (static_cast<size_t>(n) * 2u);
  if ((lddM * n2) < lwork)
    n2 = ((static_cast<size_t>(lwork) + (lddM - 1u)) / lddM);
  cuDoubleComplex *const dwork = allocDeviceMtx<cuDoubleComplex>(lddM, mM, n2, true);
  SYSP_CALL(dwork);
  const size_t lddF = lddM;
  cuDoubleComplex *const dF = dwork;
  const size_t lddG = lddM;
  cuDoubleComplex *const dG = dwork + (lddM * n);
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy2D(dF, lddF * sizeof(cuDoubleComplex), hF, ldhF * sizeof(cuDoubleComplex), mF * sizeof(cuDoubleComplex), n, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy2D(dG, lddG * sizeof(cuDoubleComplex), hG, ldhG * sizeof(cuDoubleComplex), mG * sizeof(cuDoubleComplex), n, cudaMemcpyHostToDevice));
  const double alpha = 1.0;
  const double beta = 0.0;
  long long sw = 0ll;
  CUDA_CALL(cudaDeviceSynchronize());
  stopwatch_reset(sw);
  CUBLAS_CALL(cublasZherk(cbh, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, n, mF, &alpha, dF, lddF, &beta, dA, lddA));
  CUBLAS_CALL(cublasZherk(cbh, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, n, mG, &alpha, dG, lddG, &beta, dB, lddB));
  CUDA_CALL(cudaDeviceSynchronize());
  (void)fprintf(stdout, "%lld,", stopwatch_lap(sw));
  (void)fflush(stdout);
  CUSOLVER_CALL(cusolverDnZhegvd(csh, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, lddA, dB, lddB, dW, dwork, lwork, dinfo));
  CUDA_CALL(cudaDeviceSynchronize());
  (void)fprintf(stdout, "%lld,", stopwatch_lap(sw));
  (void)fflush(stdout);
  CUDA_CALL(cudaFree(dwork));
  double *const hW = allocHostVec<double>(n + 1u);
  SYSP_CALL(hW);
  CUDA_CALL(cudaMemcpy(hW, dW, (n + 1u) * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaFree(dW));
  CUDA_CALL(cudaMemcpy2D(hG, ldhG * sizeof(cuDoubleComplex), dB, lddB * sizeof(cuDoubleComplex), n * sizeof(cuDoubleComplex), n, cudaMemcpyDeviceToHost));
  (void)fprintf(stdout, "%d\n", *reinterpret_cast<const int*>(hW + n));
  (void)fflush(stdout);
  FILE *const l = fopen(strcat(strcpy(fn, argv[4]), ".L"), "wb");
  SYSP_CALL(l);
  SYSI_CALL(fwrite(hW, sizeof(double), n, l) != n);
  SYSI_CALL(fclose(l));
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaFree(dB));
  CUDA_CALL(cudaMemcpy2D(hF, ldhF * sizeof(cuDoubleComplex), dA, lddA * sizeof(cuDoubleComplex), n * sizeof(cuDoubleComplex), n, cudaMemcpyDeviceToHost));
  FILE *const b = fopen(strcat(strcpy(fn, argv[4]), ".B"), "wb");
  SYSP_CALL(b);
  n2 = n * static_cast<size_t>(n);
  SYSI_CALL(fresize(b, n2 * sizeof(cuDoubleComplex)));
  SYSI_CALL(fwrite_bycol(b, n, n, hG, ldhG));
  SYSI_CALL(fclose(b));
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaFree(dA));
  FILE *const e = fopen(strcat(strcpy(fn, argv[4]), ".E"), "wb");
  SYSP_CALL(e);
  SYSI_CALL(fresize(e, n2 * sizeof(cuDoubleComplex)));
  SYSI_CALL(fwrite_bycol(e, n, n, hF, ldhF));
  SYSI_CALL(fclose(e));
  CUDA_CALL(cudaFreeHost(hW));
  CUDA_CALL(cudaFreeHost(hG));
  CUDA_CALL(cudaFreeHost(hF));
  free(fn);
  freeGPU(cbh, csh);
  return EXIT_SUCCESS;
}
