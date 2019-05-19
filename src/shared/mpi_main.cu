// mpi_main.cu: test driver.

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L3.hpp"

#include "cuda_memory_helper.hpp"
#include "mpi_helper.hpp"
#include "my_utils.hpp"

int main(int argc, char *argv[])
{
  if (10 != argc) {
    (void)fprintf(stderr, "%s SDY SNP0 SNP1 SNP2 ALG MF MG N FN\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char *const ca_exe = argv[0];
  const char *const ca_sdy = argv[1];
  const char *const ca_snp0 = argv[2];
  const char *const ca_snp1 = argv[3];
  const char *const ca_snp2 = argv[4];
  const char *const ca_alg = argv[5];
  const char *const ca_mF = argv[6];
  const char *const ca_mG = argv[7];
  const char *const ca_n = argv[8];
  const char *const ca_fn = argv[9];

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

  if (!*ca_sdy)
    return EXIT_FAILURE;
  if (!*ca_snp0)
    return EXIT_FAILURE;
  if (!*ca_snp1)
    return EXIT_FAILURE;
  if (!*ca_snp2)
    return EXIT_FAILURE;
  if (!*ca_fn)
    return EXIT_FAILURE;

  if (init_MPI(&argc, &argv)) {
    (void)fprintf(stderr, "[%d] %s: init_MPI failed\n", mpi_rank, ca_exe);
    return fini_MPI();
  }
  if (mpi_size < 2) {
    (void)fprintf(stderr, "[%d] %s: MPI_COMM_WORLD size (%d) < 2\n", mpi_rank, mpi_size, ca_exe);
    return fini_MPI();
  }
  if (!mpi_cuda_aware) {
    (void)fprintf(stderr, "[%d] %s: MPI is not CUDA aware\n", mpi_rank, ca_exe);
    return fini_MPI();
  }
  const int dev = assign_dev2host();
  if (dev < 0) {
    (void)fprintf(stderr, "[%d] %s: assign_dev2host failed (%d)\n", mpi_rank, ca_exe, dev);
    return fini_MPI();
  }

  const int dcc = configureGPU(dev);
#ifndef NDEBUG
  (void)fprintf(stdout, "[%d] Device %d has CC %d\n", mpi_rank, dev, dcc);
  (void)fflush(stdout);
#endif // !NDEBUG

  unsigned nrowF_ = 0u, nrowG_ = 0u, ncol_ = 0u;
  border_sizes(mpi_size, nrowF, nrowG, ncol, nrowF_, nrowG_, ncol_);
  const unsigned ncol_gpu = ncol_ / static_cast<unsigned>(mpi_size);

  const size_t mF = static_cast<size_t>(nrowF);
  const size_t mF_ = static_cast<size_t>(nrowF_);
  const size_t mG = static_cast<size_t>(nrowG);
  const size_t mG_ = static_cast<size_t>(nrowG_);
  const size_t n = static_cast<size_t>(ncol);
  const size_t n_ = static_cast<size_t>(ncol_);
  const size_t n_gpu = static_cast<size_t>(ncol_gpu);

  unsigned
    ldhF = nrowF_,
    ldhG = nrowG_,
    ldhV = ncol_;

  const unsigned n0 = (HZ_L1_NCOLB << 1u);
  const unsigned n1 = udiv_ceil(ncol_gpu, HZ_L1_NCOLB);
  const unsigned n2 = (static_cast<unsigned>(mpi_size) << 1u);
  init_strats(ca_sdy, ca_snp0, n0, ca_snp1, n1, ca_snp2, n2);

  char *const buf = static_cast<char*>(calloc(strlen(ca_fn) + 4u, sizeof(char)));
  size_t ldA = static_cast<size_t>(0u);
  FILE *f = static_cast<FILE*>(NULL);

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
  double timing[4] = { -0.0, -0.0, -0.0, -0.0 };

  // for profiling
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaDeviceReset());

  return fini_MPI();
}
