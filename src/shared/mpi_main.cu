// mpi_main.cu: test driver.

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L3.hpp"

#include "my_utils.hpp"
#include "cuda_memory_helper.hpp"

int main(int argc, char *argv[])
{
  if (10 != argc) {
    DIE("Arguments: SDY SNP0 SNP1 SNP2 ALG MF MG N FN");
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

  if (!*ca_sdy) {
    DIE("invalid argument SDY");
  }
  if (!*ca_snp0) {
    DIE("invalid argument SNP0");
  }
  if (!*ca_snp1) {
    DIE("invalid argument SNP1");
  }
  if (!*ca_snp2) {
    DIE("invalid argument SNP2");
  }
  if (!*ca_fn) {
    DIE("invalid argument FN");
  }

  if (init_MPI(&argc, &argv)) {
    (void)fprintf(stderr, "[%d] init_MPI failed\n", mpi_rank);
    return fini_MPI();
  }
  if (mpi_size < 2) {
    if (!mpi_rank)
      (void)fprintf(stderr, "MPI_COMM_WORLD size (%d) < 2\n", mpi_size);
    return fini_MPI();
  }
  if (!mpi_cuda_aware) {
    if (!mpi_rank)
      (void)fprintf(stderr, "MPI is not CUDA aware\n");
    return fini_MPI();
  }

  const unsigned gpus = static_cast<unsigned>(mpi_size);
  const unsigned n2 = (gpus << 1u);
  if (ncol < n2) {
    if (!mpi_rank)
      (void)fprintf(stderr, "N(%u) < n2(%u)\n", ncol, n2);
    return fini_MPI();
  }

  const int dev = assign_dev2host();
  if (dev < 0) {
    if (!mpi_rank)
      (void)fprintf(stderr, "assign_dev2host failed (%d)\n", dev);
    return fini_MPI();
  }

  const int dcc = configureGPU(dev);
#ifndef NDEBUG
  (void)fprintf(stdout, "[%d] device(%d) has CC(%d)\n", mpi_rank, dev, dcc);
  (void)fflush(stdout);
#endif // !NDEBUG

  unsigned nrowF_ = 0u, nrowG_ = 0u, ncol_ = 0u;
  border_sizes(gpus, nrowF, nrowG, ncol, nrowF_, nrowG_, ncol_);
  const unsigned ncol_gpu = ncol_ / gpus;

  const unsigned n0 = (HZ_L1_NCOLB << 1u);
  const unsigned n1 = ncol_gpu / HZ_L1_NCOLB;
  if (ncol_gpu % HZ_L1_NCOLB) {
    if (!mpi_rank)
      (void)fprintf(stderr, "ncol_gpu(%u)\n", ncol_gpu);
    return fini_MPI();
  }
  init_strats(ca_sdy, ca_snp0, n0, ca_snp1, n1, ca_snp2, n2);

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

  char *const buf = static_cast<char*>(calloc(strlen(ca_fn) + 4u, sizeof(char)));
  SYSP_CALL(buf);
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
