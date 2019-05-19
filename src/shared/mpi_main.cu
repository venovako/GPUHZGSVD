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

  // for profiling
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaDeviceReset());

  return fini_MPI();
}
