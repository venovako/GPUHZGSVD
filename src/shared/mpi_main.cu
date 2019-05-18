// mpi_main.cu: test driver.

#include "cuda_memory_helper.hpp"
#include "mpi_helper.hpp"
#include "my_utils.hpp"

#ifdef USE_COMPLEX
typedef std::complex<double> CT;
#else // !USE_COMPLEX
typedef double CT;
#endif // ?USE_COMPLEX

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

  if (init_MPI(&argc, &argv)) {
    (void)fprintf(stderr, "[%d] %s: init_MPI failed\n", mpi_rank, argv[0]);
    return fini_MPI();
  }
  if (mpi_size < 2) {
    (void)fprintf(stderr, "[%d] %s: MPI_COMM_WORLD size < 2\n", mpi_rank, argv[0]);
    return fini_MPI();
  }
  if (!mpi_cuda_aware) {
    (void)fprintf(stderr, "[%d] %s: MPI is not CUDA aware\n", mpi_rank, argv[0]);
    return fini_MPI();
  }
  const int dev = assign_dev2host();
  if (dev < 0) {
    (void)fprintf(stderr, "[%d] %s: assign_dev2host failed (%d)\n", mpi_rank, argv[0], dev);
    return fini_MPI();
  }

  const int dcc = configureGPU(dev);
  (void)fprintf(stdout, "[%d] Device %d has CC %d\n", mpi_rank, dev, dcc);
  (void)fflush(stdout);

  // for profiling
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaDeviceReset());

  return fini_MPI();
}
