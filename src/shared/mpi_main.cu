// mpi_main.cu: test driver.

#include "cuda_memory_helper.hpp"
#include "mpi_helper.hpp"
#include "my_utils.hpp"

template <typename CT>
int CT_main(int argc, char *argv[])
{
  if (init_MPI(&argc, &argv)) {
    (void)fprintf(stderr, "[%d] %s: init_MPI failed\n", mpi_rank, argv[0]);
    return fini_MPI();
  }
  if (!mpi_cuda_aware) {
    (void)fprintf(stderr, "[%d] %s: MPI is not CUDA aware\n", mpi_rank, argv[0]);
    // return fini_MPI();
  }
  const int dev = assign_dev2host();
  if (dev < 0) {
    (void)fprintf(stderr, "[%d] %s: assign_dev2host failed (%d)\n", mpi_rank, argv[0], dev);
    return fini_MPI();
  }

  const int dcc = configureGPU(dev);
  (void)fprintf(stdout, "[%d] Device %d has CC %d\n", mpi_rank, dev, dcc);
  (void)fflush(stdout);

  return fini_MPI();
}

#ifndef CT
#ifdef USE_COMPLEX
#define CT std::complex<double>
#else // !USE_COMPLEX
#define CT double
#endif // ?USE_COMPLEX
#else // CT
#error CT not definable externally
#endif // ?CT

int main(int argc, char *argv[])
{
  return CT_main<CT>(argc, argv);
}
