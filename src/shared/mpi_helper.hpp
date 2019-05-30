#ifndef MPI_HELPER_HPP
#define MPI_HELPER_HPP

#include "defines.hpp"

#include <mpi.h>

extern int mpi_size;
extern int mpi_rank;
#ifdef USE_MPI_CUDA
extern bool mpi_cuda_aware;
#endif // USE_MPI_CUDA

EXTERN_C int init_MPI(int *const argc, char ***const argv) throw();
EXTERN_C int fini_MPI() throw();
EXTERN_C int assign_dev2host() throw();

#endif // !MPI_HELPER_HPP
