#ifndef MPI_HELPER_HPP
#define MPI_HELPER_HPP

#include "defines.hpp"

#include <mpi.h>

extern int mpi_size;
extern int mpi_rank;
extern bool mpi_cuda_aware;

#ifdef USE_COMPLEX
extern MPI_Datatype DT_V112D;
#endif // USE_COMPLEX

EXTERN_C int init_MPI(int *const argc, char ***const argv) throw();
EXTERN_C int fini_MPI() throw();
EXTERN_C int assign_dev2host() throw();

#endif // !MPI_HELPER_HPP
