#ifndef MPI_HELPER_HPP
#define MPI_HELPER_HPP

#include "defines.hpp"

#include <mpi.h>

EXTERN_C int init_MPI(int *const argc, char ***const argv) throw();
EXTERN_C int fini_MPI() throw();
EXTERN_C bool mpi_cuda() throw();
EXTERN_C int assign_dev2host() throw();

#endif // !MPI_HELPER_HPP
