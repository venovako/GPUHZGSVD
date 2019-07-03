#ifndef HZ_HPP
#define HZ_HPP

#include "defines.hpp"

EXTERN_C void border_sizes(const size_t gpus, const size_t mF, const size_t mG, const size_t n, size_t &mF_, size_t &mG_, size_t &n_) throw();
EXTERN_C int bdinit(const size_t m, const size_t n, double *const A, const size_t ldA) throw();

#endif // !HZ_HPP
