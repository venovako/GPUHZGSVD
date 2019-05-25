#ifndef HZ_HPP
#define HZ_HPP

#include "defines.hpp"

#ifndef HZ_MAX_DEVICES
#ifdef USE_MPI
#define HZ_MAX_DEVICES 512u
#else // !USE_MPI
#define HZ_MAX_DEVICES 1u
#endif // ?USE_MPI
#else // HZ_MAX_DEVICES
#error HZ_MAX_DEVICES not definable externally
#endif // !HZ_MAX_DEVICES

#ifndef HZ_MAX_LEVELS
#ifdef USE_MPI
#define HZ_MAX_LEVELS 3u
#else // !USE_MPI
#define HZ_MAX_LEVELS 2u
#endif // ?USE_MPI
#else // HZ_MAX_LEVELS
#error HZ_MAX_LEVELS not definable externally
#endif // !HZ_MAX_LEVELS

EXTERN_C void border_sizes(const size_t gpus, const size_t mF, const size_t mG, const size_t n, size_t &mF_, size_t &mG_, size_t &n_) throw();
EXTERN_C int bdinit(const size_t m, const size_t n, double *const A, const size_t ldA) throw();

#endif // !HZ_HPP
