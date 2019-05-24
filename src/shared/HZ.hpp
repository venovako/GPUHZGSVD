#ifndef HZ_HPP
#define HZ_HPP

#include "defines.hpp"
#include "my_utils.hpp"

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

template <typename T>
void border_sizes(const T gpus, const T mF, const T mG, const T n, T &mF_, T &mG_, T &n_) throw()
{
  if (gpus) {
    // (n % (2*gpus) == 0) && ((n / gpus) % 32 == 0)
    n_ = (dimToMod((dimToMod(n, (gpus << 1u)) / gpus), static_cast<T>(32u)) * gpus);
    mF_ = dimToMod(((n_ > mF) ? n_ : mF), static_cast<T>(64u));
    mG_ = dimToMod(((n_ > mG) ? n_ : mG), static_cast<T>(64u));
  }
  else {
    mF_ = mF;
    mG_ = mG;
    n_ = n;
  }
}

EXTERN_C int bdinit(const size_t m, const size_t n, double *const A, const size_t ldA) throw();

#endif // !HZ_HPP
