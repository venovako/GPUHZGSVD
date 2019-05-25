#include "HZ.hpp"

#include "my_utils.hpp"

void border_sizes(const size_t gpus, const size_t mF, const size_t mG, const size_t n, size_t &mF_, size_t &mG_, size_t &n_) throw()
{
  if (gpus) {
    // (n % (2*gpus) == 0) && ((n / gpus) % 32 == 0)
    n_ = (dimToMod((dimToMod(n, (gpus << 1u)) / gpus), static_cast<size_t>(32u)) * gpus);
    mF_ = dimToMod(((n_ > mF) ? n_ : mF), static_cast<size_t>(64u));
    mG_ = dimToMod(((n_ > mG) ? n_ : mG), static_cast<size_t>(64u));
  }
  else {
    mF_ = mF;
    mG_ = mG;
    n_ = n;
  }
}

int bdinit(const size_t m, const size_t n, double *const A, const size_t ldA) throw()
{
  if (!m)
    return 0;
  if (!n)
    return 0;
  if (!A)
    return -3;
  if (ldA < (m + n))
    return -4;

  for (size_t j = 0u; j < n; ++j) {
    const size_t i = m + j;
    A[j * ldA + i] = 1.0;
  }

  return 0;
}
