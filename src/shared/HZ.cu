#include "HZ.hpp"

#include "my_utils.hpp"

void border_sizes(const unsigned gpus, const unsigned mF, const unsigned mG, const unsigned n, unsigned &mF_, unsigned &mG_, unsigned &n_) throw()
{
  if (gpus) {
    // (n % (2*gpus) == 0) && ((n / gpus) % 32 == 0)
    n_ = (dimToMod((dimToMod(n, (gpus << 1u)) / gpus), 32u) * gpus);
    mF_ = dimToMod(((n_ > mF) ? n_ : mF), 64u);
    mG_ = dimToMod(((n_ > mG) ? n_ : mG), 64u);
  }
  else {
    mF_ = mF;
    mG_ = mG;
    n_ = n;
  }
}

int bdinit(const size_t n, const size_t n_, double *const A, const size_t ldA) throw()
{
  if (!n)
    return -1;
  if (!n_)
    return -2;
  if (n_ < n)
    return -2;
  if (!A)
    return -3;
  if (!ldA)
    return -4;
  if (ldA < n_)
    return -4;

  for (size_t j = n; j < n_; ++j)
    A[j * ldA + j] = 1.0;

  return 0;
}
