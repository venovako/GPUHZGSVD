#include "HZ.hpp"

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
