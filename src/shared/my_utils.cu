#include "my_utils.hpp"

#include <dlfcn.h>

char err_msg[err_msg_size] = { '\0' };

int fexist(const char *const fn) throw()
{
  struct stat buf;
  return (fn ? (0 == stat(fn, &buf)) : 0);
}

void *strat_open(const char *const sdy) throw()
{
  return dlopen(sdy, RTLD_LAZY);
}

int strat_close(void *const h) throw()
{
  return dlclose(h);
}

const void *strat_ptr(void *const h, const char *const snp, const unsigned n) throw()
{
  char arrn[12] = { '\0' };
  (void)snprintf(arrn, sizeof(arrn), "%s%05u", snp, n);
  return (arrn[11] ? NULL : dlsym(h, arrn));
}

long long timestamp() throw()
{
  struct timeval tv;
  SYSI_CALL(gettimeofday(&tv, static_cast<struct timezone*>(NULL)));
  return (tv.tv_sec * TS_S + tv.tv_usec);
}

void stopwatch_reset(long long &sw) throw()
{
  sw = timestamp();
}

long long stopwatch_lap(long long &sw) throw()
{
  const long long
    ts = timestamp(),
    lap = ts - sw;
  sw = ts;
  return lap;
}

int border1sz(const unsigned mF, const unsigned mG, const unsigned n, unsigned &mF_, unsigned &mG_, unsigned &n_) throw()
{
  if (!mF)
    return -1;
  if (!mG)
    return -2;
  if (!n)
    return -3;
  if (n > mF)
    return -3;
  if (n > mG)
    return -3;

  mF_ = dimToMod(mF, 64u);
  mG_ = dimToMod(mG, 64u);
  n_ = dimToMod(n, 32u);

  return 0;
}
