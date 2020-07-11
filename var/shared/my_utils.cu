#include "my_utils.hpp"

char err_msg[err_msg_size] = { '\0' };

unsigned long long atou(const char *const s) throw()
{
  if (!s) {
    DIE("atou(NULL)");
  }
  const char *s_ = s;
  while (*s_) {
    if (isspace(*s_)) {
      ++s_;
    }
    else if ('-' == *s_) {
      DIE("atou(-)");
    }
    else if ('+' == *s_) {
      break;
    }
    else if (isdigit(*s_)) {
      break;
    }
    else {
      DIE("atou(\?)");
    }
  }
  if (!*s_) {
    DIE("atou(\"\")");
  }
  char *e = static_cast<char*>(NULL);
  const unsigned long long u = strtoull(s_, &e, 0);
  if (e && *e) {
    DIE("atou(!)");
  }
  return u;
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
