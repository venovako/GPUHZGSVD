// defines.hpp: macro definitions and system header includes.

#ifndef DEFINES_HPP
#define DEFINES_HPP

#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* !_GNU_SOURCE */
#endif /* !_WIN32 */

#ifdef __INTEL_COMPILER
#include <mathimf.h>
#else /* NVCC host compiler */
#ifdef __cplusplus
#include <cmath>
#include <complex>
#else /* C */
#include <math.h>
#include <complex.h>
#endif /* ?__cplusplus */
#endif /* ?__INTEL_COMPILER */

#ifdef __cplusplus
#include <cassert>
#include <cerrno>
#include <cctype>
#include <cfloat>
#include <climits>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#else /* C */
#include <assert.h>
#include <errno.h>
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#endif /* ?__cplusplus */

#ifdef _WIN32
#include <io.h>
#include <sys/timeb.h>
#else /* POSIX */
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif /* ?_WIN32 */

// defines

#ifndef EXTERN_C
#ifdef __cplusplus
#define EXTERN_C extern "C"
#else /* C */
#define EXTERN_C extern
#endif /* ?__cplusplus */
#else /* EXTERN_C */
#error EXTERN_C not definable externally
#endif /* ?EXTERN_C */

#ifdef USE_COMPLEX
typedef double cuD;
typedef double cuJ;
#endif /* USE_COMPLEX */

#endif /* !DEFINES_HPP */
