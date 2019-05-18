// defines.hpp: macro definitions and system header includes.

#ifndef DEFINES_HPP
#define DEFINES_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // !_GNU_SOURCE

#ifdef __INTEL_COMPILER
#include <mathimf.h>
#else // NVCC host compiler
#ifdef __cplusplus
#include <cmath>
#include <complex>
#else // C99
#include <math.h>
#include <complex.h>
#endif // __cplusplus
#endif // __INTEL_COMPILER

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
#else // C
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
#endif // __cplusplus

#include <alloca.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// defines

#ifndef EXTERN_C
#ifdef __cplusplus
#define EXTERN_C extern "C"
#else // C
#define EXTERN_C extern
#endif // __cplusplus
#else // EXTERN_C
#error EXTERN_C not definable externally
#endif // !EXTERN_C

#ifdef VAR_UNUSED
#error VAR_UNUSED not definable externally
#endif // VAR_UNUSED
#define VAR_UNUSED __attribute__ ((unused))

#ifdef USE_COMPLEX
typedef double cuD;
typedef double cuJ;
#endif // USE_COMPLEX

#endif // !DEFINES_HPP
