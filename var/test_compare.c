#include <defines.hpp>

static int dd(const double a[static 1], const double b[static 1])
{
  if (a == b)
    return 0;
  if (*a < *b)
    return 1;
  if (*a > *b)
    return -1;
  return 0;
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    (void)fprintf(stderr, "%s n fn\n", *argv);
    return EXIT_FAILURE;
  }

  const unsigned n = (unsigned)atoi(argv[1]);
  if (!n)
    return EXIT_FAILURE;
  char *const fn = (char*)calloc(strlen(argv[2]) + 4u, sizeof(char));
  if (!fn)
    return EXIT_FAILURE;

  FILE *const s = fopen(strcat(strcpy(fn, argv[2]), ".S"), "rb");
  if (!s)
    return EXIT_FAILURE;
  double *const se = (double*)calloc(n, sizeof(double));
  if (!se)
    return EXIT_FAILURE;
  if (fread(se, sizeof(double), n, s) != n)
    return EXIT_FAILURE;
  if (fclose(s))
    return EXIT_FAILURE;
  qsort(se, n, sizeof(double), (int (*)(const void*, const void*))dd);

  FILE *const ss = fopen(strcat(strcpy(fn, argv[2]), ".SS"), "rb");
  if (!ss)
    return EXIT_FAILURE;
  double *const sv = (double*)calloc(n, sizeof(double));
  if (!sv)
    return EXIT_FAILURE;
  if (fread(sv, sizeof(double), n, ss) != n)
    return EXIT_FAILURE;
  if (fclose(ss))
    return EXIT_FAILURE;
  qsort(sv, n, sizeof(double), (int (*)(const void*, const void*))dd);

  FILE *const l = fopen(strcat(strcpy(fn, argv[2]), ".L"), "rb");
  if (!l)
    return EXIT_FAILURE;
  double *const ev = (double*)calloc(n, sizeof(double));
  if (!ev)
    return EXIT_FAILURE;
  if (fread(ev, sizeof(double), n, l) != n)
    return EXIT_FAILURE;
  if (fclose(l))
    return EXIT_FAILURE;
  qsort(ev, n, sizeof(double), (int (*)(const void*, const void*))dd);

  long double MreS = 0.0, MreE = 0.0;
  (void)fprintf(stdout, "\"IDX\",\"GSVD\",\"GESV\",\"RESV\",\"GEEV\",\"REEV\"\n");
  (void)fflush(stdout);
  for (unsigned i = 0u; i < n; ++i) {
    const long double lse = (long double)(se[i]);
    const long double lsv = (long double)(sv[i]);
    const long double lev = (long double)(ev[i]);
    (void)fprintf(stdout, "%4u,%# .17Le,", i, lse);

    (void)fprintf(stdout, "%# .17Le,", lsv);
    long double reS = ((lsv - lse) / lse);
    (void)fprintf(stdout, "%# .17Le,", reS);
    reS = fabsl(reS);
    if (reS > MreS)
      MreS = reS;

    (void)fprintf(stdout, "%# .17Le,", i, lev);
    long double reE = ((sqrtl(lev) - lse) / lse);
    (void)fprintf(stdout, "%# .17Le\n", i, reE);
    reE = fabsl(reE);
    if ((reE > MreE) || isnanl(reE))
      MreE = reE;

    (void)fflush(stdout);
  }

  (void)fprintf(stdout, "%4u,%# .17Le,%# .17Le,%# .17Le,%# .17Le,%# .17Le\n", n, -0.0, -0.0, MreS, -0.0, MreE);
  (void)fflush(stdout);

  free(ev);
  free(sv);
  free(se);
  return EXIT_SUCCESS;
}
