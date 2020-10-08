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
  FILE *const s = fopen(strcat(strcpy(fn, argv[2]), ".SS"), "rb");
  if (!s)
    return EXIT_FAILURE;
  double *const sv = (double*)calloc(n, sizeof(double));
  if (!sv)
    return EXIT_FAILURE;
  if (fread(sv, sizeof(double), n, s) != n)
    return EXIT_FAILURE;
  if (fclose(s))
    return EXIT_FAILURE;
  qsort(sv, n, sizeof(double), (int (*)(const void*, const void*))dd);
  FILE *const e = fopen(strcat(strcpy(fn, argv[2]), ".L"), "rb");
  if (!e)
    return EXIT_FAILURE;
  double *const ev = (double*)calloc(n, sizeof(double));
  if (!ev)
    return EXIT_FAILURE;
  if (fread(ev, sizeof(double), n, e) != n)
    return EXIT_FAILURE;
  if (fclose(e))
    return EXIT_FAILURE;
  qsort(ev, n, sizeof(double), (int (*)(const void*, const void*))dd);
  long double Mre = 0.0;
  (void)fprintf(stdout, "\"IDX\",\"GESV\",\"GEEV\",\"RELERR\"\n");
  for (unsigned i = 0u; i < n; ++i) {
    const long double ls = (long double)(sv[i]);
    const long double le = (long double)(ev[i]);
    (void)fprintf(stdout, "%4u,%# .17Le,%# .17Le,", i, ls, le);
    const long double re = ((sqrtl(le) - ls) / ls);
    (void)fprintf(stdout, "%# .17Le\n", i, re);
    const long double re_ = fabsl(re);
    if (re_ > Mre)
      Mre = re_;
  }
  (void)fprintf(stdout, "%4u,%# .17Le\n", n, Mre);
  free(ev);
  free(sv);
  return EXIT_SUCCESS;
}
