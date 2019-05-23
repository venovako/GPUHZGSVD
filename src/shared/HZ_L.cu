#include "HZ_L.hpp"

#include "my_utils.hpp"

unsigned STRAT0 = 0u, STRAT0_STEPS = 0u, STRAT0_PAIRS = 0u;
unsigned STRAT1 = 0u, STRAT1_STEPS = 0u, STRAT1_PAIRS = 0u;

unsigned STRAT0_DTYPE strat0[STRAT0_MAX_STEPS][STRAT0_MAX_PAIRS][2u];
unsigned STRAT1_DTYPE strat1[STRAT1_MAX_STEPS][STRAT1_MAX_PAIRS][2u];

jstrat_common js0, js1;

#ifdef USE_MPI
unsigned STRAT2 = 0u, STRAT2_STEPS = 0u, STRAT2_PAIRS = 0u;
STRAT2_DTYPE strat2[STRAT2_MAX_STEPS][STRAT2_MAX_PAIRS][2u][2u];
jstrat_common js2;

void init_strats(const unsigned snp0, const unsigned n0, const unsigned snp1, const unsigned n1, const unsigned snp2, const unsigned n2) throw()
#else // !USE_MPI
void init_strats(const unsigned snp0, const unsigned n0, const unsigned snp1, const unsigned n1) throw()
#endif // ?USE_MPI
{
  switch (snp0) {
  case STRAT_CYCWOR:
    STRAT0 = snp0;
    STRAT0_STEPS = n0 - 1u;
    break;
  case STRAT_MMSTEP:
    STRAT0 = snp0;
    STRAT0_STEPS = n0;
    break;
  default:
    DIE("SNP0 \\notin { 2, 4 }");
  }

  STRAT0_PAIRS = (n0 >> 1u);
  (void)memset(strat0, 0, sizeof(strat0));

  switch (snp1) {
  case STRAT_CYCWOR:
    STRAT1 = snp1;
    STRAT1_STEPS = n1 - 1u;
    break;
  case STRAT_MMSTEP:
    STRAT1 = snp1;
    STRAT1_STEPS = n1;
    break;
  default:
    DIE("SNP1 \\notin { 2, 4 }");
  }

  STRAT1_PAIRS = (n1 >> 1u);
  (void)memset(strat1, 0, sizeof(strat1));

  unsigned ap = ((n1 >= n0) ? n1 : n0);

#ifdef USE_MPI
  switch (snp2) {
  case (STRAT_CYCWOR + 1u):
    STRAT2 = snp2;
    STRAT2_STEPS = n2 - 1u;
    break;
  case (STRAT_MMSTEP + 1u):
    STRAT2 = snp2;
    STRAT2_STEPS = n2;
    break;
  default:
    DIE("SNP2 \\notin { 3, 5 }");
  }

  STRAT2_PAIRS = (n2 >> 1u);
  (void)memset(strat2, 0, sizeof(strat2));

  ap = (((ap >= n2) ? ap : n2) << 1u);
#endif // USE_MPI

  integer (*const arr)[2u] = (integer (*)[2u])malloc(ap * sizeof(integer));
  if (!arr) {
    DIE("arr out of memory");
  }
  int (*const arri)[2u][2u] = (int (*)[2u][2u])arr;

  if (STRAT0_STEPS != jstrat_init(&js0, static_cast<integer>(STRAT0), static_cast<integer>(n0))) {
    DIE("STRAT0 init");
  }

  for (unsigned s = 0u; s < STRAT0_STEPS; ++s) {
    if (STRAT0_PAIRS != jstrat_next(&js0, (integer*)arr)) {
      DIE("STRAT0 next");
    }
    for (unsigned p = 0u; p < STRAT0_PAIRS; ++p) {
      for (unsigned i = 0u; i < 2u; ++i) {
        switch (STRAT0) {
        case STRAT_CYCWOR:
          strat0[s][p][i] = static_cast<unsigned STRAT0_DTYPE>(arr[p][i]);
          break;
        case STRAT_MMSTEP:
          strat0[s][p][i] = static_cast<unsigned STRAT0_DTYPE>(arri[p][i][0u]);
          break;
        }
      }
    }
  }

  if (STRAT1_STEPS != jstrat_init(&js1, static_cast<integer>(STRAT1), static_cast<integer>(n1))) {
    DIE("STRAT1 init");
  }

  for (unsigned s = 0u; s < STRAT1_STEPS; ++s) {
    if (STRAT1_PAIRS != jstrat_next(&js1, (integer*)arr)) {
      DIE("STRAT1 next");
    }
    for (unsigned p = 0u; p < STRAT1_PAIRS; ++p) {
      for (unsigned i = 0u; i < 2u; ++i) {
        switch (STRAT1) {
        case STRAT_CYCWOR:
          strat1[s][p][i] = static_cast<unsigned STRAT1_DTYPE>(arr[p][i]);
          break;
        case STRAT_MMSTEP:
          strat1[s][p][i] = static_cast<unsigned STRAT1_DTYPE>(arri[p][i][0u]);
          break;
        }
      }
    }
  }

#ifdef USE_MPI
  if (STRAT2_STEPS != jstrat_init(&js2, static_cast<integer>(STRAT2), static_cast<integer>(n2))) {
    DIE("STRAT2 init");
  }

  integer (*const a)[2u][2u] = (integer (*)[2u][2u])arr;
  int (*const ai)[2u][2u][2u] = (int (*)[2u][2u][2u])a;

  for (unsigned s = 0u; s < STRAT2_STEPS; ++s) {
    if (STRAT2_PAIRS != jstrat_next(&js2, (integer*)a)) {
      DIE("STRAT2 next");
    }
    for (unsigned p = 0u; p < STRAT2_PAIRS; ++p) {
      for (unsigned c = 0u; c < 2u; ++c) {
        for (unsigned i = 0u; i < 2u; ++i) {
          switch (STRAT2) {
          case STRAT_CYCWOR:
            strat2[s][p][c][i] = static_cast<STRAT2_DTYPE>(a[p][c][i]);
            break;
          case STRAT_MMSTEP:
            strat2[s][p][c][i] = static_cast<STRAT2_DTYPE>(ai[p][c][i][0u]);
            break;
          }
        }
      }
    }
  }

  if (!mpi_rank)
    (void)fprintf(stderr, "\nSTRAT2 & COMM PATTERN\n");
  for (unsigned s = 0u; s < STRAT2_STEPS; ++s) {
    if (!mpi_rank)
      (void)fprintf(stderr, "%u: ", s);
    for (unsigned p = 0u; p < STRAT2_PAIRS; ++p) {
      if (!mpi_rank)
        (void)fprintf
          (stderr, "(%u%c%d,%u%c%d)",
           static_cast<unsigned>(strat2[s][p][0u][0u]), ((strat2[s][p][1u][0u] < 0) ? 'L' : 'R'), abs(strat2[s][p][1u][0u])-1,
           static_cast<unsigned>(strat2[s][p][0u][1u]), ((strat2[s][p][1u][1u] < 0) ? 'L' : 'R'), abs(strat2[s][p][1u][1u])-1);
      if (!mpi_rank)
        (void)fprintf(stderr, "%c", ((p == (STRAT2_PAIRS - 1u)) ? '\n' : ','));
    }
  }
#endif // USE_MPI

  free(arr);
}

void free_strats() throw()
{
#ifdef USE_MPI
  jstrat_free(&js2);
#endif // USE_MPI
  jstrat_free(&js1);
  jstrat_free(&js0);
}
