#include "HZ_L.hpp"

#include "my_utils.hpp"

unsigned STRAT0 = 0u, STRAT0_STEPS = 0u, STRAT0_PAIRS = 0u;
unsigned STRAT1 = 0u, STRAT1_STEPS = 0u, STRAT1_PAIRS = 0u;

unsigned STRAT0_DTYPE strat0[STRAT0_MAX_STEPS][STRAT0_MAX_PAIRS][2u];
unsigned STRAT1_DTYPE strat1[STRAT1_MAX_STEPS][STRAT1_MAX_PAIRS][2u];

jstrat_common js0, js1;

void init_strats(const unsigned snp0, const unsigned n0, const unsigned snp1, const unsigned n1) throw()
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
  if (STRAT0_STEPS > STRAT0_MAX_STEPS) {
    DIE("STRAT0_STEPS > STRAT0_MAX_STEPS");
  }

  STRAT0_PAIRS = (n0 >> 1u);
  if (STRAT0_PAIRS > STRAT0_MAX_PAIRS) {
    DIE("STRAT0_PAIRS > STRAT0_MAX_PAIRS");
  }
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
  if (STRAT1_STEPS > STRAT1_MAX_STEPS) {
    DIE("STRAT1_STEPS > STRAT1_MAX_STEPS");
  }

  STRAT1_PAIRS = (n1 >> 1u);
  if (STRAT1_PAIRS > STRAT1_MAX_PAIRS) {
    DIE("STRAT1_PAIRS > STRAT1_MAX_PAIRS");
  }
  (void)memset(strat1, 0, sizeof(strat1));

  const unsigned ap = ((n1 >= n0) ? n1 : n0);
  integer (*const arr)[2u] = (integer (*)[2u])malloc(ap * sizeof(integer));
  if (!arr) {
    DIE("arr out of memory");
  }
  int (*const arri)[2u][2u] = (int (*)[2u][2u])arr;

  if (STRAT0_STEPS != jstrat_init(&js0, static_cast<integer>(STRAT0), static_cast<integer>(n0))) {
    DIE("STRAT0 init");
  }

  for (unsigned s = 0u; s < STRAT0_STEPS; ++s) {
    if (STRAT0_PAIRS != static_cast<unsigned>(iabs(jstrat_next(&js0, (integer*)arr)))) {
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
    if (STRAT1_PAIRS != static_cast<unsigned>(iabs(jstrat_next(&js1, (integer*)arr)))) {
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

  free(arr);
}

void free_strats() throw()
{
  jstrat_free(&js1);
  jstrat_free(&js0);
}
