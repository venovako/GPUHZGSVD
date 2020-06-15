#ifndef DEVICE_CODE_PROF_HPP
#define DEVICE_CODE_PROF_HPP

#ifndef OTHER_PC
#define OTHER_PC 0
#endif /* !OTHER_PC */

#ifndef DADD_PC
#define DADD_PC 1
#endif /* !DADD_PC */

#ifndef DSUB_PC
#define DSUB_PC 2
#endif /* !DSUB_PC */

#ifndef DMUL_PC
#define DMUL_PC 3
#endif /* !DMUL_PC */

#ifndef FMA_PC
#define FMA_PC 4
#endif /* !FMA_PC */

#ifndef DDIV_PC
#define DDIV_PC 5
#endif /* !DDIV_PC */

#ifndef DRCP_PC
#define DRCP_PC 6
#endif /* !DRCP_PC */

#ifndef DSQRT_PC
#define DSQRT_PC 7
#endif /* !DSQRT_PC */

#if (defined(PROFILE) && (PROFILE > 0))
#ifdef _dadd_rn
#error _dadd_rn already defined
#else /* !_dadd_rn */
#define _dadd_rn(x,y) (__prof_trigger(DADD_PC), __dadd_rn((x),(y)))
#endif /* ?_dadd_rn */
#ifdef _dadd_rz
#error _dadd_rz already defined
#else /* !_dadd_rz */
#define _dadd_rz(x,y) (__prof_trigger(DADD_PC), __dadd_rz((x),(y)))
#endif /* ?_dadd_rz */
#ifdef _dadd_ru
#error _dadd_ru already defined
#else /* !_dadd_ru */
#define _dadd_ru(x,y) (__prof_trigger(DADD_PC), __dadd_ru((x),(y)))
#endif /* ?_dadd_ru */
#ifdef _dadd_rd
#error _dadd_rd already defined
#else /* ?_dadd_rd */
#define _dadd_rd(x,y) (__prof_trigger(DADD_PC), __dadd_rd((x),(y)))
#endif /* !_dadd_rd */
#ifdef _dsub_rn
#error _dsub_rn already defined
#else /* !_dsub_rn */
#define _dsub_rn(x,y) (__prof_trigger(DSUB_PC), __dsub_rn((x),(y)))
#endif /* ?_dsub_rn */
#ifdef _dsub_rz
#error _dsub_rz already defined
#else /* !_dsub_rz */
#define _dsub_rz(x,y) (__prof_trigger(DSUB_PC), __dsub_rz((x),(y)))
#endif /* ?_dsub_rz */
#ifdef _dsub_ru
#error _dsub_ru already defined
#else /* !_dsub_ru */
#define _dsub_ru(x,y) (__prof_trigger(DSUB_PC), __dsub_ru((x),(y)))
#endif /* ?_dsub_ru */
#ifdef _dsub_rd
#error _dsub_rd already defined
#else /* ?_dsub_rd */
#define _dsub_rd(x,y) (__prof_trigger(DSUB_PC), __dsub_rd((x),(y)))
#endif /* !_dsub_rd */
#ifdef _dmul_rn
#error _dmul_rn already defined
#else /* !_dmul_rn */
#define _dmul_rn(x,y) (__prof_trigger(DMUL_PC), __dmul_rn((x),(y)))
#endif /* ?_dmul_rn */
#ifdef _dmul_rz
#error _dmul_rz already defined
#else /* !_dmul_rz */
#define _dmul_rz(x,y) (__prof_trigger(DMUL_PC), __dmul_rz((x),(y)))
#endif /* ?_dmul_rz */
#ifdef _dmul_ru
#error _dmul_ru already defined
#else /* !_dmul_ru */
#define _dmul_ru(x,y) (__prof_trigger(DMUL_PC), __dmul_ru((x),(y)))
#endif /* ?_dmul_ru */
#ifdef _dmul_rd
#error _dmul_rd already defined
#else /* ?_dmul_rd */
#define _dmul_rd(x,y) (__prof_trigger(DMUL_PC), __dmul_rd((x),(y)))
#endif /* !_dmul_rd */
#ifdef _fma_rn
#error _fma_rn already defined
#else /* !_fma_rn */
#define _fma_rn(x,y,z) (__prof_trigger(FMA_PC), __fma_rn((x),(y),(z)))
#endif /* ?_fma_rn */
#ifdef _fma_rz
#error _fma_rz already defined
#else /* !_fma_rz */
#define _fma_rz(x,y,z) (__prof_trigger(FMA_PC), __fma_rz((x),(y),(z)))
#endif /* ?_fma_rz */
#ifdef _fma_ru
#error _fma_ru already defined
#else /* !_fma_ru */
#define _fma_ru(x,y,z) (__prof_trigger(FMA_PC), __fma_ru((x),(y),(z)))
#endif /* ?_fma_ru */
#ifdef _fma_rd
#error _fma_rd already defined
#else /* ?_fma_rd */
#define _fma_rd(x,y,z) (__prof_trigger(FMA_PC), __fma_rd((x),(y),(z)))
#endif /* !_fma_rd */
#ifdef _ddiv_rn
#error _ddiv_rn already defined
#else /* !_ddiv_rn */
#define _ddiv_rn(x,y) (__prof_trigger(DDIV_PC), __ddiv_rn((x),(y)))
#endif /* ?_ddiv_rn */
#ifdef _ddiv_rz
#error _ddiv_rz already defined
#else /* !_ddiv_rz */
#define _ddiv_rz(x,y) (__prof_trigger(DDIV_PC), __ddiv_rz((x),(y)))
#endif /* ?_ddiv_rz */
#ifdef _ddiv_ru
#error _ddiv_ru already defined
#else /* !_ddiv_ru */
#define _ddiv_ru(x,y) (__prof_trigger(DDIV_PC), __ddiv_ru((x),(y)))
#endif /* ?_ddiv_ru */
#ifdef _ddiv_rd
#error _ddiv_rd already defined
#else /* !_ddiv_rd */
#define _ddiv_rd(x,y) (__prof_trigger(DDIV_PC), __ddiv_rd((x),(y)))
#endif /* ?_ddiv_rd */
#ifdef _drcp_rn
#error _drcp_rn already defined
#else /* !_drcp_rn */
#define _drcp_rn(x) (__prof_trigger(DRCP_PC), __drcp_rn(x))
#endif /* ?_drcp_rn */
#ifdef _drcp_rz
#error _drcp_rz already defined
#else /* !_drcp_rz */
#define _drcp_rz(x) (__prof_trigger(DRCP_PC), __drcp_rz(x))
#endif /* ?_drcp_rz */
#ifdef _drcp_ru
#error _drcp_ru already defined
#else /* !_drcp_ru */
#define _drcp_ru(x) (__prof_trigger(DRCP_PC), __drcp_ru(x))
#endif /* ?_drcp_ru */
#ifdef _drcp_rd
#error _drcp_rd already defined
#else /* !_drcp_rd */
#define _drcp_rd(x) (__prof_trigger(DRCP_PC), __drcp_rd(x))
#endif /* ?_drcp_rd */
#ifdef _dsqrt_rn
#error _dsqrt_rn already defined
#else /* !_dsqrt_rn */
#define _dsqrt_rn(x) (__prof_trigger(DSQRT_PC), __dsqrt_rn(x))
#endif /* ?_dsqrt_rn */
#ifdef _dsqrt_rz
#error _dsqrt_rz already defined
#else /* !_dsqrt_rz */
#define _dsqrt_rz(x) (__prof_trigger(DSQRT_PC), __dsqrt_rz(x))
#endif /* ?_dsqrt_rz */
#ifdef _dsqrt_ru
#error _dsqrt_ru already defined
#else /* !_dsqrt_ru */
#define _dsqrt_ru(x) (__prof_trigger(DSQRT_PC), __dsqrt_ru(x))
#endif /* ?_dsqrt_ru */
#ifdef _dsqrt_rd
#error _dsqrt_rd already defined
#else /* !_dsqrt_rd */
#define _dsqrt_rd(x) (__prof_trigger(DSQRT_PC), __dsqrt_rd(x))
#endif /* ?_dsqrt_rd */
#ifdef _drsqrt_rn
#error _drsqrt_rn already defined
#else /* !_drsqrt_rn */
#define _drsqrt_rn(x) (__prof_trigger(OTHER_PC), my_drsqrt_rn(x))
#endif /* ?_drsqrt_rn */
#ifdef _hypot
#error _hypot already defined
#else /* !_hypot */
#define _hypot(x,y) (__prof_trigger(OTHER_PC), hypot((x),(y)))
#endif /* ?_hypot */
#ifdef _scalbn
#error _scalbn already defined
#else /* !_scalbn */
#define _scalbn(x,y) (__prof_trigger(DMUL_PC), scalbn((x),(y)))
#endif /* ?_scalbn */
#else /* !PROFILE || PROFILE <= 0 */
#ifdef _dadd_rn
#error _dadd_rn already defined
#else /* !_dadd_rn */
#define _dadd_rn(x,y) __dadd_rn((x),(y))
#endif /* ?_dadd_rn */
#ifdef _dadd_rz
#error _dadd_rz already defined
#else /* !_dadd_rz */
#define _dadd_rz(x,y) __dadd_rz((x),(y))
#endif /* ?_dadd_rz */
#ifdef _dadd_ru
#error _dadd_ru already defined
#else /* !_dadd_ru */
#define _dadd_ru(x,y) __dadd_ru((x),(y))
#endif /* ?_dadd_ru */
#ifdef _dadd_rd
#error _dadd_rd already defined
#else /* ?_dadd_rd */
#define _dadd_rd(x,y) __dadd_rd((x),(y))
#endif /* !_dadd_rd */
#ifdef _dsub_rn
#error _dsub_rn already defined
#else /* !_dsub_rn */
#define _dsub_rn(x,y) __dsub_rn((x),(y))
#endif /* ?_dsub_rn */
#ifdef _dsub_rz
#error _dsub_rz already defined
#else /* !_dsub_rz */
#define _dsub_rz(x,y) __dsub_rz((x),(y))
#endif /* ?_dsub_rz */
#ifdef _dsub_ru
#error _dsub_ru already defined
#else /* !_dsub_ru */
#define _dsub_ru(x,y) __dsub_ru((x),(y))
#endif /* ?_dsub_ru */
#ifdef _dsub_rd
#error _dsub_rd already defined
#else /* ?_dsub_rd */
#define _dsub_rd(x,y) __dsub_rd((x),(y))
#endif /* !_dsub_rd */
#ifdef _dmul_rn
#error _dmul_rn already defined
#else /* !_dmul_rn */
#define _dmul_rn(x,y) __dmul_rn((x),(y))
#endif /* ?_dmul_rn */
#ifdef _dmul_rz
#error _dmul_rz already defined
#else /* !_dmul_rz */
#define _dmul_rz(x,y) __dmul_rz((x),(y))
#endif /* ?_dmul_rz */
#ifdef _dmul_ru
#error _dmul_ru already defined
#else /* !_dmul_ru */
#define _dmul_ru(x,y) __dmul_ru((x),(y))
#endif /* ?_dmul_ru */
#ifdef _dmul_rd
#error _dmul_rd already defined
#else /* ?_dmul_rd */
#define _dmul_rd(x,y) __dmul_rd((x),(y))
#endif /* !_dmul_rd */
#ifdef _fma_rn
#error _fma_rn already defined
#else /* !_fma_rn */
#define _fma_rn(x,y,z) __fma_rn((x),(y),(z))
#endif /* ?_fma_rn */
#ifdef _fma_rz
#error _fma_rz already defined
#else /* !_fma_rz */
#define _fma_rz(x,y,z) __fma_rz((x),(y),(z))
#endif /* ?_fma_rz */
#ifdef _fma_ru
#error _fma_ru already defined
#else /* !_fma_ru */
#define _fma_ru(x,y,z) __fma_ru((x),(y),(z))
#endif /* ?_fma_ru */
#ifdef _fma_rd
#error _fma_rd already defined
#else /* ?_fma_rd */
#define _fma_rd(x,y,z) __fma_rd((x),(y),(z))
#endif /* !_fma_rd */
#ifdef _ddiv_rn
#error _ddiv_rn already defined
#else /* !_ddiv_rn */
#define _ddiv_rn(x,y) __ddiv_rn((x),(y))
#endif /* ?_ddiv_rn */
#ifdef _ddiv_rz
#error _ddiv_rz already defined
#else /* !_ddiv_rz */
#define _ddiv_rz(x,y) __ddiv_rz((x),(y))
#endif /* ?_ddiv_rz */
#ifdef _ddiv_ru
#error _ddiv_ru already defined
#else /* !_ddiv_ru */
#define _ddiv_ru(x,y) __ddiv_ru((x),(y))
#endif /* ?_ddiv_ru */
#ifdef _ddiv_rd
#error _ddiv_rd already defined
#else /* !_ddiv_rd */
#define _ddiv_rd(x,y) __ddiv_rd((x),(y))
#endif /* ?_ddiv_rd */
#ifdef _drcp_rn
#error _drcp_rn already defined
#else /* !_drcp_rn */
#define _drcp_rn(x) __drcp_rn(x)
#endif /* ?_drcp_rn */
#ifdef _drcp_rz
#error _drcp_rz already defined
#else /* !_drcp_rz */
#define _drcp_rz(x) __drcp_rz(x)
#endif /* ?_drcp_rz */
#ifdef _drcp_ru
#error _drcp_ru already defined
#else /* !_drcp_ru */
#define _drcp_ru(x) __drcp_ru(x)
#endif /* ?_drcp_ru */
#ifdef _drcp_rd
#error _drcp_rd already defined
#else /* !_drcp_rd */
#define _drcp_rd(x) __drcp_rd(x)
#endif /* ?_drcp_rd */
#ifdef _dsqrt_rn
#error _dsqrt_rn already defined
#else /* !_dsqrt_rn */
#define _dsqrt_rn(x) __dsqrt_rn(x)
#endif /* ?_dsqrt_rn */
#ifdef _dsqrt_rz
#error _dsqrt_rz already defined
#else /* !_dsqrt_rz */
#define _dsqrt_rz(x) __dsqrt_rz(x)
#endif /* ?_dsqrt_rz */
#ifdef _dsqrt_ru
#error _dsqrt_ru already defined
#else /* !_dsqrt_ru */
#define _dsqrt_ru(x) __dsqrt_ru(x)
#endif /* ?_dsqrt_ru */
#ifdef _dsqrt_rd
#error _dsqrt_rd already defined
#else /* !_dsqrt_rd */
#define _dsqrt_rd(x) __dsqrt_rd(x)
#endif /* ?_dsqrt_rd */
#ifdef _drsqrt_rn
#error _drsqrt_rn already defined
#else /* !_drsqrt_rn */
#define _drsqrt_rn(x) my_drsqrt_rn(x)
#endif /* ?_drsqrt_rn */
#ifdef _hypot
#error _hypot already defined
#else /* !_hypot */
#define _hypot(x,y) hypot((x),(y))
#endif /* ?_hypot */
#ifdef _scalbn
#error _scalbn already defined
#else /* !_scalbn */
#define _scalbn(x,y) scalbn((x),(y))
#endif /* ?_scalbn */
#endif /* ?PROFILE */

#endif /* !DEVICE_CODE_PROF_HPP */
