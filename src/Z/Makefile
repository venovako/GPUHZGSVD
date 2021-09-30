RM=del /F

!IFNDEF NVCC
NVCC=nvcc.exe
!ENDIF # !NVCC

!IFNDEF CVG
CVG=0
!ENDIF # !CVG

!IFDEF HCVG
!ERROR HCVG defined externally
!ENDIF # HCVG

!IF (($(CVG)==0)||($(CVG)==1))
HCVG=0
!ELSE IF (($(CVG)==2)||($(CVG)==3))
HCVG=1
!ELSE IF (($(CVG)==4)||($(CVG)==5))
HCVG=2
!ELSE IF (($(CVG)==6)||($(CVG)==7))
HCVG=3
!ELSE # not supported
!ERROR unsupported CVG
!ENDIF # ?CVG

!IFNDEF ARCH
ARCH=35 -Wno-deprecated-gpu-targets
!ENDIF # !ARCH

NVPPFLAGS=-I. -I..\shared -I..\..\..\JACSD\jstrat -DCVG=$(CVG) -DUSE_COMPLEX -DNDEBUG
NVCCFLAGS=-std c++17 -restrict -res-usage -arch sm_$(ARCH) -extra-device-vectorization
NVLDFLAGS=-L..\..\..\JACSD -ljstrat

HEADERS=				         \
	..\shared\HZ.hpp		         \
	..\shared\HZ_L.hpp		         \
	..\shared\cuda_helper.hpp	         \
	..\shared\cuda_memory_helper.hpp         \
	..\shared\defines.hpp		         \
	..\shared\device_code_common_defs.hpp    \
	..\shared\device_code_common_globals.hpp \
	..\shared\device_code_prof.hpp           \
	..\shared\my_utils.hpp		         \
	HZ_L2.hpp			         \
	cuZ.hpp                                  \
	device_code.hpp			         \
	device_code_accumV.hpp 		         \
	device_code_cdsort_$(HCVG).hpp	         \
	device_code_nosort_$(HCVG).hpp	         \
	device_code_cdsort_accumV.hpp	         \
	device_code_nosort_accumV.hpp	         \
	device_code_common.hpp		         \
	device_code_globals.hpp		         \
	device_code_common_rotate.hpp	         \
	device_code_common_Kepler.hpp	         \
	device_code_common_Cholesky.hpp

SOURCES=			 \
	..\shared\HZ.cu		 \
	..\shared\HZ_L.cu	 \
	..\shared\cuda_helper.cu \
	..\shared\main.cu        \
	..\shared\my_utils.cu	 \
	HZ_L2.cu		 \
	device_code.cu

EXE=HZ$(CVG).exe
all: $(EXE)

HZ$(CVG).exe: $(HEADERS) $(SOURCES) Makefile
	$(NVCC) $(NVPPFLAGS) $(NVCCFLAGS) $(SOURCES) -o $@ $(NVLDFLAGS)

clean:
	-$(RM) $(EXE)
