RM=del /F

!IFNDEF NVCC
NVCC=nvcc.exe
!ENDIF # !NVCC

!IFNDEF CVG
CVG=0
!ENDIF # !CVG

!IFNDEF ARCH
ARCH=native
!ENDIF # !ARCH

NVPPFLAGS=-I. -I..\shared -I..\..\..\JACSD\jstrat -DCVG=$(CVG) -DNDEBUG
NVCCFLAGS=-std c++20 -restrict -res-usage -arch $(ARCH) -extra-device-vectorization -Wno-deprecated-gpu-targets
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
	device_code.hpp			         \
	device_code_accumV.hpp 		         \
	device_code_cdsort_$(CVG).hpp	         \
	device_code_cdsort_accumV.hpp	         \
	device_code_common.hpp		         \
	device_code_globals.hpp		         \
	device_code_common_rotate.hpp	         \
	device_code_common_Kepler.hpp	         \
	device_code_common_Cholesky.hpp          \
	device_code_common_QR.hpp

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
