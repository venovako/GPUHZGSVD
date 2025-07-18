RM=rm -rf

ifndef NVCC
NVCC=nvcc
endif # !NVCC

ifdef NDEBUG
DEBUG=
else # DEBUG
DEBUG=g
endif # ?NDEBUG

NVPPFLAGS=-I. -I../shared -I../../../JACSD/jstrat
ifdef PROFILE
NVPPFLAGS+=-DPROFILE=$(PROFILE)
endif # PROFILE
ifndef MARCH
MARCH=native
endif # !MARCH
NVCCFLAGS=-std c++20 -restrict -res-usage -Wno-deprecated-gpu-targets -Xcompiler -march=$(MARCH) -Xcompiler -pthread # -Xcompiler -fopenmp
ifdef GPU_ARCH
GENCODE=-gencode arch=compute_$(ARCH),code=sm_$(ARCH)
NVCCFLAGS+=$(foreach ARCH,$(GPU_ARCH),$(GENCODE))
endif # GPU_ARCH
NVLDFLAGS=-l cublas,dl,m

ifdef NDEBUG
NVPPFLAGS+=-DNDEBUG
NVCCFLAGS+=-O$(NDEBUG) --extra-device-vectorization
else # debug
NVCCFLAGS+=-O0 -$(DEBUG) -G -src-in-ptx -Xptxas -no-bb-merge,-ret-end
endif # NDEBUG

HEADERS=				         \
	../shared/HZ.hpp		         \
	../shared/HZ_L.hpp		         \
	../shared/cuda_helper.hpp	         \
	../shared/cuda_memory_helper.hpp         \
	../shared/defines.hpp		         \
	../shared/device_code_common_defs.hpp    \
	../shared/device_code_common_globals.hpp \
	../shared/device_code_prof.hpp           \
	../shared/my_utils.hpp		         \
	HZ_L2.hpp			         \
	device_code.hpp			         \
	device_code_cdsort_0.hpp	         \
	device_code_cdsort_accumV.hpp	         \
	device_code_common.hpp		         \
	device_code_common_rotate.hpp	         \
	device_code_common_Kepler.hpp	         \
	device_code_common_Cholesky.hpp

SOURCES=			 \
	../shared/HZ.cu		 \
	../shared/HZ_L.cu	 \
	../shared/cuda_helper.cu \
	../shared/main.cu        \
	../shared/my_utils.cu	 \
	HZ_L2.cu		 \
	device_code.cu		 \
	../../../JACSD/jstrat/jstrat.c

EXE=HZ0.exe

all: $(EXE)

HZ0.exe: $(HEADERS) $(SOURCES) GNUmakefile
	$(NVCC) $(NVPPFLAGS) $(NVCCFLAGS) $(SOURCES) -o $@ $(NVLDFLAGS)

clean:
	-$(RM) $(EXE)
