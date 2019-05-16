RM=rm -rf

ifndef NVCC
NVCC=nvcc
endif # !NVCC

ifndef GPU_ARCH
GPU_ARCH=30
endif # !GPU_ARCH

ifndef CVG
CVG=0
endif # !CVG

ifndef STD
STD=11
endif # !STD

NVPPFLAGS=-I. -I../shared -DCVG=$(CVG) -DUSE_COMPLEX
NVCCFLAGS=-std c++$(STD) -arch sm_$(GPU_ARCH) -restrict -res-usage -Xcompiler -march=native
NVLDFLAGS=

ifdef NDEBUG
NVPPFLAGS+=-DNDEBUG
NVCCFLAGS+=-O$(NDEBUG)
else # debug
NVPPFLAGS+=-UNDEBUG
NVCCFLAGS+=-O0 -g -G -src-in-ptx -Xptxas -no-bb-merge,-ret-end
endif # NDEBUG

# e.g. -ffp-contract=on,-integrated-as
ifdef HOST_FLAGS
NVCCFLAGS+=-Xcompiler $(HOST_FLAGS)
endif # HOST_FLAGS

ifdef ANIMATE
NVPPFLAGS+=-DUSE_MKL -DANIMATE=$(ANIMATE) -I../../../JACSD/vn -I${MKLROOT}/include/intel64/ilp64 -I${MKLROOT}/include
NVLDFLAGS+=-L../../../JACSD -lvn -L${MKLROOT}/lib -Xlinker -rpath,${MKLROOT}/lib -L${MKLROOT}/../compiler/lib -Xlinker -rpath,${MKLROOT}/../compiler/lib -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread
endif # ANIMATE

NVLDFLAGS+=-l dl,m

HEADERS=				\
	../shared/HZ.hpp		\
	../shared/HZ_L.hpp		\
	HZ_L2.hpp			\
	cuZ.hpp                         \
	../shared/cuda_helper.hpp	\
	../shared/cuda_memory_helper.hpp\
	../shared/defines.hpp		\
	device_code.hpp			\
	device_code_accumV.hpp 		\
	device_code_cdsort_$(CVG).hpp	\
	device_code_cdsort_accumV.hpp	\
	device_code_common.hpp		\
	device_code_globals.hpp		\
	device_code_common_rotate.hpp	\
	device_code_common_Kepler.hpp	\
	device_code_common_Cholesky.hpp	\
	../shared/my_utils.hpp

SOURCES=			\
	../shared/HZ_L.cu	\
	HZ_L2.cu		\
	../shared/cuda_helper.cu\
	device_code.cu		\
	../shared/main.cu	\
	../shared/my_utils.cu

EXE=HZ$(CVG).exe

all: $(EXE)

$(EXE): $(HEADERS) $(SOURCES) GNUmakefile
	$(NVCC) $(NVPPFLAGS) $(NVCCFLAGS) $(SOURCES) -o $@ $(NVLDFLAGS)

clean:
	-$(RM) $(EXE)
	-$(RM) $(EXE).dSYM
