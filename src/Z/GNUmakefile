RM=rm -rf

ifndef NVCC
NVCC=nvcc
endif # !NVCC

ifndef CVG
CVG=0
endif # !CVG
HCVG=$(shell expr $(CVG) / 2)

ifndef STD
STD=11
endif # !STD

ifdef NDEBUG
DEBUG=
else # DEBUG
DEBUG=g
endif # ?NDEBUG

NVPPFLAGS=-I. -I../shared -I../../../JACSD/jstrat -DCVG=$(CVG) -DUSE_COMPLEX
NVCCFLAGS=-std c++$(STD) -restrict -res-usage -Xcompiler -march=native -Xcompiler -pthread
ifdef GPU_ARCH
GENCODE=-gencode arch=compute_$(ARCH),code=sm_$(ARCH)
NVCCFLAGS+=$(foreach ARCH,$(GPU_ARCH),$(GENCODE))
endif # GPU_ARCH
NVLDFLAGS=-L../../../JACSD -ljstrat$(DEBUG)

ifdef NDEBUG
NVPPFLAGS+=-DNDEBUG
NVCCFLAGS+=-O$(NDEBUG)
else # debug
NVCCFLAGS+=-O0 -$(DEBUG) -G -src-in-ptx -Xptxas -no-bb-merge,-ret-end
endif # NDEBUG

# e.g. -ffp-contract=on,-integrated-as
ifdef HOST_FLAGS
NVCCFLAGS+=-Xcompiler $(HOST_FLAGS)
endif # HOST_FLAGS

ifdef ANIMATE
NVPPFLAGS+=-DUSE_MKL -DANIMATE=$(ANIMATE) -I../../../JACSD/vn -I${MKLROOT}/include/intel64/ilp64 -I${MKLROOT}/include
NVLDFLAGS+=-lvn$(DEBUG) -L${MKLROOT}/lib -Xlinker -rpath,${MKLROOT}/lib -L${MKLROOT}/../compiler/lib -Xlinker -rpath,${MKLROOT}/../compiler/lib -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread
endif # ANIMATE

NVLDFLAGS+=-l dl,m

HEADERS=				\
	../shared/HZ.hpp		\
	../shared/HZ_L.hpp		\
	../shared/cuda_helper.hpp	\
	../shared/cuda_memory_helper.hpp\
	../shared/defines.hpp		\
	../shared/my_utils.hpp		\
	HZ_L2.hpp			\
	cuZ.hpp                         \
	device_code.hpp			\
	device_code_accumV.hpp 		\
	device_code_cdsort_$(HCVG).hpp	\
	device_code_nosort_$(HCVG).hpp	\
	device_code_cdsort_accumV.hpp	\
	device_code_nosort_accumV.hpp	\
	device_code_common.hpp		\
	device_code_globals.hpp		\
	device_code_common_rotate.hpp	\
	device_code_common_Kepler.hpp	\
	device_code_common_Cholesky.hpp

SOURCES=			\
	../shared/HZ.cu		\
	../shared/HZ_L.cu	\
	../shared/cuda_helper.cu\
	../shared/my_utils.cu	\
	HZ_L2.cu		\
	device_code.cu

ifdef MPI
NVPPFLAGS+=-DUSE_MPI
ifndef MPIINC
MPIINC=-I$(MPI)/include
endif # !MPIINC
ifndef MPILIB
MPILIB=-Xlinker -rpath,$(MPI)/lib -Xlinker --enable-new-dtags -L$(MPI)/lib -lmpi
endif # !MPILIB
MPIHDR= \
	../shared/mpi_helper.hpp\
	HZ_L3.hpp
MPISRC= \
	../shared/mpi_helper.cu	\
	../shared/mpi_main.cu	\
	HZ_L3.cu
EXE=MHZ$(CVG).exe
else # !MPI
SOURCES+=../shared/main.cu
EXE=HZ$(CVG).exe
endif # MPI

all: $(EXE)

HZ$(CVG).exe: $(HEADERS) $(SOURCES) GNUmakefile
	$(NVCC) $(NVPPFLAGS) $(NVCCFLAGS) $(SOURCES) -o $@ $(NVLDFLAGS)

MHZ$(CVG).exe: $(MPIHDR) $(HEADERS) $(MPISRC) $(SOURCES) GNUmakefile
	$(NVCC) $(MPIINC) $(NVPPFLAGS) $(NVCCFLAGS) $(MPISRC) $(SOURCES) -o $@ $(MPILIB) $(NVLDFLAGS)

clean:
	-$(RM) $(EXE)
	-$(RM) $(EXE).dSYM
